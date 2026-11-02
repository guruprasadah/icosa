"""Transmission: encode bytes into acoustic tone bursts and play them.

Public entry point: :func:`transmit`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import weakref

import numpy as np
import sounddevice as sd

from .constants import (
    BASE_FREQ,
    END_OF_MESSAGE,
    FADE_DURATION,
    MARKER_OFFSET,
    SAMPLING_RATE,
    SLOT_OFFSET,
    SYMBOL_DURATION,
    TONE_STEP,
)
from .log import setup_logging

logger = logging.getLogger(__name__)


def render_burst(
    frequencies, duration=SYMBOL_DURATION, sample_rate=SAMPLING_RATE
) -> np.ndarray:
    """
    Renders a multi-frequency burst using Inverse FFT.

    This is pure CPU work and is safe to call directly from an event loop.

    :param frequencies: List or array of target frequencies in Hz
    :param duration: Duration of the audio in seconds
    :param sample_rate: Audio sampling rate (default 44100 Hz)
    :returns: A one-dimensional float array containing the audio samples
    """
    # 1. Calculate total number of samples
    n_samples = int(sample_rate * duration)

    # 2. Initialize an empty frequency domain spectrum (complex numbers)
    spectrum = np.zeros(n_samples, dtype=complex)

    # 3. Map frequencies to their closest spectral bin index
    # Formula: bin_index = frequency * total_samples / sample_rate
    for freq in frequencies:
        bin_idx = int(round(freq * n_samples / sample_rate))

        # Keep frequencies below the Nyquist limit
        if 0 <= bin_idx < n_samples // 2:
            # Scale by n_samples / 2 because IFFT implicitly divides by n_samples.
            spectrum[bin_idx] = n_samples / 2

            # For a strictly real time-domain signal, negative frequencies must
            # be the complex conjugate of the positive ones.
            spectrum[-bin_idx] = np.conj(spectrum[bin_idx])

    # 4. Transform frequency domain back to the time domain using IFFT
    time_signal = np.fft.ifft(spectrum)

    # Extract the real part (the imaginary part should be practically zero)
    audio_signal = time_signal.real

    # Gentle enveloping (cosine fades)
    fade_samples = int(sample_rate * FADE_DURATION)
    fade_samples = min(fade_samples, n_samples // 2)  # don't consume whole signal

    envelope = np.ones(n_samples)
    ramp = np.arange(fade_samples) / fade_samples
    envelope[:fade_samples] = 0.5 * (1 - np.cos(np.pi * ramp))  # 0 -> 1
    envelope[-fade_samples:] = 0.5 * (1 + np.cos(np.pi * ramp))  # 1 -> 0

    audio_signal *= envelope

    # 5. Normalize the signal to prevent clipping (-1.0 to 1.0)
    max_val = np.max(np.abs(audio_signal))
    if max_val > 0:
        audio_signal = audio_signal / max_val

    return audio_signal


def _open_output_stream(**kwargs) -> sd.OutputStream:
    """Create and start an output stream (called from a worker thread)."""
    stream = sd.OutputStream(**kwargs)
    try:
        stream.start()
    except BaseException:
        stream.close(ignore_errors=True)
        raise
    return stream


async def _play(signal: np.ndarray, sample_rate: int) -> None:
    """Play an audio buffer and await its completion without blocking the loop."""
    loop = asyncio.get_running_loop()
    done = asyncio.Event()
    signal = np.asarray(signal, dtype=np.float32)
    position = 0

    def callback(outdata, frames, time_info, status):
        nonlocal position
        if status:
            logger.warning("%s", status)

        remaining = len(signal) - position
        n = min(frames, remaining)
        if n > 0:
            outdata[:n, 0] = signal[position : position + n]
            position += n

        # The callback must always fill the whole buffer. Once the signal is
        # exhausted, pad with silence and tell PortAudio to play out what it has
        # and then finish (CallbackStop, not CallbackAbort, avoids truncation).
        if n < frames:
            outdata[n:, 0] = 0.0
            raise sd.CallbackStop

    def finished() -> None:
        # Invoked from PortAudio's thread; hop back onto the event loop.
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(done.set)

    stream = await asyncio.to_thread(
        _open_output_stream,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=callback,
        finished_callback=finished,
    )
    try:
        await done.wait()
    finally:
        await asyncio.to_thread(stream.close)


async def play_frequencies_ifft(
    frequencies, duration=2.0, sample_rate=SAMPLING_RATE
) -> None:
    """
    Generates a multi-frequency signal using Inverse FFT and plays it.

    Rendering happens synchronously on the event loop (it is short CPU work);
    playback is awaited via the audio stream's finished callback.

    :param frequencies: List or array of target frequencies in Hz
    :param duration: Duration of the audio in seconds
    :param sample_rate: Audio sampling rate (default 44100 Hz)
    """
    signal = render_burst(frequencies, duration, sample_rate)
    await _play(signal, sample_rate)


def bytearray_to_bit_pairs(data: bytearray, msb_first: bool = True) -> list[int]:
    """Converts a bytearray into a list of 2-bit integers (values 0-3)."""
    if msb_first:
        # Pairs from left to right: bits [7:6], [5:4], [3:2], [1:0]
        return [(byte >> shift) & 0x03 for byte in data for shift in (6, 4, 2, 0)]
    else:
        # Pairs from right to left: bits [1:0], [3:2], [5:4], [7:6]
        return [(byte >> shift) & 0x03 for byte in data for shift in (0, 2, 4, 6)]


# Playback is a process-global resource (`sd.play` stops any previous playback),
# so transmissions must be serialized. Keep one lock per event loop so repeated
# `asyncio.run()` calls (tests, CLI invocations) don't share a bound loop.
_tx_locks: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _tx_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _tx_locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _tx_locks[loop] = lock
    return lock


async def transmit(data: bytearray) -> None:
    """Encode and play ``data`` as a sequence of acoustic tone bursts."""
    setup_logging(logger_name=__name__)
    async with _tx_lock():
        payload = data + bytearray([END_OF_MESSAGE])
        bps = bytearray_to_bit_pairs(payload)
        logger.info("Transmitting %d bytes (%d symbols)", len(data), len(bps))
        a = True
        for ind, bp in enumerate(bps):
            base = BASE_FREQ
            d = base + (bp + 1) * TONE_STEP
            if not a:
                base += SLOT_OFFSET
                d += SLOT_OFFSET
            logger.debug("symbol %d: slot %s tones %d/%d Hz", ind, "A" if a else "B", base, d)
            await play_frequencies_ifft([base, d], duration=SYMBOL_DURATION)
            if ind % 4 == 3:
                # End of a byte (4 symbols): emit the boundary marker. Use
                # BASE_FREQ explicitly since `base` is shifted by SLOT_OFFSET on
                # odd (slot B) symbols.
                await play_frequencies_ifft(
                    [BASE_FREQ + MARKER_OFFSET], duration=SYMBOL_DURATION
                )
            a = not a
        logger.info("Transmission complete")
