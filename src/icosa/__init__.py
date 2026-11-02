"""icosa: acoustic digital data transmission prototype.

Protocol summary
----------------
Data is split into 2-bit symbols (MSB first). Symbols alternate between two
"slots", A and B, each transmitted as a 0.15 s two-tone burst:

    slot A: tones at BASE and BASE + (sym + 1) * 100
    slot B: tones at BASE + 500 and BASE + 500 + (sym + 1) * 100

A marker tone at BASE + 1000 is emitted after every 4th symbol (end of each byte)

Usage:
    icosa --tx "hello world"
    icosa --rx            # Ctrl+C to stop
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys
import weakref
from collections.abc import AsyncIterator

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

# --- SHARED CONFIGURATION ---
SAMPLING_RATE = 44100
BASE_FREQ = 16000
SYMBOL_DURATION = 0.15  # seconds per tone burst (TX)

# --- RX CONFIGURATION ---
CHANNELS = 1
THRESHOLD = 10
RADIX2_EXP = 11
BLOCK_LENGTH = 2**RADIX2_EXP
SLIDE_LENGTH = BLOCK_LENGTH // 2


# =============================================================================
# Transmit
# =============================================================================


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

    # Gentle enveloping (50 ms cosine fades)
    fade_samples = int(sample_rate * 0.05)
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


def _open_input_stream(**kwargs) -> sd.InputStream:
    """Create and start an input stream (called from a worker thread)."""
    stream = sd.InputStream(**kwargs)
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
    async with _tx_lock():
        bps = bytearray_to_bit_pairs(data + bytearray([17]))
        a = True
        for ind, bp in enumerate(bps):
            base = BASE_FREQ
            d = base + (bp + 1) * 100
            if not a:
                base += 500
                d += 500
            await play_frequencies_ifft([base, d], duration=SYMBOL_DURATION)
            if ind % 4 == 3:
                # End of a byte (4 symbols): emit the boundary marker. Use
                # BASE_FREQ explicitly since `base` is shifted by 500 on odd
                # (slot B) symbols.
                await play_frequencies_ifft(
                    [BASE_FREQ + 1000], duration=SYMBOL_DURATION
                )
            a = not a


# =============================================================================
# Receive
# =============================================================================


def bit_pairs_to_bytes(bit_pairs: list[int]) -> bytearray:
    """Packs 2-bit symbols (MSB first) into bytes; a short tail is zero-padded."""
    byte_data = bytearray()
    for i in range(0, len(bit_pairs), 4):
        chunk = bit_pairs[i : i + 4]
        byte_val = 0
        for j, pair in enumerate(chunk):
            # Shift each 2-bit pair to its position in the 8-bit byte
            shift = (3 - j) * 2
            byte_val |= (pair & 0b11) << shift
        byte_data.append(byte_val)
    return byte_data


async def receive(
    *, stop_event: asyncio.Event | None = None
) -> AsyncIterator[bytes]:
    """Listen on the microphone and yield decoded payload bytes as they arrive.

    The PortAudio callback runs on a real-time thread and hands blocks to an
    `asyncio.Queue` via `call_soon_threadsafe`; the decoding loop awaits that
    queue, so the event loop stays free while waiting for audio.

    Set `stop_event` to stop gracefully (e.g. on ASGI shutdown); cancelling the
    consumer also closes the stream.
    """
    # Imported lazily so that `--tx` works without audioflux installed/loaded.
    import audioflux as af
    from audioflux.type import SpectralFilterBankScaleType

    loop = asyncio.get_running_loop()
    audio_queue: asyncio.Queue = asyncio.Queue()
    stop_sentinel = object()

    bft = af.BFT(
        num=BLOCK_LENGTH // 2 + 1,
        radix2_exp=RADIX2_EXP,
        samplate=SAMPLING_RATE,
        slide_length=SLIDE_LENGTH,
        scale_type=SpectralFilterBankScaleType.LINEAR,
    )
    freq_arr = bft.get_fre_band_arr()

    def fidx(frequency: int) -> int:
        return np.abs(freq_arr - frequency).argmin()

    def audio_callback(indata, frames, time_info, status):
        if status:
            logger.warning("%s", status)
        try:
            loop.call_soon_threadsafe(audio_queue.put_nowait, indata.copy())
        except RuntimeError:
            # The loop was closed while the stream was shutting down.
            pass

    stream = await asyncio.to_thread(
        _open_input_stream,
        samplerate=SAMPLING_RATE,
        channels=CHANNELS,
        callback=audio_callback,
        blocksize=SLIDE_LENGTH,
    )

    watcher: asyncio.Task | None = None
    if stop_event is not None:

        async def watch_stop() -> None:
            await stop_event.wait()
            audio_queue.put_nowait(stop_sentinel)

        watcher = asyncio.create_task(watch_stop())

    buffer = np.zeros(BLOCK_LENGTH, dtype=np.float32)
    last_b = True
    base = BASE_FREQ
    pending: list[int] = []

    logger.info("Targeting: %s Hz", float(BASE_FREQ))
    logger.info("Listening...")

    try:
        while True:
            chunk = await audio_queue.get()
            if chunk is stop_sentinel:
                return
            data_chunk = chunk.flatten()

            buffer = np.roll(buffer, -len(data_chunk))
            buffer[-len(data_chunk) :] = data_chunk

            spec_arr = bft.bft(buffer)
            magnitudes = np.abs(spec_arr)

            if magnitudes[fidx(base + 1000)] > THRESHOLD:
                # Pad any missed symbols so we land on a byte boundary. This is
                # idempotent, so repeat detections of the same marker are harmless.
                if pending:
                    pending.extend([0] * (-len(pending) % 4))
                    yield bytes(bit_pairs_to_bytes(pending))
                    pending.clear()
                # A byte always ends on a slot B symbol, so the next is slot A.
                last_b = True
                continue

            if magnitudes[fidx(base)] > THRESHOLD and last_b:
                ms = [magnitudes[fidx(base + k * 100)] for k in (1, 2, 3, 4)]
                pending.append(ms.index(max(ms)))
                last_b = False

            if magnitudes[fidx(base + 500)] > THRESHOLD and not last_b:
                ms = [magnitudes[fidx(base + k * 100)] for k in (6, 7, 8, 9)]
                pending.append(ms.index(max(ms)))
                last_b = True

            if len(pending) == 4:
                byte_val = bit_pairs_to_bytes(pending)[0]
                pending.clear()
                if byte_val == 0x11:
                    # End-of-message sentinel ([0, 1, 0, 1] in 2-bit symbols).
                    return
                yield bytes([byte_val])
    finally:
        if watcher is not None:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher
        await asyncio.to_thread(stream.close)


# =============================================================================
# CLI
# =============================================================================


async def _main_async(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="icosa",
        description="Near-ultrasonic acoustic data transmission prototype.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--tx",
        metavar="TEXT",
        help="transmit TEXT (UTF-8) over the speaker",
    )
    mode.add_argument(
        "--rx",
        action="store_true",
        help="listen on the microphone and decode (Ctrl+C to stop)",
    )
    args = parser.parse_args(argv)

    if args.tx is not None:
        await transmit(bytearray(args.tx, encoding="utf-8"))
    else:
        async with contextlib.aclosing(receive()) as chunks:
            async for chunk in chunks:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()


def main(argv: list[str] | None = None) -> None:
    try:
        asyncio.run(_main_async(argv))
    except KeyboardInterrupt:
        # The receive generator's `finally` has already closed the stream.
        pass
