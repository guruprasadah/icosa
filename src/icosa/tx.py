"""Asynchronous transmitter: plays a payload as near-ultrasonic bursts."""

from __future__ import annotations

import asyncio
import contextlib

import numpy as np
import sounddevice as sd

from .logging import print_event
from .protocol import (
    SAMPLING_RATE,
    SYMBOL_DURATION,
    Burst,
    Event,
    EventKind,
    plan_transmission,
)

# Silence before the first burst, and after the last, so the receiver has room
# to settle and the tail of the final marker is not truncated on close.
_LEAD_IN = 0.05
_DRAIN = 0.25


def build_tone(
    frequencies, duration: float = 2.0, sample_rate: int = SAMPLING_RATE
) -> np.ndarray:
    """Multi-tone signal built with an inverse FFT (no I/O, safe to call anywhere)."""
    # 1. Calculate total number of samples
    n_samples = int(sample_rate * duration)

    # 2. Initialize an empty frequency domain spectrum (complex numbers)
    spectrum = np.zeros(n_samples, dtype=complex)

    # 3. Map frequencies to their closest spectral bin index
    for freq in frequencies:
        bin_idx = int(round(freq * n_samples / sample_rate))

        # Keep frequencies below the Nyquist limit
        if 0 <= bin_idx < n_samples // 2:
            # Scale by n_samples / 2 because IFFT implicitly divides by n_samples.
            spectrum[bin_idx] = n_samples / 2
            # For a strictly real signal, negative frequencies are conjugates.
            spectrum[-bin_idx] = np.conj(spectrum[bin_idx])

    # 4. Transform frequency domain back to the time domain
    audio_signal = np.fft.ifft(spectrum).real

    # Gentle enveloping (50 ms cosine fades)
    fade_samples = min(int(sample_rate * 0.05), n_samples // 2)
    envelope = np.ones(n_samples)
    ramp = np.arange(fade_samples) / fade_samples
    envelope[:fade_samples] = 0.5 * (1 - np.cos(np.pi * ramp))  # 0 -> 1
    envelope[-fade_samples:] = 0.5 * (1 + np.cos(np.pi * ramp))  # 1 -> 0
    audio_signal *= envelope

    # 5. Normalize to prevent clipping (-1.0 to 1.0)
    max_val = np.max(np.abs(audio_signal))
    if max_val > 0:
        audio_signal = audio_signal / max_val

    return audio_signal.astype(np.float32)


def build_burst(burst: Burst, duration: float = SYMBOL_DURATION) -> np.ndarray:
    return build_tone(burst.tones, duration=duration)


async def transmit(
    data: bytes | bytearray,
    *,
    on_event=None,
    sample_rate: int = SAMPLING_RATE,
    device: int | str | None = None,
    receiver=None,
) -> int:
    """Play ``data`` as a full transmission; returns the number of bursts sent.

    ``on_event`` may be a sync or async callable taking an :class:`Event`; it
    is called before each burst is played and once when playback finishes.
    Pass ``receiver`` (an :class:`icosa.rx.Receiver`) to mute it for the length
    of the transmission so it does not decode its own speaker output.
    Cancelling the task stops playback immediately and closes the stream.
    """
    bursts = plan_transmission(data)

    async def emit(event: Event) -> None:
        if on_event is None:
            return
        result = on_event(event)
        if asyncio.iscoroutine(result):
            await result

    # Longer than needed so extra writes never block and the event loop stays
    # free to sleep between bursts (the pacing gap exceeds the buffer).
    stream = sd.OutputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
        latency="high",
    )
    mute_for = None if receiver is None else receiver.mute_for

    try:
        await asyncio.to_thread(stream.start)
        await asyncio.to_thread(
            stream.write, np.zeros(int(sample_rate * _LEAD_IN), dtype=np.float32)
        )
        if mute_for is not None:
            mute_for(_LEAD_IN + len(bursts) * SYMBOL_DURATION + _DRAIN + 1)

        for n, burst in enumerate(bursts, start=1):
            await emit(
                Event(
                    EventKind.SYMBOL,
                    f"burst {n}/{len(bursts)}: slot {burst.slot}",
                    slot=burst.slot,
                    symbol=burst.symbol,
                    symbol_index=burst.index,
                )
            )
            chunk = await asyncio.to_thread(build_burst, burst, SYMBOL_DURATION)
            await asyncio.to_thread(stream.write, chunk)
            await asyncio.sleep(SYMBOL_DURATION)

        await asyncio.to_thread(
            stream.write, np.zeros(int(sample_rate * _DRAIN), dtype=np.float32)
        )
    finally:
        with contextlib.suppress(Exception):
            await asyncio.to_thread(stream.stop)
        with contextlib.suppress(Exception):
            await asyncio.to_thread(stream.close)

    await emit(Event(EventKind.END, "transmission complete", data=bytes(data)))
    return len(bursts)


def run_transmit(data: bytes | bytearray) -> None:
    """CLI entry point: transmit with pretty logging on stdout."""

    async def on_event(event: Event) -> None:
        print_event(event)

    try:
        asyncio.run(transmit(data, on_event=on_event))
    except KeyboardInterrupt:
        print("\nTransmission cancelled.")
