"""Reception: listen on the microphone and decode acoustic tone bursts.

Public entry points: :func:`receive` and :func:`receive_events`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from .constants import (
    BASE_FREQ,
    BLOCK_LENGTH,
    CHANNELS,
    END_OF_MESSAGE,
    MARKER_OFFSET,
    RADIX2_EXP,
    SAMPLING_RATE,
    SLIDE_LENGTH,
    SLOT_OFFSET,
    SYMBOLS_PER_BYTE,
    THRESHOLD,
    SYMBOL_PAD,
    TONE_STEP,
)
from .log import setup_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RxSymbol:
    """A single received 2-bit symbol (one bit pair)."""

    slot: str
    pair: int


@dataclass(frozen=True)
class RxByte:
    """A complete received byte (four bit pairs packed MSB-first)."""

    value: int
    symbols: tuple[int, ...]


@dataclass(frozen=True)
class RxMarker:
    """A detected byte-boundary marker."""

    pass


@dataclass(frozen=True)
class RxEnd:
    """Sentinel yielded once when the end-of-message marker is decoded."""

    pass


RxEvent = RxSymbol | RxByte | RxMarker | RxEnd


def _open_input_stream(**kwargs) -> sd.InputStream:
    """Create and start an input stream (called from a worker thread)."""
    stream = sd.InputStream(**kwargs)
    try:
        stream.start()
    except BaseException:
        stream.close(ignore_errors=True)
        raise
    return stream


def bit_pairs_to_bytes(bit_pairs: list[int]) -> bytearray:
    """Packs 2-bit symbols (MSB first) into bytes; a short tail is zero-padded."""
    byte_data = bytearray()
    for i in range(0, len(bit_pairs), SYMBOLS_PER_BYTE):
        chunk = bit_pairs[i : i + SYMBOLS_PER_BYTE]
        byte_val = 0
        for j, pair in enumerate(chunk):
            # Shift each 2-bit pair to its position in the 8-bit byte
            shift = (SYMBOLS_PER_BYTE - 1 - j) * 2
            byte_val |= (pair & 0b11) << shift
        byte_data.append(byte_val)
    return byte_data


async def receive_events(
    *, stop_event: asyncio.Event | None = None
) -> AsyncIterator[RxEvent]:
    """Listen on the microphone and yield structured decode events as they occur.

    Yields :class:`RxSymbol` for each bit pair, :class:`RxByte` for each
    completed byte, :class:`RxMarker` on byte boundaries, and :class:`RxEnd`
    when the end-of-message sentinel is decoded. This lets a UI render the
    three-stage view (bit pairs -> bytes -> assembled output) directly.

    The PortAudio callback runs on a real-time thread and hands blocks to an
    `asyncio.Queue` via `call_soon_threadsafe`; the decoding loop awaits that
    queue, so the event loop stays free while waiting for audio.

    Set `stop_event` to stop gracefully (e.g. on ASGI shutdown); cancelling the
    consumer also closes the stream.
    """
    setup_logging(logger_name=__name__)
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
    pending: list[RxSymbol] = []

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

            if magnitudes[fidx(base + MARKER_OFFSET)] > THRESHOLD:
                # Pad any missed symbols so we land on a byte boundary. This is
                # idempotent, so repeat detections of the same marker are harmless.
                if pending:
                    pairs = [s.pair for s in pending]
                    pairs.extend([SYMBOL_PAD] * (-len(pairs) % SYMBOLS_PER_BYTE))
                    yield RxByte(
                        value=bit_pairs_to_bytes(pairs)[0],
                        symbols=tuple(pairs),
                    )
                    pending.clear()
                yield RxMarker()
                # A byte always ends on a slot B symbol, so the next is slot A.
                last_b = True
                continue

            if magnitudes[fidx(base)] > THRESHOLD and last_b:
                ms = [magnitudes[fidx(base + k * TONE_STEP)] for k in (1, 2, 3, 4)]
                symbol = RxSymbol(slot="A", pair=ms.index(max(ms)))
                pending.append(symbol)
                yield symbol
                last_b = False

            if magnitudes[fidx(base + SLOT_OFFSET)] > THRESHOLD and not last_b:
                ms = [
                    magnitudes[fidx(base + SLOT_OFFSET + k * TONE_STEP)]
                    for k in (1, 2, 3, 4)
                ]
                symbol = RxSymbol(slot="B", pair=ms.index(max(ms)))
                pending.append(symbol)
                yield symbol
                last_b = True

            if len(pending) == SYMBOLS_PER_BYTE:
                byte_val = bit_pairs_to_bytes([s.pair for s in pending])[0]
                byte_symbols = tuple(s.pair for s in pending)
                pending.clear()
                if byte_val == END_OF_MESSAGE:
                    # End-of-message sentinel ([0, 1, 0, 1] in 2-bit symbols).
                    yield RxEnd()
                    return
                yield RxByte(value=byte_val, symbols=byte_symbols)
    finally:
        if watcher is not None:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher
        await asyncio.to_thread(stream.close)


async def receive(
    *, stop_event: asyncio.Event | None = None
) -> AsyncIterator[bytes]:
    """Listen on the microphone and yield decoded payload bytes as they arrive.

    Thin wrapper over :func:`receive_events` that surfaces only payload bytes
    (preserving the original CLI-facing behaviour).
    """
    async for event in receive_events(stop_event=stop_event):
        if isinstance(event, RxByte):
            yield bytes([event.value])
        elif isinstance(event, RxEnd):
            return
