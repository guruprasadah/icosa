"""Asynchronous receiver: turns microphone audio into events as they arrive."""

from __future__ import annotations

import asyncio
import contextlib
import sys
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from .logging import print_event
from .protocol import (
    BLOCK_LENGTH,
    CHANNELS,
    SAMPLING_RATE,
    SLIDE_LENGTH,
    Event,
    EventKind,
    FrameDecoder,
)

# How many audio chunks may queue up before the oldest are dropped. The audio
# callback never blocks; if the consumer falls this far behind, losing audio is
# preferable to stalling the OS audio thread.
_QUEUE_SIZE = 64


@dataclass
class Receiver:
    """Listens on the microphone and streams decoded events.

    Async generator (what a FastAPI/SSE/HTMX endpoint would consume)::

        async with Receiver() as rx:
            async for event in rx:
                ...

    Or callback style, stopping at the end marker::

        await Receiver().listen(on_event=handler)

    Playing and listening on the same machine feeds the speaker output back
    into the microphone; call :meth:`mute_for` around a transmission to stop the
    receiver from decoding its own signal.
    """

    sample_rate: int = SAMPLING_RATE
    device: int | str | None = None

    def __post_init__(self) -> None:
        self._decoder = FrameDecoder()
        self._chunks: list[np.ndarray] = []
        self._wakeup = asyncio.Event()
        self._buffer = np.zeros(BLOCK_LENGTH, dtype=np.float32)
        self._stream: sd.InputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._muted_until = 0.0
        self._stopped = False

    # -- lifecycle -------------------------------------------------------------

    async def __aenter__(self) -> Receiver:
        await self.start()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.stop()

    async def start(self) -> None:
        """Open the input stream and begin queueing audio."""
        if self._stream is not None:
            return
        self._loop = asyncio.get_running_loop()

        def audio_callback(indata, frames, time_info, status) -> None:
            loop = self._loop
            if loop is None:
                return
            if status:
                print(status, file=sys.stderr)
            self._chunks.append(indata.copy())
            while len(self._chunks) > _QUEUE_SIZE:
                self._chunks.pop(0)
            try:
                loop.call_soon_threadsafe(self._notify)
            except RuntimeError:  # loop already closed during shutdown
                pass

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            callback=audio_callback,
            blocksize=SLIDE_LENGTH,
            device=self.device,
        )
        await asyncio.to_thread(stream.start)
        self._stream = stream

    async def stop(self) -> None:
        self._stopped = True
        stream, self._stream = self._stream, None
        if stream is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(stream.stop)
            with contextlib.suppress(Exception):
                await asyncio.to_thread(stream.close)
        self._notify()

    def _notify(self) -> None:
        self._wakeup.set()

    # -- crosstalk -------------------------------------------------------------

    def mute_for(self, seconds: float) -> None:
        """Ignore incoming audio for ``seconds`` (call around a transmission)."""
        if self._loop is not None:
            self._muted_until = max(self._muted_until, self._loop.time() + seconds)

    def _muted(self) -> bool:
        loop = self._loop
        return loop is not None and loop.time() < self._muted_until

    # -- decoding --------------------------------------------------------------

    async def _drain_chunks(self) -> list[Event]:
        events: list[Event] = []
        while self._chunks:
            chunk = self._chunks.pop(0).flatten()
            if self._muted():
                continue
            n = len(chunk)
            self._buffer = np.roll(self._buffer, -n)
            self._buffer[-n:] = chunk
            # Cheap FFT, but keep it off the event loop regardless.
            frame = self._buffer.copy()
            events.extend(await asyncio.to_thread(self._decoder.feed_frame, frame))
        return events

    async def __aiter__(self):
        """Yield events as they decode, until the end marker or ``stop()``."""
        if self._stream is None:
            await self.start()
        try:
            while not self._stopped and not self._decoder.finished:
                for event in await self._drain_chunks():
                    yield event
                if self._decoder.finished or self._stopped:
                    break
                self._wakeup.clear()
                await self._wakeup.wait()
        finally:
            await self.stop()

    async def listen(self, *, on_event=None) -> bytes:
        """Consume until the end marker (or cancellation) and return the payload.

        ``on_event`` may be sync or async and sees every event, including the
        final END event that carries the whole payload.
        """
        try:
            async for event in self:
                if on_event is not None:
                    result = on_event(event)
                    if asyncio.iscoroutine(result):
                        await result
        except asyncio.CancelledError:
            # Stopped by the caller: hand back whatever was decoded so far.
            return self.result()
        return self.result()

    # `await receiver(on_event=...)` is shorthand for `.listen(...)`.
    __call__ = listen

    def result(self) -> bytes:
        """Bytes decoded so far (useful after stopping or cancelling)."""
        return self._decoder.result()


async def receive(*, on_event=None, **kwargs) -> bytes:
    """One-shot helper: listen until the end marker and return the payload."""
    async with Receiver(**kwargs) as rx:
        return await rx.listen(on_event=on_event)


def run_receive() -> None:
    """CLI entry point: listen with pretty logging, Ctrl+C to report progress."""
    receiver = Receiver()

    async def run() -> None:
        print_event(
            Event(EventKind.LISTENING, "Listening\u2026 Press Ctrl+C to stop.")
        )
        payload = await receiver.listen(on_event=print_event)
        if payload:
            print(payload.decode("latin-1"))

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nStream stopped.")
        payload = receiver.result()
        if payload:
            print(payload.decode("latin-1"))
