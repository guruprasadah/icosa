"""FastAPI application: landing page, transmitter, receiver and SSE streams.

The audio work reuses the existing :mod:`icosa.tx` / :mod:`icosa.rx` event
generators verbatim; this module only adapts them into HTTP responses and
Server-Sent Events.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..constants import (
    BASE_FREQ,
    END_OF_MESSAGE,
    MARKER_OFFSET,
    SYMBOLS_PER_BYTE,
)
from ..rx import RxByte, RxEnd, RxMarker, RxSymbol, receive_events
from ..tx import TxDone, TxMarker, TxSymbol, build_transmit_plan, transmit_stream
from .broadcast import Broadcaster

logger = logging.getLogger(__name__)

_WEB_DIR = Path(__file__).parent
_TEMPLATES = Jinja2Templates(directory=str(_WEB_DIR / "templates"))
# Protocol constants surfaced to the templates so the copy never drifts.
_PROTOCOL = {
    "base_freq": BASE_FREQ,
    "marker_freq": BASE_FREQ + MARKER_OFFSET,
    "symbols_per_byte": SYMBOLS_PER_BYTE,
    "end_of_message": END_OF_MESSAGE,
}

# How long a receiver keeps the mic open after the last SSE client leaves,
# so a page reload / tab bounce doesn't tear the audio stream down and up.
_RX_GRACE_SECONDS = 3.0


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _printable(value: int) -> str | None:
    return chr(value) if 32 <= value < 127 else None


def _group_plan(events: list) -> list[dict]:
    """Group a transmit plan into per-byte blocks for the preview template.

    Returns a list of ``{"index": int, "symbols": [TxSymbol, ...],
    "marker": TxMarker | None, "is_eom": bool}`` dicts, one per byte.
    """
    blocks: list[dict] = []
    current: dict | None = None
    for event in events:
        if isinstance(event, TxSymbol):
            if current is None or event.byte_index != current["index"]:
                current = {
                    "index": event.byte_index,
                    "symbols": [],
                    "marker": None,
                    "is_eom": event.is_end_of_message,
                }
                blocks.append(current)
            current["symbols"].append(event)
        elif isinstance(event, TxMarker):
            if current is not None:
                current["marker"] = event
    return blocks


class TransmitterState:
    """Owns the transmit task and broadcasts per-burst progress.

    The UI flow is: POST /tx/start stores the payload, then the browser opens
    /tx/events. To avoid losing the first symbols to a subscription race, the
    actual playback task is launched by the SSE endpoint (the sole subscriber),
    not by /tx/start itself.
    """

    def __init__(self) -> None:
        self.task: asyncio.Task | None = None
        self.broadcaster = Broadcaster("tx")
        self.running = False
        self.pending: bytearray | None = None

    def request(self, data: bytearray) -> bool:
        """Queue a payload for playback; False if one is already running."""
        if self.task is not None and not self.task.done():
            return False
        self.pending = data
        return True

    def launch(self) -> None:
        """Start playing a queued payload (called once a client subscribes)."""
        data = self.pending
        if data is None or (self.task is not None and not self.task.done()):
            return
        self.pending = None
        self.running = True
        self.task = asyncio.create_task(self._run(data))

    async def _run(self, data: bytearray) -> None:
        try:
            self.broadcaster.publish({"type": "started", "bytes": len(data)})
            async for event in transmit_stream(data):
                if isinstance(event, TxDone):
                    continue
                if isinstance(event, TxMarker):
                    self.broadcaster.publish(
                        {"type": "marker", "index": event.index}
                    )
                else:
                    self.broadcaster.publish(
                        {
                            "type": "symbol",
                            "index": event.index,
                            "slot": event.slot,
                            "pair": event.pair,
                        }
                    )
            self.broadcaster.publish({"type": "done"})
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - hardware dependent
            logger.exception("Transmit task failed")
            self.broadcaster.publish({"type": "error"})
        finally:
            self.running = False


class ReceiverState:
    """Owns the microphone task, the event hub, and the accumulated text."""

    def __init__(self) -> None:
        self.task: asyncio.Task | None = None
        self.stop_event: asyncio.Event | None = None
        self.broadcaster = Broadcaster("rx")
        self.running = False
        self.assembled: str = ""
        # True once an end-of-message tone has been decoded; prevents an
        # automatic reconnect from reopening the mic or wiping the output.
        self.finished = False
        self._stop_timer: asyncio.Task | None = None

    # --- lifecycle ---------------------------------------------------------
    def start(self, *, fresh: bool = False) -> None:
        """Start listening; no-op if already running.

        :param fresh: Clear any previously assembled output and end-of-message
            latch. Set for an explicit user Start; automatic starts triggered by
            a client connecting preserve the current output.
        """
        if self._stop_timer is not None:
            self._stop_timer.cancel()
            self._stop_timer = None
        if self.task is not None and not self.task.done():
            return
        # A finished session stays finished until the user asks to start again.
        if self.finished and not fresh:
            return
        if fresh:
            self.broadcaster = Broadcaster("rx")
            self.assembled = ""
            self.finished = False
        self.stop_event = asyncio.Event()
        self.running = True
        self.task = asyncio.create_task(self._run(self.stop_event))

    def stop(self) -> None:
        """Signal the listen loop to finish."""
        if self._stop_timer is not None:
            self._stop_timer.cancel()
            self._stop_timer = None
        if self.stop_event is not None:
            self.stop_event.set()

    def stop_when_idle(self) -> None:
        """Stop after a grace period unless a new client subscribes."""

        async def _delayed() -> None:
            await asyncio.sleep(_RX_GRACE_SECONDS)
            if self.broadcaster.subscriber_count == 0:
                self.stop()

        if self.finished:
            return
        if self._stop_timer is not None:
            self._stop_timer.cancel()
        self._stop_timer = asyncio.ensure_future(_delayed())

    async def _run(self, stop_event: asyncio.Event) -> None:
        try:
            async for event in receive_events(stop_event=stop_event):
                self._handle(event)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - hardware dependent
            logger.exception("Receiver task failed")
            self.broadcaster.publish({"type": "error"})
        finally:
            self.running = False
            self.broadcaster.publish({"type": "stopped"})

    def _handle(self, event: object) -> None:
        if isinstance(event, RxSymbol):
            self.broadcaster.publish(
                {"type": "symbol", "slot": event.slot, "pair": event.pair}
            )
        elif isinstance(event, RxByte):
            char = _printable(event.value)
            self.broadcaster.publish(
                {
                    "type": "byte",
                    "value": event.value,
                    "symbols": list(event.symbols),
                    "char": char,
                }
            )
            if char is not None:
                self.assembled += char
                self.broadcaster.publish(
                    {"type": "assembled", "text": self.assembled}
                )
        elif isinstance(event, RxMarker):
            self.broadcaster.publish({"type": "marker"})
        elif isinstance(event, RxEnd):
            # The end tone was received: latch so we don't auto-restart and
            # keep the assembled message on screen.
            self.finished = True
            self.broadcaster.publish({"type": "end", "text": self.assembled})

    def status(self) -> dict:
        return {"type": "status", "running": self.running}


def create_app() -> FastAPI:
    """Build the FastAPI application (used by ``icosa --serve``)."""
    app = FastAPI(title="icosa web UI")
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR / "static")), name="static")
    tx = TransmitterState()
    rx = ReceiverState()

    @app.on_event("shutdown")
    async def _shutdown() -> None:  # pragma: no cover - lifecycle hook
        for task in (rx.task, tx.task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    # --- pages -------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    async def landing(request: Request) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(
            request, "index.html", {"protocol": _PROTOCOL}
        )

    @app.get("/tx", response_class=HTMLResponse)
    async def transmitter_page(request: Request) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(
            request, "transmitter.html", {"protocol": _PROTOCOL}
        )

    @app.get("/rx", response_class=HTMLResponse)
    async def receiver_page(request: Request) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(
            request, "receiver.html", {"protocol": _PROTOCOL}
        )

    # --- transmitter -------------------------------------------------------
    @app.post("/tx/plan", response_class=HTMLResponse)
    async def tx_plan(request: Request) -> HTMLResponse:
        """Render the split-up transmission plan for the supplied text."""
        form = await request.form()
        text = str(form.get("text", ""))
        data = bytearray(text, encoding="utf-8")
        events = build_transmit_plan(data)
        blocks = _group_plan(events)
        return _TEMPLATES.TemplateResponse(
            request,
            "partials/tx_plan.html",
            {
                "blocks": blocks,
                "text": text,
                "n_bytes": len(data),
                "n_symbols": sum(1 for e in events if isinstance(e, TxSymbol)),
                "protocol": _PROTOCOL,
            },
        )

    @app.post("/tx/start")
    async def tx_start(request: Request) -> Response:
        form = await request.form()
        text = str(form.get("text", ""))
        if not text:
            return Response(status_code=204)
        queued = tx.request(bytearray(text, encoding="utf-8"))
        event = "tx-started" if queued else "tx-busy"
        return Response(status_code=204, headers={"HX-Trigger": event})

    @app.get("/tx/events")
    async def tx_events() -> StreamingResponse:
        """Stream live transmission progress as SSE.

        Launching the playback here (rather than in /tx/start) guarantees this
        sole subscriber is registered before the first symbol is broadcast.
        """

        async def gen() -> AsyncIterator[str]:
            async with tx.broadcaster.subscribe():
                if tx.pending is None and not tx.running:
                    yield _sse({"type": "idle"})
                    return
                tx.launch()
                async for event in tx.broadcaster.stream():
                    yield _sse(event)
                    if event["type"] in ("done", "error"):
                        return

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # --- receiver ----------------------------------------------------------
    @app.get("/rx/status", response_class=HTMLResponse)
    async def rx_status(request: Request) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(
            request, "partials/rx_status.html", {"running": rx.running}
        )

    @app.post("/rx/start", response_class=HTMLResponse)
    async def rx_start(request: Request) -> HTMLResponse:
        rx.start(fresh=True)
        return _TEMPLATES.TemplateResponse(
            request, "partials/rx_status.html", {"running": rx.running}
        )

    @app.post("/rx/stop", response_class=HTMLResponse)
    async def rx_stop(request: Request) -> HTMLResponse:
        rx.stop()
        return _TEMPLATES.TemplateResponse(
            request, "partials/rx_status.html", {"running": False}
        )

    @app.get("/rx/events")
    async def rx_events() -> StreamingResponse:
        """Stream decoder events (symbols, bytes, assembled text) as SSE."""
        # Auto-start the mic when a client connects (the UI's on-load contract),
        # unless we already finished a message and are waiting for the user to
        # explicitly start a new session.
        rx.start()

        async def gen() -> AsyncIterator[str]:
            async with rx.broadcaster.subscribe():
                # Replay current assembled text so a reload shows prior output.
                yield _sse({"type": "assembled", "text": rx.assembled})
                yield _sse(rx.status())
                if rx.finished and not rx.running:
                    # Hold the display steady; nothing more will arrive until
                    # the user presses Start again.
                    yield _sse({"type": "end", "text": rx.assembled})
                    return
                try:
                    async for event in rx.broadcaster.stream():
                        yield _sse(event)
                        if event["type"] in ("stopped", "error"):
                            return
                finally:
                    # Client disconnected: close the mic after a short grace
                    # period unless a new tab subscribes (handles reloads).
                    rx.stop_when_idle()

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
