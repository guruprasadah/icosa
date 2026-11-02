"""Console rendering for protocol events, shared by the CLI and any caller.

Every consumer (CLI, FastAPI, ...) receives structured :class:`Event` objects;
this module turns them into human-friendly lines.
"""

from __future__ import annotations

import sys
from typing import TextIO

from .protocol import END_BYTE, Event, EventKind

_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_MAGENTA = "\033[35m"


def _printable(data: bytes) -> str:
    return data.decode("latin-1").replace("\x1b", "\\x1b")


def format_event(event: Event) -> list[str]:
    """Render one event as one or more pretty (ANSI-coloured) lines."""
    if event.kind is EventKind.LISTENING:
        return [f"{_DIM}{event.message}{_RESET}"]

    if event.kind is EventKind.SYMBOL:
        index = event.symbol_index or 0
        slot_color = _CYAN if event.slot == "A" else _MAGENTA
        lines = [
            f"  {_DIM}#{index:03d}{_RESET} "
            f"{slot_color}slot {event.slot}{_RESET} "
            f"{_DIM}\u2192{_RESET} {_BOLD}{event.symbol}{_RESET}"
        ]
        if event.data:
            text = _printable(event.data[-1:])
            lines.append(
                f"    {_GREEN}byte {len(event.data):03d}{_RESET} "
                f"{_DIM}0x{event.data[-1]:02x}{_RESET}  {text!r}"
            )
        return lines

    if event.kind is EventKind.BOUNDARY:
        suffix = f" {_YELLOW}(resynchronised){_RESET}" if event.message_resynced else ""
        completed = len(event.data or b"")
        return [
            f"{_DIM}\u2500\u2500 {event.message} after {completed} bytes"
            f"{_RESET}{suffix}"
        ]

    if event.kind is EventKind.END:
        payload = event.data or b""
        lines = [
            f"{_GREEN}\u2714 {event.message}{_RESET} "
            f"{_DIM}({len(payload)} bytes, end marker 0x{END_BYTE:02x}){_RESET}"
        ]
        if payload:
            lines.append(f"  {_BOLD}{_printable(payload)!r}{_RESET}")
        return lines

    return [event.message]


def print_event(event: Event, *, file: TextIO | None = None) -> None:
    out = file if file is not None else sys.stdout
    for line in format_event(event):
        print(line, file=out, flush=True)
