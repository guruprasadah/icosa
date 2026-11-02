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
import sys

from .rx import receive, receive_events
from .tx import build_transmit_plan, transmit, transmit_stream

__all__ = [
    "transmit",
    "transmit_stream",
    "build_transmit_plan",
    "receive",
    "receive_events",
    "main",
    "__version__",
]

__version__ = "0.1.0"


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
