"""icosa: acoustic digital data transmission prototype.

Data is split into 2-bit symbols (MSB first) and played as two-tone bursts,
alternating between two slots, with a marker tone after every byte. Both
directions are asyncio-first so they can drive a UI (FastAPI, SSE, HTMX...).

    from icosa import Receiver, transmit

    async with Receiver() as rx:
        async for event in rx:
            ...            # push event to the UI as bytes arrive

    await transmit(b"hello world")

See :mod:`icosa.protocol` for the wire format. CLI equivalent:

    icosa --tx "hello world"
    icosa --rx            # Ctrl+C to stop and print decoded text
"""

import argparse

from .protocol import (
    BASE_FREQ,
    BLOCK_LENGTH,
    CHANNELS,
    MARKER_OFFSET,
    RADIX2_EXP,
    SAMPLING_RATE,
    SLIDE_LENGTH,
    SYMBOL_DURATION,
    THRESHOLD,
    Burst,
    Event,
    EventKind,
    FrameDecoder,
    encode,
    decode,
    plan_transmission,
    symbol_frequencies,
)
from .logging import format_event, print_event
from .rx import Receiver, receive, run_receive
from .tx import build_burst, build_tone, run_transmit, transmit

__all__ = [
    # protocol
    "BASE_FREQ",
    "BLOCK_LENGTH",
    "CHANNELS",
    "MARKER_OFFSET",
    "RADIX2_EXP",
    "SAMPLING_RATE",
    "SLIDE_LENGTH",
    "SYMBOL_DURATION",
    "THRESHOLD",
    "Burst",
    "Event",
    "EventKind",
    "FrameDecoder",
    "encode",
    "decode",
    "plan_transmission",
    "symbol_frequencies",
    # logging
    "format_event",
    "print_event",
    # transmitter
    "transmit",
    "build_tone",
    "build_burst",
    "run_transmit",
    # receiver
    "Receiver",
    "receive",
    "run_receive",
    # CLI
    "main",
]


def main(argv: list[str] | None = None) -> None:
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
        help="listen on the microphone and decode (Ctrl+C to stop and print)",
    )
    args = parser.parse_args(argv)

    if args.tx is not None:
        run_transmit(bytearray(args.tx, encoding="utf-8"))
    else:
        run_receive()
