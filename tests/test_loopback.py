"""End-to-end loopback tests for the TX/RX event refactor.

These drive the receiver with a synthesized transmission rather than real
hardware, so they run offline and deterministically. See
:mod:`tests.fakesounddevice`.
"""

from __future__ import annotations

import asyncio

import pytest

from icosa.rx import RxByte, RxEnd, RxMarker, RxSymbol, receive, receive_events
from icosa.tx import TxDone, TxMarker, TxSymbol, build_transmit_plan, transmit_stream
from tests.fakesounddevice import install, set_input_signal, synthesize_signal


@pytest.fixture(autouse=True)
def _fake_audio():
    install()
    yield


def test_build_transmit_plan_structure():
    """The plan itemizes 4 symbols + 1 marker per byte, ending with the EOM byte."""
    plan = build_transmit_plan(bytearray(b"Hi"))

    symbols = [e for e in plan if isinstance(e, TxSymbol)]
    markers = [e for e in plan if isinstance(e, TxMarker)]

    # "Hi" -> 2 payload bytes + 1 end-of-message sentinel byte => 12 symbols.
    assert len(symbols) == 12
    # One marker per byte (3 bytes).
    assert len(markers) == 3

    # Slots strictly alternate A/B.
    assert [s.slot for s in symbols] == ["A", "B"] * 6

    # The EOM sentinel byte is [0, 1, 0, 1].
    eom = [s for s in symbols if s.is_end_of_message]
    assert [s.pair for s in eom] == [0, 1, 0, 1]

    # Each symbol carries the correct two-tone frequencies and byte grouping.
    first = symbols[0]
    assert (first.freq_a, first.freq_b) == (16000, 16200)  # pair 1, slot A
    assert {s.byte_index for s in symbols} == {0, 1, 2}


def test_transmit_stream_events():
    """transmit_stream plays audio and yields symbol/marker/done events in order."""

    async def run():
        events = [e async for e in transmit_stream(bytearray(b"Hi"))]
        symbols = [e for e in events if isinstance(e, TxSymbol)]
        markers = [e for e in events if isinstance(e, TxMarker)]
        done = [e for e in events if isinstance(e, TxDone)]
        assert [s.pair for s in symbols] == [1, 0, 2, 0, 1, 2, 2, 1, 0, 1, 0, 1]
        assert len(markers) == 3
        assert len(done) == 1
        assert events[-1] is done[0]

    asyncio.run(run())


def test_receive_events_loopback():
    """A synthesized transmission decodes into bit pairs, bytes, and EOM."""

    async def run():
        set_input_signal(synthesize_signal(bytearray(b"Hi")))
        symbols: list[RxSymbol] = []
        bytes_received = bytearray()
        markers = 0
        ended = False
        async for event in receive_events():
            if isinstance(event, RxSymbol):
                symbols.append(event)
            elif isinstance(event, RxByte):
                bytes_received.append(event.value)
            elif isinstance(event, RxMarker):
                markers += 1
            elif isinstance(event, RxEnd):
                ended = True
        assert len(symbols) == 12
        assert bytes_received == b"Hi"
        assert ended
        assert markers >= 3

    asyncio.run(run())


def test_receive_wrapper_yields_only_bytes():
    """The bytes-only wrapper preserves the original CLI-facing contract."""

    async def run():
        set_input_signal(synthesize_signal(bytearray(b"Hi")))
        out = bytearray()
        async for chunk in receive():
            out.extend(chunk)
        assert out == b"Hi"

    asyncio.run(run())
