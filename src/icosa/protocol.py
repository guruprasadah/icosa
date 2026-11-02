"""Wire format for icosa: encoding, symbol layout and frame decoding.

This module is pure (no audio, no asyncio) so it can be exercised headless and
reused by any transport (CLI, FastAPI, ...).

Protocol summary
----------------
Data is split into 2-bit symbols (MSB first). Symbols alternate between two
"slots", A and B, each transmitted as a ``SYMBOL_DURATION`` two-tone burst:

    slot A: tones at BASE_FREQ and BASE_FREQ + (sym + 1) * 100
    slot B: tones at BASE_FREQ + 500 and BASE_FREQ + (sym + 1) * 100 + 500

A marker tone at BASE_FREQ + 1000 is emitted after every 4th symbol (end of
each byte). ``END_BYTE`` (17 == ``0b0101``) terminates a transmission and is
what the receiver looks for to know the stream is complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# --- SHARED CONFIGURATION ---
SAMPLING_RATE = 44100
BASE_FREQ = 16000
SYMBOL_DURATION = 0.15  # seconds per tone burst

SLOT_OFFSET = 500  # slot B is shifted up by this much
TONE_STEP = 100  # symbol n sits at (n + 1) * TONE_STEP above the slot base
MARKER_OFFSET = 1000  # byte-boundary marker tone

# --- RX CONFIGURATION ---
CHANNELS = 1
THRESHOLD = 10
RADIX2_EXP = 11
BLOCK_LENGTH = 2**RADIX2_EXP
SLIDE_LENGTH = BLOCK_LENGTH // 2

# Symbol sent after the payload to flag the end of the transmission (4 symbols
# -> one byte, so the end marker byte is 0b00010101 == 17).
END_BYTE = 17
END_PATTERN = [0, 1, 0, 1]


# -----------------------------------------------------------------------------
# Symbol layout
# -----------------------------------------------------------------------------


def symbol_frequencies(sym: int, *, slot_b: bool) -> list[float]:
    """Tones that carry ``sym`` in slot A (``slot_b=False``) or slot B."""
    base = BASE_FREQ + (SLOT_OFFSET if slot_b else 0)
    return [base, base + (sym + 1) * TONE_STEP]


def marker_frequency() -> int:
    return BASE_FREQ + MARKER_OFFSET


# -----------------------------------------------------------------------------
# Byte <-> 2-bit symbols
# -----------------------------------------------------------------------------


def bytearray_to_bit_pairs(data: bytes | bytearray, msb_first: bool = True) -> list[int]:
    """Converts bytes into a list of 2-bit integers (values 0-3)."""
    if msb_first:
        # Pairs from left to right: bits [7:6], [5:4], [3:2], [1:0]
        return [(byte >> shift) & 0x03 for byte in data for shift in (6, 4, 2, 0)]
    # Pairs from right to left: bits [1:0], [3:2], [5:4], [7:6]
    return [(byte >> shift) & 0x03 for byte in data for shift in (0, 2, 4, 6)]


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


def encode(data: bytes | bytearray) -> list[int]:
    """Payload bytes -> bit pairs, terminated by the end-of-transmission byte."""
    return bytearray_to_bit_pairs(bytes(data) + bytes([END_BYTE]))


def decode(bit_pairs: list[int]) -> bytes:
    """Bit pairs -> payload bytes; drops the trailing end byte if present."""
    raw = bytes(bit_pairs_to_bytes(bit_pairs))
    if raw.endswith(bytes([END_BYTE])):
        return raw[:-1]
    return raw


# -----------------------------------------------------------------------------
# Transmission plan (what TX plays, in order)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Burst:
    """One chunk of audio to play: a symbol burst or a boundary marker."""

    index: int  # symbol index (0-based) this burst belongs to
    slot: str  # "A" or "B" for symbols, "marker" for the boundary tone
    symbol: int | None  # 0-3 for symbols, None for the marker
    tones: tuple[int, ...]
    is_boundary: bool  # True if this burst closes a byte


def plan_transmission(data: bytes | bytearray) -> list[Burst]:
    """Expand payload bytes into the ordered list of bursts to play."""
    bursts: list[Burst] = []
    for ind, sym in enumerate(encode(data)):
        slot_b = bool(ind % 2)
        bursts.append(
            Burst(
                index=ind,
                slot="B" if slot_b else "A",
                symbol=sym,
                tones=tuple(symbol_frequencies(sym, slot_b=slot_b)),
                is_boundary=False,
            )
        )
        if ind % 4 == 3:
            # End of a byte (4 symbols): emit the boundary marker.
            bursts.append(
                Burst(
                    index=ind,
                    slot="marker",
                    symbol=None,
                    tones=(marker_frequency(),),
                    is_boundary=True,
                )
            )
    return bursts


# -----------------------------------------------------------------------------
# Frame decoding (what RX does with each spectrum frame)
# -----------------------------------------------------------------------------


class EventKind(str, Enum):
    LISTENING = "listening"
    SYMBOL = "symbol"
    BOUNDARY = "boundary"
    END = "end"


@dataclass(frozen=True)
class Event:
    """Something worth reporting while receiving or transmitting."""

    kind: EventKind
    message: str
    slot: str | None = None
    symbol: int | None = None
    symbol_index: int | None = None
    # Bytes decoded so far (freshly completed byte, or the whole payload once
    # the end marker is reached). None when no byte completed with this event.
    data: bytes | None = None
    # BOUNDARY only: True when padding was needed because symbols were missed.
    message_resynced: bool = False


# Multiplier Hz -> FFT bin for an rfft of BLOCK_LENGTH samples. The bins of the
# old audioflux BFT matched this spacing, so levels and thresholds are unchanged;
# numpy's rfft just lets the decoder run without audioflux.
_BINS_PER_HZ = BLOCK_LENGTH / SAMPLING_RATE


class FrameDecoder:
    """Turns successive magnitude spectra into symbols, bytes and events.

    Feed it ``BLOCK_LENGTH``-sample frames (50% overlapping, as the audio
    stream produces them). The marker tone resynchronises byte boundaries, the
    ``END_PATTERN`` byte ends the transmission.

    The detection rules are the original ones, unchanged: marker first (which
    pads to a byte boundary and forces the next symbol to slot A), then a slot
    A symbol if the base tone is loud, then a slot B symbol.
    """

    def __init__(self):
        self.bit_pairs: list[int] = []
        self.last_b = True
        self.finished = False
        self._magnitudes: np.ndarray | None = None

    # -- spectra ---------------------------------------------------------------

    @staticmethod
    def magnitudes(frame: np.ndarray) -> np.ndarray:
        """Magnitude spectrum of one ``BLOCK_LENGTH`` frame."""
        return np.abs(np.fft.rfft(frame, n=BLOCK_LENGTH))

    def feed_frame(self, frame: np.ndarray) -> list[Event]:
        return self.feed(self.magnitudes(frame))

    # -- decoding --------------------------------------------------------------

    def _level(self, frequency: float) -> float:
        magnitudes = self._magnitudes
        if magnitudes is None:  # only happens if feed() is called directly
            raise RuntimeError("feed() must be called before _level()")
        return float(magnitudes[int(round(frequency * _BINS_PER_HZ))])

    def feed(self, magnitudes: np.ndarray) -> list[Event]:
        """Decode one magnitude spectrum; returns the events it produced."""
        if self.finished:
            return []
        self._magnitudes = magnitudes
        events: list[Event] = []

        if self._level(marker_frequency()) > THRESHOLD:
            events.extend(self._boundary())
            return events

        if self._level(BASE_FREQ) > THRESHOLD and self.last_b:
            levels = [self._level(BASE_FREQ + k * TONE_STEP) for k in (1, 2, 3, 4)]
            self.bit_pairs.append(levels.index(max(levels)))
            self.last_b = False
            events.append(self._symbol_event("A", self.bit_pairs[-1]))
        elif self._level(BASE_FREQ + SLOT_OFFSET) > THRESHOLD and not self.last_b:
            # Mirrors the original receiver, which compared against
            # BASE + 500 + k*100 for k in (6, 7, 8, 9); kept as-is.
            levels = [
                self._level(BASE_FREQ + SLOT_OFFSET + k * TONE_STEP)
                for k in (6, 7, 8, 9)
            ]
            self.bit_pairs.append(levels.index(max(levels)))
            self.last_b = True
            events.append(self._symbol_event("B", self.bit_pairs[-1]))

        if self.bit_pairs[-4:] == END_PATTERN:
            self.finished = True
            events.append(
                Event(
                    EventKind.END,
                    "end of transmission",
                    symbol_index=len(self.bit_pairs),
                    data=decode(self.bit_pairs),
                )
            )

        return events

    def _boundary(self) -> list[Event]:
        was_aligned = len(self.bit_pairs) % 4 == 0
        self.bit_pairs.extend([0] * (-len(self.bit_pairs) % 4))
        # A byte always ends on a slot B symbol, so the next is slot A.
        self.last_b = True
        return [
            Event(
                EventKind.BOUNDARY,
                "byte boundary",
                symbol_index=len(self.bit_pairs),
                data=decode(self.bit_pairs),
                message_resynced=not was_aligned,
            )
        ]

    def _symbol_event(self, slot: str, sym: int) -> Event:
        index = len(self.bit_pairs) - 1
        complete_byte = len(self.bit_pairs) % 4 == 0
        return Event(
            EventKind.SYMBOL,
            f"slot {slot} symbol {sym}",
            slot=slot,
            symbol=sym,
            symbol_index=index,
            # Byte is complete once four symbols have landed.
            data=decode(self.bit_pairs) if complete_byte else None,
        )

    # -- convenience -----------------------------------------------------------

    def result(self) -> bytes:
        """Everything decoded so far (used when stopping before the end marker)."""
        return decode(self.bit_pairs)
