"""Shared configuration for the icosa acoustic transmission protocol.

Both the transmitter (:mod:`icosa.tx`) and the receiver (:mod:`icosa.rx`)
import their numeric configuration and protocol magic numbers from here so
the two sides can never drift apart.

Protocol summary
----------------
Data is split into 2-bit symbols (MSB first). Symbols alternate between two
"slots", A and B, each transmitted as a two-tone burst:

    slot A: tones at BASE and BASE + (sym + 1) * TONE_STEP
    slot B: tones at BASE + SLOT_OFFSET and BASE + SLOT_OFFSET + (sym + 1) * TONE_STEP

A marker tone at BASE + MARKER_OFFSET is emitted after every 4th symbol
(end of each byte).
"""

from __future__ import annotations

# --- SHARED AUDIO CONFIGURATION ---
SAMPLING_RATE = 44100
BASE_FREQ = 16000
SYMBOL_DURATION = 0.15  # seconds per tone burst (TX)

# --- TONE / SLOT LAYOUT ---
# Spacing between the data tones that encode a single symbol.
TONE_STEP = 100
# Frequency shift applied to slot B so A and B bursts never overlap.
SLOT_OFFSET = 500
# Offset of the byte-boundary marker tone from BASE_FREQ.
MARKER_OFFSET = 1000

# --- RX CONFIGURATION ---
CHANNELS = 1
THRESHOLD = 10
RADIX2_EXP = 11
BLOCK_LENGTH = 2**RADIX2_EXP
SLIDE_LENGTH = BLOCK_LENGTH // 2

# --- PROTOCOL SENTINELS ---
# A byte is built from 4 symbols, so a byte boundary falls every 4 symbols.
SYMBOLS_PER_BYTE = 4
# Byte emitted at the end of a message: 0x11 -> symbols [0, 1, 0, 1].
END_OF_MESSAGE = 0x11
# The byte that a short or missing tail is padded to before packing.
SYMBOL_PAD = 0

# --- TX ENVELOPE ---
# Duration of the cosine fade applied to each end of a burst, in seconds.
FADE_DURATION = 0.05
