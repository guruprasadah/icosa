"""Fake sounddevice backends for offline loopback testing.

Rather than touching real microphones/speakers, these classes stand in for
`sounddevice.InputStream` and `sounddevice.OutputStream`. A synthesized
transmission signal set via :func:`set_input_signal` is fed to the receiver
through the fake input stream, and the transmitter's rendered audio is consumed
by the fake output stream.
"""

from __future__ import annotations

import numpy as np
import sounddevice as sd

from icosa.constants import SLIDE_LENGTH, SYMBOL_DURATION
from icosa.tx import TxMarker, TxSymbol, build_transmit_plan, render_burst

_INPUT_SIGNAL: np.ndarray = np.zeros(0, dtype=np.float32)


def set_input_signal(signal: np.ndarray) -> None:
    """Set the waveform the fake input stream will deliver to the receiver."""
    global _INPUT_SIGNAL
    _INPUT_SIGNAL = signal


def synthesize_signal(data: bytearray, pad_seconds: float = 0.3) -> np.ndarray:
    """Render the full transmission waveform (bursts + markers) back to back.

    :param data: Payload bytes to transmit (end-of-message sentinel is included
        automatically by :func:`build_transmit_plan`).
    :param pad_seconds: Seconds of silence appended before and after the signal
        so the receiver's sliding filter banks warm up and drain on quiet.
    """
    plan = build_transmit_plan(data)
    parts = []
    for event in plan:
        if isinstance(event, TxMarker):
            parts.append(render_burst([event.freq], duration=SYMBOL_DURATION))
        elif isinstance(event, TxSymbol):
            parts.append(
                render_burst([event.freq_a, event.freq_b], duration=SYMBOL_DURATION)
            )
    silence = np.zeros(int(44100 * pad_seconds), dtype=np.float32)
    return np.concatenate([silence, *parts, silence]).astype(np.float32)


class FakeInputStream:
    """Streams the module-level input signal to a PortAudio-style callback."""

    def __init__(self, **kwargs):
        self.blocksize = kwargs.get("blocksize", SLIDE_LENGTH)
        self.channels = kwargs.get("channels", 1)
        self.callback = kwargs["callback"]

    def start(self):
        signal = _INPUT_SIGNAL
        # Feed the whole signal in blocksize chunks, then a short silence tail.
        total = len(signal) + self.blocksize * 4
        pos = 0
        while pos < total:
            end = min(pos + self.blocksize, len(signal))
            chunk = np.pad(signal[pos:end], (0, self.blocksize - (end - pos)))
            self.callback(chunk.reshape(-1, self.channels), self.blocksize, 0, None)
            pos += self.blocksize

    def close(self, ignore_errors=False):
        pass


class FakeOutputStream:
    """Consumes rendered audio via the callback and signals completion."""

    def __init__(self, **kwargs):
        self.callback = kwargs["callback"]
        self.finished = kwargs.get("finished_callback")
        self.blocksize = SLIDE_LENGTH

    def start(self):
        out = np.zeros((self.blocksize, 1), dtype=np.float32)
        try:
            while True:
                self.callback(out, self.blocksize, 0, None)
        except sd.CallbackStop:
            pass
        if self.finished is not None:
            self.finished()

    def close(self, ignore_errors=False):
        pass


def install() -> None:
    """Patch :mod:`sounddevice` with the fake streams."""
    sd.InputStream = FakeInputStream  # type: ignore[assignment]
    sd.OutputStream = FakeOutputStream  # type: ignore[assignment]
