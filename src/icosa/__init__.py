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
    icosa --rx            # Ctrl+C to stop and print decoded text
"""

import argparse
import queue
import sys

import numpy as np
import sounddevice as sd

# --- SHARED CONFIGURATION ---
SAMPLING_RATE = 44100
BASE_FREQ = 18000
SYMBOL_DURATION = 0.15  # seconds per tone burst (TX)

# --- RX CONFIGURATION ---
CHANNELS = 1
THRESHOLD = 10
RADIX2_EXP = 11
BLOCK_LENGTH = 2**RADIX2_EXP
SLIDE_LENGTH = BLOCK_LENGTH // 2


# =============================================================================
# Transmit
# =============================================================================


def play_frequencies_ifft(frequencies, duration=2.0, sample_rate=SAMPLING_RATE):
    """
    Generates a multi-frequency signal using Inverse FFT and plays it.

    :param frequencies: List or array of target frequencies in Hz
    :param duration: Duration of the audio in seconds
    :param sample_rate: Audio sampling rate (default 44100 Hz)
    """
    # 1. Calculate total number of samples
    n_samples = int(sample_rate * duration)

    # 2. Initialize an empty frequency domain spectrum (complex numbers)
    spectrum = np.zeros(n_samples, dtype=complex)

    # 3. Map frequencies to their closest spectral bin index
    # Formula: bin_index = frequency * total_samples / sample_rate
    for freq in frequencies:
        bin_idx = int(round(freq * n_samples / sample_rate))

        # Keep frequencies below the Nyquist limit
        if 0 <= bin_idx < n_samples // 2:
            # Scale by n_samples / 2 because IFFT implicitly divides by n_samples.
            spectrum[bin_idx] = n_samples / 2

            # For a strictly real time-domain signal, negative frequencies must
            # be the complex conjugate of the positive ones.
            spectrum[-bin_idx] = np.conj(spectrum[bin_idx])

    # 4. Transform frequency domain back to the time domain using IFFT
    time_signal = np.fft.ifft(spectrum)

    # Extract the real part (the imaginary part should be practically zero)
    audio_signal = time_signal.real

    # Gentle enveloping (50 ms cosine fades)
    fade_samples = int(sample_rate * 0.05)
    fade_samples = min(fade_samples, n_samples // 2)  # don't consume whole signal

    envelope = np.ones(n_samples)
    ramp = np.arange(fade_samples) / fade_samples
    envelope[:fade_samples] = 0.5 * (1 - np.cos(np.pi * ramp))  # 0 -> 1
    envelope[-fade_samples:] = 0.5 * (1 + np.cos(np.pi * ramp))  # 1 -> 0

    audio_signal *= envelope

    # 5. Normalize the signal to prevent clipping (-1.0 to 1.0)
    max_val = np.max(np.abs(audio_signal))
    if max_val > 0:
        audio_signal = audio_signal / max_val

    # 6. Play the audio and wait until it finishes
    sd.play(audio_signal, samplerate=sample_rate)
    sd.wait()


def bytearray_to_bit_pairs(data: bytearray, msb_first: bool = True) -> list[int]:
    """Converts a bytearray into a list of 2-bit integers (values 0-3)."""
    if msb_first:
        # Pairs from left to right: bits [7:6], [5:4], [3:2], [1:0]
        return [(byte >> shift) & 0x03 for byte in data for shift in (6, 4, 2, 0)]
    else:
        # Pairs from right to left: bits [1:0], [3:2], [5:4], [7:6]
        return [(byte >> shift) & 0x03 for byte in data for shift in (0, 2, 4, 6)]


def transmit(data: bytearray) -> None:
    bps = bytearray_to_bit_pairs(data + bytearray([17]))
    a = True
    for ind, bp in enumerate(bps):
        base = BASE_FREQ
        d = base + (bp + 1) * 100
        if not a:
            base += 500
            d += 500
        play_frequencies_ifft([base, d], duration=SYMBOL_DURATION)
        if ind % 4 == 3:
            # End of a byte (4 symbols): emit the boundary marker. Use BASE_FREQ
            # explicitly since `base` is shifted by 500 on odd (slot B) symbols.
            play_frequencies_ifft([BASE_FREQ + 1000], duration=SYMBOL_DURATION)
        a = not a


# =============================================================================
# Receive
# =============================================================================


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


def receive() -> None:
    # Imported lazily so that `--tx` works without audioflux installed/loaded.
    import audioflux as af
    from audioflux.type import SpectralFilterBankScaleType

    audio_queue: queue.Queue = queue.Queue()

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
            print(status, file=sys.stderr)
        audio_queue.put(indata.copy())

    print(f"Targeting: {float(BASE_FREQ)} Hz")
    print("Listening... Press Ctrl+C to stop.\n")

    buffer = np.zeros(BLOCK_LENGTH, dtype=np.float32)
    last_b = True
    base = BASE_FREQ
    bit_pairs: list[int] = []

    try:
        with sd.InputStream(
            samplerate=SAMPLING_RATE,
            channels=CHANNELS,
            callback=audio_callback,
            blocksize=SLIDE_LENGTH,
        ):
            while True:
                data_chunk = audio_queue.get().flatten()

                buffer = np.roll(buffer, -len(data_chunk))
                buffer[-len(data_chunk) :] = data_chunk

                spec_arr = bft.bft(buffer)
                magnitudes = np.abs(spec_arr)

                if magnitudes[fidx(base + 1000)] > THRESHOLD:
                    print("BYTE BOUNDARY")
                    # Pad any missed symbols so we land on a byte boundary. This is
                    # idempotent, so repeat detections of the same marker are harmless.
                    bit_pairs.extend([0] * (-len(bit_pairs) % 4))
                    # A byte always ends on a slot B symbol, so the next is slot A.
                    last_b = True
                    continue

                if magnitudes[fidx(base)] > THRESHOLD and last_b:
                    ms = [magnitudes[fidx(base + k * 100)] for k in (1, 2, 3, 4)]
                    sym = ms.index(max(ms))
                    bit_pairs.append(sym)
                    last_b = False
                    print("GOT A", sym)

                if magnitudes[fidx(base + 500)] > THRESHOLD and not last_b:
                    ms = [magnitudes[fidx(base + k * 100)] for k in (6, 7, 8, 9)]
                    sym = ms.index(max(ms))
                    bit_pairs.append(sym)
                    last_b = True
                    print("GOT B", sym)

                if len(bit_pairs) % 4 == 0 and bit_pairs[-4:] == [0, 1, 0, 1]:
                    print("GOT END MARKER")
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\nStream stopped.")
        byte_data = bit_pairs_to_bytes(bit_pairs)
        # latin-1 safely maps raw bytes
        print(byte_data.decode("latin-1"))
    except Exception as e:
        print(f"\nAn error occurred: {e}")


# =============================================================================
# CLI
# =============================================================================


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
        transmit(bytearray(args.tx, encoding="utf-8"))
    else:
        receive()
