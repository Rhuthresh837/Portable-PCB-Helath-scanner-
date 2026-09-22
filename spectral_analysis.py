"""
spectral_analysis.py

FFT / spectral analysis module for the Portable PCB Health Scanner.

Takes a time-domain signal (real or synthetic, from signal_simulator.py or
eventually from real hardware) and converts it into a frequency-domain
"signature": arrays of frequency bins, magnitude, and phase. This signature
is what the baseline capture, comparison/scoring, and classification modules
all operate on downstream.

Pipeline for this module:
    time-domain signal
        -> windowing (Hanning/Hamming) to reduce spectral leakage
        -> FFT (via scipy.fft, real-input optimized)
        -> magnitude + phase vs frequency, positive frequencies only
        -> package into a Signature object for easy downstream use
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.fft import rfft, rfftfreq


# ---------------------------------------------------------------------------
# Signature container
# ---------------------------------------------------------------------------

@dataclass
class Signature:
    """
    A frequency-domain signature extracted from one scan.

    frequencies: bin center frequencies (Hz), positive frequencies only
    magnitude:   linear magnitude spectrum (same length as frequencies)
    magnitude_db: magnitude in dB (20*log10), convenient for plotting/scoring
    phase:       phase spectrum in radians (same length as frequencies)
    sample_rate_hz / n_samples: retained for reference / re-derivation
    """
    frequencies: np.ndarray
    magnitude: np.ndarray
    magnitude_db: np.ndarray
    phase: np.ndarray
    sample_rate_hz: float
    n_samples: int

    def magnitude_at(self, freq_hz: float, tolerance_hz: float | None = None) -> float:
        """Look up magnitude at (or nearest to) a given frequency."""
        idx = self._nearest_index(freq_hz, tolerance_hz)
        return float(self.magnitude[idx]) if idx is not None else 0.0

    def phase_at(self, freq_hz: float, tolerance_hz: float | None = None) -> float:
        """Look up phase at (or nearest to) a given frequency."""
        idx = self._nearest_index(freq_hz, tolerance_hz)
        return float(self.phase[idx]) if idx is not None else 0.0

    def _nearest_index(self, freq_hz: float, tolerance_hz: float | None) -> int | None:
        idx = int(np.argmin(np.abs(self.frequencies - freq_hz)))
        if tolerance_hz is not None and abs(self.frequencies[idx] - freq_hz) > tolerance_hz:
            return None
        return idx

    def band_energy(self, low_hz: float, high_hz: float) -> float:
        """Sum of squared magnitude within a frequency band -- used heavily
        by the comparison/scoring and classification modules later."""
        mask = (self.frequencies >= low_hz) & (self.frequencies <= high_hz)
        return float(np.sum(self.magnitude[mask] ** 2))


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def apply_window(signal: np.ndarray, window_type: str = "hanning") -> np.ndarray:
    """
    Apply a window function to reduce spectral leakage before FFT.

    window_type: "hanning" or "hamming"
    """
    n = len(signal)
    if window_type == "hanning":
        window = np.hanning(n)
    elif window_type == "hamming":
        window = np.hamming(n)
    else:
        raise ValueError(f"Unknown window_type '{window_type}', expected 'hanning' or 'hamming'")
    return signal * window


# ---------------------------------------------------------------------------
# Core spectral analysis
# ---------------------------------------------------------------------------

def compute_signature(
    signal: np.ndarray,
    sample_rate_hz: float,
    window_type: str = "hanning",
) -> Signature:
    """
    Convert a time-domain signal into a frequency-domain Signature.

    Steps: window -> real FFT -> magnitude/phase -> package.
    Uses rfft since input signals are real-valued, which is both correct
    and roughly 2x cheaper than a full complex FFT.
    """
    n = len(signal)
    windowed = apply_window(signal, window_type)

    spectrum = rfft(windowed)
    frequencies = rfftfreq(n, d=1.0 / sample_rate_hz)

    # Normalize magnitude by number of samples so amplitude scale is
    # comparable across signals/windows regardless of capture length.
    magnitude = np.abs(spectrum) / n
    magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-12))  # avoid log(0)
    phase = np.angle(spectrum)

    return Signature(
        frequencies=frequencies,
        magnitude=magnitude,
        magnitude_db=magnitude_db,
        phase=phase,
        sample_rate_hz=sample_rate_hz,
        n_samples=n,
    )


# ---------------------------------------------------------------------------
# Self-test / demo: confirm the spectrum looks clean against a known-good signal
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from signal_simulator import (
        AcquisitionConfig,
        BoardProfile,
        generate_known_good,
        generate_open_trace_fault,
        generate_short_circuit_fault,
        generate_regulator_fault,
    )

    cfg = AcquisitionConfig()
    board = BoardProfile()

    print(f"Board's designed characteristic frequencies (approx, before jitter):")
    for freq, amp, phase in board.components:
        print(f"  {freq:>8.1f} Hz  amplitude={amp:.2f}")
    print()

    # --- known-good check ---
    good_signal = generate_known_good(cfg, board, unit_variation=False, seed=1)
    sig = compute_signature(good_signal, cfg.sample_rate_hz)

    print(f"FFT bins: {len(sig.frequencies)}  "
          f"(resolution = {sig.frequencies[1] - sig.frequencies[0]:.2f} Hz/bin)")
    print(f"Frequency range: 0 to {sig.frequencies[-1]:.0f} Hz "
          f"(Nyquist = {cfg.sample_rate_hz/2:.0f} Hz)\n")

    print("Detected peaks in known-good signature (top 6 by magnitude):")
    top_idx = np.argsort(sig.magnitude)[::-1][:6]
    top_idx = sorted(top_idx)
    for idx in top_idx:
        print(f"  {sig.frequencies[idx]:>9.1f} Hz   "
              f"mag={sig.magnitude[idx]:.4f}  ({sig.magnitude_db[idx]:6.1f} dB)   "
              f"phase={sig.phase[idx]:+.2f} rad")

    print("\n-> Expect peaks close to the board's 4 characteristic frequencies "
          "(1000, 5000, 15000, 40000 Hz) -- confirms the FFT pipeline is correct.\n")

    # --- sanity check against faults: band energy should visibly drop/shift ---
    print("Band-energy sanity checks (good vs. faulty), band = 4500-5500 Hz "
          "(around the 5000 Hz component):")

    open_signal = generate_open_trace_fault(cfg, board, band_index=1, seed=2)
    short_signal = generate_short_circuit_fault(cfg, board, band_indices=(1, 2), seed=3)
    reg_signal = generate_regulator_fault(cfg, board, seed=4)

    for label, s in [
        ("good", good_signal),
        ("open_trace (band 1 = ~5kHz)", open_signal),
        ("short_circuit (bands 1,2)", short_signal),
        ("regulator_fault", reg_signal),
    ]:
        sig_i = compute_signature(s, cfg.sample_rate_hz)
        energy = sig_i.band_energy(4500, 5500)
        print(f"  {label:32s} band_energy = {energy:.6f}")

    print("\n-> open_trace/short_circuit should show much lower energy in this band "
          "than 'good'; regulator_fault should look closer to 'good' here since it "
          "doesn't target this band directly.")