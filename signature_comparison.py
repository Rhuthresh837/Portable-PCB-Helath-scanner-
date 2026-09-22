"""
signature_comparison.py

Core signature comparison and scoring module for the
Portable PCB Health Scanner.

Purpose:
    Compare a live PCB frequency-domain signature against a
    known-good baseline using Euclidean spectral distance.

The spectrum is divided into frequency bands so that the system
can identify WHERE the signature differs, not only how much it differs.

Pipeline:

    Live signal
        -> FFT / Signature
        -> frequency-band comparison
        -> Euclidean spectral distance
        -> normalized deviation score
        -> overall PCB deviation

Lower deviation = more similar to healthy baseline.
Higher deviation = more different from healthy baseline.

This module intentionally focuses on magnitude-spectrum comparison.
Phase-based scoring and ML classification will be added later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from signal_simulator import (
    AcquisitionConfig,
    BoardProfile,
    generate_known_good,
    generate_open_trace_fault,
    generate_short_circuit_fault,
    generate_regulator_fault,
    generate_capacitor_drift_fault,
)

from spectral_analysis import (
    Signature,
    compute_signature,
)

from baseline_capture import (
    capture_baseline,
    build_average_baseline,
    load_signature,
)


# ---------------------------------------------------------------------------
# Frequency bands
# ---------------------------------------------------------------------------

DEFAULT_BANDS = [
    (0, 2_000),
    (2_000, 7_500),
    (7_500, 20_000),
    (20_000, 50_000),
    (50_000, 100_000),
]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class BandScore:
    """Comparison result for one frequency band."""

    low_hz: float
    high_hz: float

    distance: float
    normalized_distance: float

    baseline_energy: float
    live_energy: float

    n_bins: int


@dataclass
class ComparisonResult:
    """Complete comparison between live signature and baseline."""

    overall_distance: float
    overall_deviation: float

    band_scores: list[BandScore]

    worst_band_low_hz: float
    worst_band_high_hz: float

    similarity_percent: float


# ---------------------------------------------------------------------------
# Basic validation
# ---------------------------------------------------------------------------

def _validate_signatures(
    baseline: Signature,
    live: Signature,
) -> None:
    """
    Ensure baseline and live signatures can be compared.
    """

    if len(baseline.frequencies) != len(live.frequencies):
        raise ValueError(
            "Baseline and live signatures have different FFT sizes."
        )

    if not np.allclose(
        baseline.frequencies,
        live.frequencies,
    ):
        raise ValueError(
            "Baseline and live signatures have different frequency bins."
        )


# ---------------------------------------------------------------------------
# Euclidean spectral distance
# ---------------------------------------------------------------------------

def spectral_euclidean_distance(
    baseline_magnitude: np.ndarray,
    live_magnitude: np.ndarray,
) -> float:
    """
    Calculate Euclidean distance between two magnitude spectra.

    Formula:

        D = sqrt(sum((live - baseline)^2))

    Smaller D means the spectra are more similar.
    """

    baseline_magnitude = np.asarray(
        baseline_magnitude,
        dtype=float,
    )

    live_magnitude = np.asarray(
        live_magnitude,
        dtype=float,
    )

    if baseline_magnitude.shape != live_magnitude.shape:
        raise ValueError(
            "Baseline and live magnitude arrays must have "
            "the same shape."
        )

    difference = live_magnitude - baseline_magnitude

    return float(np.sqrt(np.sum(difference ** 2)))


# ---------------------------------------------------------------------------
# Normalized spectral distance
# ---------------------------------------------------------------------------

def normalized_spectral_distance(
    baseline_magnitude: np.ndarray,
    live_magnitude: np.ndarray,
) -> float:
    """
    Calculate Euclidean spectral distance normalized by
    baseline spectral energy.

    This makes the score less dependent on the absolute
    amplitude scale of the signal.

        normalized_distance =
            ||live - baseline|| / ||baseline||

    0.0 means identical spectra.
    Larger values mean larger deviation.
    """

    baseline_magnitude = np.asarray(
        baseline_magnitude,
        dtype=float,
    )

    live_magnitude = np.asarray(
        live_magnitude,
        dtype=float,
    )

    denominator = np.linalg.norm(baseline_magnitude)

    if denominator < 1e-12:
        denominator = 1e-12

    distance = np.linalg.norm(
        live_magnitude - baseline_magnitude
    )

    return float(distance / denominator)


# ---------------------------------------------------------------------------
# Frequency-band comparison
# ---------------------------------------------------------------------------

def compare_frequency_band(
    baseline: Signature,
    live: Signature,
    low_hz: float,
    high_hz: float,
) -> BandScore:
    """
    Compare baseline and live signatures within one frequency band.

    IMPORTANT: this uses a band-ENERGY comparison, not a bin-by-bin
    magnitude comparison.

    Bin-by-bin comparison was tried first and rejected: individual
    boards have small (normal, healthy) manufacturing jitter that
    shifts each characteristic peak by a handful of FFT bins. That
    tiny, harmless frequency shift makes a bin-by-bin distance look
    enormous, because a peak that moved by a few bins contributes
    almost as much "distance" as a peak that vanished entirely. On
    this project's own data, that bug alone made every healthy board
    score in the same deviation range as genuine faults.

    Comparing total energy within a band instead is robust to that
    kind of small in-band frequency movement -- the energy is still
    "in there somewhere" for a healthy board, just at a slightly
    different bin -- while still reacting strongly to genuine
    amplitude-based faults (attenuation, collapse) where the energy
    actually leaves the band.
    """

    _validate_signatures(baseline, live)

    mask = (
        (baseline.frequencies >= low_hz)
        & (baseline.frequencies <= high_hz)
    )

    n_bins = int(np.sum(mask))

    if n_bins == 0:
        raise ValueError(
            f"No FFT bins exist between "
            f"{low_hz} Hz and {high_hz} Hz."
        )

    baseline_band = baseline.magnitude[mask]
    live_band = live.magnitude[mask]

    # Kept for reference / diagnostics -- no longer used to drive
    # classification decisions (see docstring above).
    distance = spectral_euclidean_distance(
        baseline_band,
        live_band,
    )

    baseline_energy = float(
        np.sum(baseline_band ** 2)
    )

    live_energy = float(
        np.sum(live_band ** 2)
    )

    # Normalized band deviation = how much the band's total energy
    # changed, relative to its healthy baseline energy.
    #   0.0   -> identical energy
    #   0.35  -> live energy differs from baseline by 35%
    normalized_distance = float(
        abs(live_energy - baseline_energy)
        / max(baseline_energy, 1e-12)
    )

    return BandScore(
        low_hz=low_hz,
        high_hz=high_hz,
        distance=distance,
        normalized_distance=normalized_distance,
        baseline_energy=baseline_energy,
        live_energy=live_energy,
        n_bins=n_bins,
    )


# ---------------------------------------------------------------------------
# Complete signature comparison
# ---------------------------------------------------------------------------

def compare_signatures(
    baseline: Signature,
    live: Signature,
    bands: list[tuple[float, float]] | None = None,
) -> ComparisonResult:
    """
    Compare a live PCB signature against a known-good baseline.

    The spectrum is divided into frequency bands.

    The overall deviation is calculated using normalized
    Euclidean distance across the complete spectrum.

    Returns:
        ComparisonResult
    """

    _validate_signatures(baseline, live)

    if bands is None:
        bands = DEFAULT_BANDS

    # Kept for reference / diagnostics -- see compare_frequency_band()
    # docstring for why full-spectrum bin-by-bin distance is not used
    # to drive the actual deviation score.
    overall_distance = spectral_euclidean_distance(
        baseline.magnitude,
        live.magnitude,
    )

    # -------------------------------------------------------
    # Per-band scores
    # -------------------------------------------------------

    band_scores = []

    for low_hz, high_hz in bands:

        # Skip bands outside the actual FFT range.
        if low_hz > baseline.frequencies[-1]:
            continue

        high_hz_actual = min(
            high_hz,
            baseline.frequencies[-1],
        )

        band_score = compare_frequency_band(
            baseline,
            live,
            low_hz,
            high_hz_actual,
        )

        band_scores.append(band_score)

    if not band_scores:
        raise ValueError(
            "None of the requested frequency bands overlap "
            "the signature frequency range."
        )

    # Find the frequency band with the largest deviation.
    worst_band = max(
        band_scores,
        key=lambda x: x.normalized_distance,
    )

    # Overall deviation = worst single-band energy deviation.
    #
    # A fault is, by construction, localized to one or two frequency
    # bands (an open trace/short/regulator ripple/etc. doesn't change
    # the *whole* spectrum uniformly). Taking the worst band therefore
    # tracks real fault severity, while staying insensitive to the
    # small, harmless energy fluctuation every band shows from normal
    # board-to-board manufacturing variation.
    overall_deviation = float(worst_band.normalized_distance)

    # Convert deviation into an initial similarity percentage.
    #
    # This is NOT the final health-score calibration.
    # It is only a convenient representation for this module.
    similarity_percent = 100.0 / (
        1.0 + overall_deviation
    )

    return ComparisonResult(
        overall_distance=overall_distance,
        overall_deviation=overall_deviation,
        band_scores=band_scores,
        worst_band_low_hz=worst_band.low_hz,
        worst_band_high_hz=worst_band.high_hz,
        similarity_percent=similarity_percent,
    )


# ---------------------------------------------------------------------------
# Pretty-print comparison
# ---------------------------------------------------------------------------

def print_comparison(
    label: str,
    result: ComparisonResult,
) -> None:
    """
    Print a human-readable comparison result.
    """

    print()
    print("-" * 70)
    print(f"TEST: {label}")
    print("-" * 70)

    print(
        f"Overall Euclidean distance : "
        f"{result.overall_distance:.6f}"
    )

    print(
        f"Normalized deviation       : "
        f"{result.overall_deviation:.6f}"
    )

    print(
        f"Similarity                 : "
        f"{result.similarity_percent:.2f}%"
    )

    print()
    print("Frequency-band analysis:")

    for band in result.band_scores:

        print(
            f"  {band.low_hz:8.0f} - "
            f"{band.high_hz:8.0f} Hz | "
            f"distance = {band.distance:.6f} | "
            f"normalized = {band.normalized_distance:.6f}"
        )

    print()
    print(
        f"Worst deviation band: "
        f"{result.worst_band_low_hz:.0f} - "
        f"{result.worst_band_high_hz:.0f} Hz"
    )


# ---------------------------------------------------------------------------
# Sanity test
# ---------------------------------------------------------------------------

def run_sanity_test() -> None:
    """
    Test the scorer against synthetic healthy and faulty boards.

    Expected behavior:

        Healthy board  -> low deviation
        Open trace     -> high deviation
        Short circuit  -> high deviation
        Regulator fault -> high deviation
        Capacitor drift -> high deviation
    """

    print("=" * 70)
    print("SIGNATURE COMPARISON / SCORING SANITY TEST")
    print("=" * 70)

    cfg = AcquisitionConfig()
    board = BoardProfile()

    # -------------------------------------------------------
    # Step 1: Generate several healthy boards
    # -------------------------------------------------------

    print()
    print("Generating healthy baseline boards...")

    baseline_signatures = []

    for i in range(10):

        signal = generate_known_good(
            config=cfg,
            board=board,
            unit_variation=True,
            seed=100 + i,
        )

        signature = compute_signature(
            signal,
            cfg.sample_rate_hz,
            window_type="hanning",
        )

        baseline_signatures.append(signature)

    print(
        f"Generated {len(baseline_signatures)} "
        f"healthy baseline signatures."
    )

    # -------------------------------------------------------
    # Step 2: Build average baseline
    # -------------------------------------------------------

    baseline = build_average_baseline(
        baseline_signatures,
        "baseline_library/test_average_baseline.npz",
    )

    print("Average baseline created.")

    # -------------------------------------------------------
    # Step 3: Generate test signals
    # -------------------------------------------------------

    test_signals = {

        "GOOD BOARD": generate_known_good(
            cfg,
            board,
            unit_variation=True,
            seed=500,
        ),

        "OPEN TRACE": generate_open_trace_fault(
            cfg,
            board,
            band_index=1,
            seed=501,
        ),

        "SHORT CIRCUIT": generate_short_circuit_fault(
            cfg,
            board,
            band_indices=(1, 2),
            seed=502,
        ),

        "REGULATOR FAULT": generate_regulator_fault(
            cfg,
            board,
            seed=503,
        ),

        "CAPACITOR DRIFT": generate_capacitor_drift_fault(
            cfg,
            board,
            band_index=2,
            seed=504,
        ),
    }

    # -------------------------------------------------------
    # Step 4: Compare every test signal
    # -------------------------------------------------------

    results = {}

    for label, signal in test_signals.items():

        live_signature = compute_signature(
            signal,
            cfg.sample_rate_hz,
            window_type="hanning",
        )

        result = compare_signatures(
            baseline,
            live_signature,
        )

        results[label] = result

        print_comparison(
            label,
            result,
        )

    # -------------------------------------------------------
    # Step 5: Final sanity check
    # -------------------------------------------------------

    good_deviation = results["GOOD BOARD"].overall_deviation

    faulty_deviations = [
        result.overall_deviation
        for label, result in results.items()
        if label != "GOOD BOARD"
    ]

    average_fault_deviation = float(
        np.mean(faulty_deviations)
    )

    print()
    print("=" * 70)
    print("SANITY-CHECK SUMMARY")
    print("=" * 70)

    print(
        f"Healthy-board deviation : "
        f"{good_deviation:.6f}"
    )

    print(
        f"Average faulty deviation: "
        f"{average_fault_deviation:.6f}"
    )

    print()

    if average_fault_deviation > good_deviation:
        print(
            "PASS: Faulty signatures show greater deviation "
            "than the healthy signature."
        )
    else:
        print(
            "WARNING: Faulty signatures did not separate "
            "clearly from the healthy signature."
        )

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_sanity_test()