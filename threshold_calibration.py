"""
threshold_calibration.py

Threshold calibration module for the Portable PCB Health Scanner.

Purpose
-------
Determine how much signature deviation is naturally produced by
healthy PCB-to-PCB manufacturing variation.

Method
------
1. Build a known-good baseline from several healthy boards.
2. Generate a separate population of healthy boards.
3. Run every healthy board through the signature scorer.
4. Measure the distribution of healthy deviation scores.
5. Set the fault threshold above the natural healthy variation.
6. Validate the threshold against synthetic faulty boards.

Important:
    The healthy calibration population is kept separate from the
    baseline population. This prevents the same samples from being
    used both to construct and validate the baseline.

Threshold strategy
------------------
The default threshold is:

    threshold = mean_healthy_deviation
                + margin_multiplier * std_healthy_deviation

A percentile-based threshold is also calculated for reference.

The final threshold should be selected experimentally after observing
the false-positive rate on a larger healthy validation population.
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

from spectral_analysis import compute_signature

from baseline_capture import build_average_baseline

from signature_comparison import (
    ComparisonResult,
    compare_signatures,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MARGIN_MULTIPLIER = 3.0

# A high percentile is useful as an additional reference.
DEFAULT_HEALTHY_PERCENTILE = 99.0


# ---------------------------------------------------------------------------
# Calibration result
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """
    Stores the statistical result of healthy-board calibration.
    """

    scores: np.ndarray

    mean_deviation: float
    std_deviation: float

    minimum_deviation: float
    maximum_deviation: float

    percentile_95: float
    percentile_99: float

    margin_multiplier: float

    fault_threshold: float


# ---------------------------------------------------------------------------
# Generate healthy calibration population
# ---------------------------------------------------------------------------

def generate_healthy_signatures(
    n_samples: int,
    config: AcquisitionConfig,
    board: BoardProfile,
    seed: int = 1000,
):
    """
    Generate a population of known-good PCB signatures.

    Manufacturing variation is enabled.

    This population should be separate from the boards used to
    construct the average baseline.
    """

    signatures = []

    for i in range(n_samples):

        signal = generate_known_good(
            config=config,
            board=board,
            unit_variation=True,
            seed=seed + i,
        )

        signature = compute_signature(
            signal,
            config.sample_rate_hz,
            window_type="hanning",
        )

        signatures.append(signature)

    return signatures


# ---------------------------------------------------------------------------
# Calculate healthy deviation scores
# ---------------------------------------------------------------------------

def calculate_healthy_scores(
    baseline,
    healthy_signatures,
):
    """
    Compare every healthy validation signature against the baseline.

    Returns an array of normalized spectral deviations.
    """

    scores = []

    for signature in healthy_signatures:

        result = compare_signatures(
            baseline,
            signature,
        )

        scores.append(
            result.overall_deviation
        )

    return np.asarray(scores, dtype=float)


# ---------------------------------------------------------------------------
# Calculate threshold
# ---------------------------------------------------------------------------

def calibrate_threshold(
    healthy_scores: np.ndarray,
    margin_multiplier: float = DEFAULT_MARGIN_MULTIPLIER,
    percentile: float = DEFAULT_HEALTHY_PERCENTILE,
) -> CalibrationResult:
    """
    Calculate a fault threshold from healthy-board deviation scores.

    Main threshold:

        threshold = mean + k * standard_deviation

    where k is normally 3.

    A percentile threshold is also calculated so the two methods
    can be compared.
    """

    healthy_scores = np.asarray(
        healthy_scores,
        dtype=float,
    )

    if healthy_scores.size == 0:
        raise ValueError(
            "No healthy scores supplied for calibration."
        )

    mean_deviation = float(
        np.mean(healthy_scores)
    )

    std_deviation = float(
        np.std(healthy_scores, ddof=1)
    )

    minimum_deviation = float(
        np.min(healthy_scores)
    )

    maximum_deviation = float(
        np.max(healthy_scores)
    )

    percentile_95 = float(
        np.percentile(healthy_scores, 95)
    )

    percentile_99 = float(
        np.percentile(healthy_scores, 99)
    )

    fault_threshold = (
        mean_deviation
        + margin_multiplier * std_deviation
    )

    return CalibrationResult(
        scores=healthy_scores,

        mean_deviation=mean_deviation,
        std_deviation=std_deviation,

        minimum_deviation=minimum_deviation,
        maximum_deviation=maximum_deviation,

        percentile_95=percentile_95,
        percentile_99=percentile_99,

        margin_multiplier=margin_multiplier,

        fault_threshold=fault_threshold,
    )


# ---------------------------------------------------------------------------
# Save calibrated threshold
# ---------------------------------------------------------------------------

def save_calibration(
    result: CalibrationResult,
    filepath: str | Path,
) -> None:
    """
    Save calibration statistics to an NPZ file.

    This allows the scanner to load the calibrated threshold later
    without recalculating it every time.
    """

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        filepath,

        scores=result.scores,

        mean_deviation=result.mean_deviation,
        std_deviation=result.std_deviation,

        minimum_deviation=result.minimum_deviation,
        maximum_deviation=result.maximum_deviation,

        percentile_95=result.percentile_95,
        percentile_99=result.percentile_99,

        margin_multiplier=result.margin_multiplier,

        fault_threshold=result.fault_threshold,
    )


# ---------------------------------------------------------------------------
# Load calibration
# ---------------------------------------------------------------------------

def load_calibration(
    filepath: str | Path,
) -> CalibrationResult:
    """
    Load previously calculated calibration statistics.
    """

    data = np.load(filepath)

    return CalibrationResult(
        scores=data["scores"],

        mean_deviation=float(
            data["mean_deviation"]
        ),

        std_deviation=float(
            data["std_deviation"]
        ),

        minimum_deviation=float(
            data["minimum_deviation"]
        ),

        maximum_deviation=float(
            data["maximum_deviation"]
        ),

        percentile_95=float(
            data["percentile_95"]
        ),

        percentile_99=float(
            data["percentile_99"]
        ),

        margin_multiplier=float(
            data["margin_multiplier"]
        ),

        fault_threshold=float(
            data["fault_threshold"]
        ),
    )


# ---------------------------------------------------------------------------
# Classification using calibrated threshold
# ---------------------------------------------------------------------------

def classify_by_threshold(
    deviation: float,
    threshold: float,
) -> str:
    """
    Classify a scan using the calibrated deviation threshold.
    """

    if deviation <= threshold:
        return "HEALTHY / WITHIN NORMAL VARIATION"

    return "POTENTIAL FAULT"


# ---------------------------------------------------------------------------
# Validate threshold against faulty boards
# ---------------------------------------------------------------------------

def validate_against_faults(
    baseline,
    threshold: float,
    config: AcquisitionConfig,
    board: BoardProfile,
):
    """
    Generate synthetic faulty boards and determine whether the
    calibrated threshold detects their deviations.

    This is a validation step, NOT part of the threshold calculation.
    """

    fault_generators = {

        "open_trace": lambda seed:
            generate_open_trace_fault(
                config=config,
                board=board,
                band_index=1,
                seed=seed,
            ),

        "short_circuit": lambda seed:
            generate_short_circuit_fault(
                config=config,
                board=board,
                band_indices=(1, 2),
                seed=seed,
            ),

        "regulator_fault": lambda seed:
            generate_regulator_fault(
                config=config,
                board=board,
                seed=seed,
            ),

        "capacitor_drift": lambda seed:
            generate_capacitor_drift_fault(
                config=config,
                board=board,
                band_index=2,
                seed=seed,
            ),
    }

    results = {}

    for fault_name, generator in fault_generators.items():

        signal = generator(
            5000 + len(results)
        )

        signature = compute_signature(
            signal,
            config.sample_rate_hz,
            window_type="hanning",
        )

        comparison = compare_signatures(
            baseline,
            signature,
        )

        deviation = comparison.overall_deviation

        results[fault_name] = {
            "deviation": deviation,
            "detected": deviation > threshold,
            "comparison": comparison,
        }

    return results


# ---------------------------------------------------------------------------
# Print calibration report
# ---------------------------------------------------------------------------

def print_calibration_report(
    result: CalibrationResult,
) -> None:
    """
    Display the statistical calibration result.
    """

    print()
    print("=" * 70)
    print("THRESHOLD CALIBRATION REPORT")
    print("=" * 70)

    print(
        f"Healthy samples          : "
        f"{len(result.scores)}"
    )

    print(
        f"Minimum deviation        : "
        f"{result.minimum_deviation:.6f}"
    )

    print(
        f"Maximum deviation        : "
        f"{result.maximum_deviation:.6f}"
    )

    print(
        f"Mean deviation           : "
        f"{result.mean_deviation:.6f}"
    )

    print(
        f"Standard deviation       : "
        f"{result.std_deviation:.6f}"
    )

    print()
    print(
        f"95th percentile          : "
        f"{result.percentile_95:.6f}"
    )

    print(
        f"99th percentile          : "
        f"{result.percentile_99:.6f}"
    )

    print()
    print(
        f"Margin multiplier        : "
        f"{result.margin_multiplier:.2f} sigma"
    )

    print(
        f"CALIBRATED FAULT THRESHOLD: "
        f"{result.fault_threshold:.6f}"
    )

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main calibration experiment
# ---------------------------------------------------------------------------

def run_calibration_experiment() -> None:
    """
    Complete threshold-calibration experiment.

    Dataset split:

        Healthy baseline population
                ↓
        average baseline

        Separate healthy calibration population
                ↓
        natural deviation distribution
                ↓
        calibrated threshold

        Separate faulty samples
                ↓
        threshold validation
    """

    print("=" * 70)
    print("PORTABLE PCB HEALTH SCANNER")
    print("THRESHOLD CALIBRATION EXPERIMENT")
    print("=" * 70)

    cfg = AcquisitionConfig()
    board = BoardProfile()

    # ==============================================================
    # STEP 1
    # Build baseline using healthy boards
    # ==============================================================

    print()
    print("STEP 1: Building healthy reference baseline...")

    baseline_signatures = generate_healthy_signatures(
        n_samples=10,
        config=cfg,
        board=board,
        seed=100,
    )

    baseline = build_average_baseline(
        baseline_signatures,
        "baseline_library/calibration_baseline.npz",
    )

    print(
        f"Baseline constructed from "
        f"{len(baseline_signatures)} healthy boards."
    )

    # ==============================================================
    # STEP 2
    # Generate SEPARATE healthy calibration population
    # ==============================================================

    print()
    print(
        "STEP 2: Generating independent healthy "
        "calibration population..."
    )

    healthy_calibration_signatures = (
        generate_healthy_signatures(
            n_samples=100,
            config=cfg,
            board=board,
            seed=10000,
        )
    )

    print(
        f"Generated "
        f"{len(healthy_calibration_signatures)} "
        f"independent healthy boards."
    )

    # ==============================================================
    # STEP 3
    # Score all healthy boards
    # ==============================================================

    print()
    print(
        "STEP 3: Measuring natural healthy-board "
        "signature variation..."
    )

    healthy_scores = calculate_healthy_scores(
        baseline,
        healthy_calibration_signatures,
    )

    print(
        f"Calculated {len(healthy_scores)} "
        f"healthy deviation scores."
    )

    # ==============================================================
    # STEP 4
    # Calculate threshold
    # ==============================================================

    print()
    print(
        "STEP 4: Calculating statistical fault threshold..."
    )

    calibration = calibrate_threshold(
        healthy_scores,
        margin_multiplier=3.0,
        percentile=99.0,
    )

    print_calibration_report(
        calibration
    )

    # ==============================================================
    # STEP 5
    # Save threshold
    # ==============================================================

    calibration_file = (
        "baseline_library/threshold_calibration.npz"
    )

    save_calibration(
        calibration,
        calibration_file,
    )

    print()
    print(
        f"Calibration saved to: "
        f"{calibration_file}"
    )

    # ==============================================================
    # STEP 6
    # Validate using synthetic faults
    # ==============================================================

    print()
    print(
        "STEP 5: Validating threshold against "
        "synthetic faulty boards..."
    )

    fault_results = validate_against_faults(
        baseline,
        calibration.fault_threshold,
        cfg,
        board,
    )

    print()
    print("-" * 70)
    print("FAULT VALIDATION")
    print("-" * 70)

    for fault_name, result in fault_results.items():

        status = (
            "DETECTED"
            if result["detected"]
            else "NOT DETECTED"
        )

        print(
            f"{fault_name:20s} | "
            f"deviation = "
            f"{result['deviation']:.6f} | "
            f"{status}"
        )

    # ==============================================================
    # STEP 7
    # Final interpretation
    # ==============================================================

    print()
    print("=" * 70)
    print("FINAL CALIBRATION INTERPRETATION")
    print("=" * 70)

    print(
        f"Healthy mean deviation : "
        f"{calibration.mean_deviation:.6f}"
    )

    print(
        f"Healthy sigma           : "
        f"{calibration.std_deviation:.6f}"
    )

    print(
        f"Fault threshold        : "
        f"{calibration.fault_threshold:.6f}"
    )

    print()
    print(
        "Interpretation:"
    )

    print(
        "  deviation <= threshold"
    )
    print(
        "      -> within calibrated healthy variation"
    )

    print(
        "  deviation > threshold"
    )
    print(
        "      -> potential fault / abnormal board"
    )

    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_calibration_experiment()