"""
fault_classifier.py

Robust rule-based fault classifier for the Portable PCB Health Scanner.

Classification order:

    1. HEALTHY
    2. SHORT CIRCUIT
    3. OPEN TRACE
    4. REGULATOR FAULT
    5. CAPACITOR SIGNATURE DRIFT
    6. UNKNOWN / ABNORMAL SIGNATURE

Important:
    Capacitor drift is NOT declared from generic spectral deviation.
    It requires evidence that a characteristic frequency peak has moved.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from spectral_analysis import Signature, compute_signature
from signature_comparison import (
    ComparisonResult,
    compare_signatures,
)


# =====================================================================
# CONFIGURATION
# =====================================================================

@dataclass
class ClassificationConfig:

    # Healthy tolerance for the generic worst-band deviation score.
    #
    # Calibrated from an actual 150-board healthy population after
    # fixing the baseline-averaging and comparison-metric bugs:
    # mean=0.070, std=0.027, observed max=0.189. This threshold only
    # matters for the final HEALTHY-vs-UNKNOWN distinction now (see
    # classify_fault) -- real faults are caught by the dedicated
    # detectors below regardless of this value.
    abnormality_threshold: float = 0.22

    # Strong amplitude loss.
    missing_band_threshold: float = 0.35

    # Severe energy collapse.
    collapse_ratio_threshold: float = 0.35

    # Regulator detection.
    #
    # The window is deliberately narrow (100-700 Hz) and deliberately
    # excludes the board's always-present ~1000 Hz structural peak.
    # That peak's energy is roughly 100x bigger than the injected
    # ripple tone, so a wide band including it drowns the ripple
    # signal almost completely (ratio barely moves from 1.0 whether
    # the fault is present or not). Isolating the near-silent gap
    # where the ripple actually appears gives a ~800x jump in energy
    # ratio when the fault is present, vs. essentially none normally.
    low_frequency_start_hz: float = 100.0
    low_frequency_end_hz: float = 700.0
    low_frequency_energy_ratio: float = 20.0
    phase_shift_threshold: float = 0.20

    # Capacitor detection.
    capacitor_min_frequency_hz: float = 8000.0
    capacitor_min_peak_shift_fraction: float = 0.08
    capacitor_min_peak_prominence: float = 0.20
    # Search width must be wide enough to still find a peak after a
    # realistic drift. A 25% shift on the 15 kHz component moves it
    # ~3750-3900 Hz -- the old 2500 Hz width could never see that, so
    # capacitor drift was silently falling through to UNKNOWN. 4500 Hz
    # comfortably covers that while staying clear of neighboring
    # characteristic peaks (5000 Hz and 40000 Hz).
    capacitor_search_width_hz: float = 4500.0

    # Classification confidence.
    minimum_fault_confidence: float = 0.50

    # Additional safety.
    healthy_peak_shift_limit: float = 0.08
    # Search width used by the healthy-gate peak-shift check. Must be
    # wide enough that a genuine large shift (e.g. capacitor drift) is
    # actually found and correctly measured as a big shift -- a too-
    # narrow window just finds noise near the old location and reports
    # a false "no shift", letting the fault slip past the healthy gate.
    healthy_peak_search_width_hz: float = 4500.0


# =====================================================================
# RESULT
# =====================================================================

@dataclass
class FaultClassification:

    label: str
    confidence: float
    overall_deviation: float
    evidence: list[str]

    dominant_band_low_hz: float
    dominant_band_high_hz: float

    abnormal: bool


# =====================================================================
# BASIC UTILITIES
# =====================================================================

def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:

    return float(
        numerator /
        max(
            denominator,
            1e-12,
        )
    )


def _band_mask(
    signature: Signature,
    low_hz: float,
    high_hz: float,
):

    return (
        (signature.frequencies >= low_hz)
        &
        (signature.frequencies <= high_hz)
    )


def _band_energy(
    signature: Signature,
    low_hz: float,
    high_hz: float,
) -> float:

    mask = _band_mask(
        signature,
        low_hz,
        high_hz,
    )

    if not np.any(mask):
        return 0.0

    magnitude = np.asarray(
        signature.magnitude
    )

    return float(
        np.sum(
            magnitude[mask] ** 2
        )
    )


def _phase_difference(
    baseline: Signature,
    live: Signature,
    low_hz: float,
    high_hz: float,
) -> float:

    mask = _band_mask(
        baseline,
        low_hz,
        high_hz,
    )

    if not np.any(mask):
        return 0.0

    baseline_phase = np.asarray(
        baseline.phase
    )[mask]

    live_phase = np.asarray(
        live.phase
    )[mask]

    difference = np.angle(
        np.exp(
            1j *
            (
                live_phase -
                baseline_phase
            )
        )
    )

    return float(
        np.mean(
            np.abs(difference)
        )
    )


# =====================================================================
# PEAK DETECTION
# =====================================================================

def _find_peak(
    signature: Signature,
    center_hz: float,
    search_width_hz: float,
):
    """
    Find the strongest spectral peak around a known characteristic
    frequency.
    """

    frequencies = np.asarray(
        signature.frequencies
    )

    magnitude = np.asarray(
        signature.magnitude
    )

    low = center_hz - search_width_hz
    high = center_hz + search_width_hz

    mask = (
        (frequencies >= low)
        &
        (frequencies <= high)
    )

    if not np.any(mask):
        return None

    indices = np.where(mask)[0]

    local_index = indices[
        np.argmax(
            magnitude[indices]
        )
    ]

    return (
        float(frequencies[local_index]),
        float(magnitude[local_index]),
    )


# =====================================================================
# HEALTHY CHECK
# =====================================================================

def _healthy_peak_shift(
    baseline: Signature,
    live: Signature,
    config: ClassificationConfig,
) -> float:

    """
    Estimate maximum normalized shift of important board peaks.

    Normal board variation is small.
    """

    reference_frequencies = [
        1000.0,
        5000.0,
        15000.0,
        40000.0,
    ]

    shifts = []

    for reference in reference_frequencies:

        base_peak = _find_peak(
            baseline,
            reference,
            config.healthy_peak_search_width_hz,
        )

        live_peak = _find_peak(
            live,
            reference,
            config.healthy_peak_search_width_hz,
        )

        if (
            base_peak is None
            or
            live_peak is None
        ):
            continue

        base_frequency = base_peak[0]
        live_frequency = live_peak[0]

        shift = abs(
            live_frequency -
            base_frequency
        ) / max(
            base_frequency,
            1.0,
        )

        shifts.append(
            shift
        )

    if not shifts:
        return 0.0

    return float(
        max(shifts)
    )


# =====================================================================
# OPEN TRACE
# =====================================================================

def detect_open_trace(
    baseline: Signature,
    live: Signature,
    comparison: ComparisonResult,
    config: ClassificationConfig,
):

    best_score = 0.0
    best_band = None
    best_ratio = 1.0

    for band in comparison.band_scores:

        baseline_energy = _band_energy(
            baseline,
            band.low_hz,
            band.high_hz,
        )

        live_energy = _band_energy(
            live,
            band.low_hz,
            band.high_hz,
        )

        ratio = _safe_ratio(
            live_energy,
            baseline_energy,
        )

        if (
            band.normalized_distance
            >= config.missing_band_threshold
            and
            ratio < 0.60
        ):

            attenuation_score = min(
                1.0,
                max(
                    0.0,
                    1.0 - ratio,
                ),
            )

            deviation_score = min(
                1.0,
                band.normalized_distance,
            )

            score = (
                0.60 * attenuation_score
                +
                0.40 * deviation_score
            )

            if score > best_score:

                best_score = score
                best_band = band
                best_ratio = ratio

    if best_band is None:
        return 0.0, []

    evidence = [

        "Strong spectral attenuation detected.",

        (
            f"Energy ratio in "
            f"{best_band.low_hz:.0f}-"
            f"{best_band.high_hz:.0f} Hz band "
            f"is {best_ratio:.3f}."
        ),

        (
            "Pattern is consistent with an "
            "open trace."
        ),
    ]

    return best_score, evidence


# =====================================================================
# SHORT CIRCUIT
# =====================================================================

def detect_short_circuit(
    baseline: Signature,
    live: Signature,
    comparison: ComparisonResult,
    config: ClassificationConfig,
):

    collapsed = []

    for band in comparison.band_scores:

        baseline_energy = _band_energy(
            baseline,
            band.low_hz,
            band.high_hz,
        )

        live_energy = _band_energy(
            live,
            band.low_hz,
            band.high_hz,
        )

        ratio = _safe_ratio(
            live_energy,
            baseline_energy,
        )

        if (
            ratio
            <
            config.collapse_ratio_threshold
        ):

            collapsed.append(
                (
                    band,
                    ratio,
                )
            )

    # A short circuit in this model always collapses impedance across
    # more than one characteristic component -- that's what makes it
    # physically distinct from an open trace, which attenuates a
    # single band. Without this check, a single collapsed band scored
    # just as highly here as it did for detect_open_trace(), and ties
    # were resolved in favor of SHORT CIRCUIT purely by candidate-list
    # ordering -- silently misclassifying a large fraction of real
    # open-trace faults.
    if len(collapsed) < 2:
        return 0.0, []

    strongest_band, strongest_ratio = max(
        collapsed,
        key=lambda item:
        1.0 - item[1],
    )

    collapse_strength = min(
        1.0,
        1.0 - strongest_ratio,
    )

    # Multiple collapsed bands are a strong
    # indicator of short circuit in this model.
    collapse_strength = min(
        1.0,
        collapse_strength + 0.25,
    )

    evidence = [

        (
            f"Severe energy collapse detected in "
            f"{len(collapsed)} frequency band(s)."
        ),

        (
            f"Strongest collapse: "
            f"{strongest_band.low_hz:.0f}-"
            f"{strongest_band.high_hz:.0f} Hz."
        ),

        (
            f"Live/baseline energy ratio = "
            f"{strongest_ratio:.3f}."
        ),

        (
            "Multiple-band collapse is consistent "
            "with a short circuit."
        ),
    ]

    return collapse_strength, evidence


# =====================================================================
# REGULATOR FAULT
# =====================================================================

def detect_regulator_fault(
    baseline: Signature,
    live: Signature,
    config: ClassificationConfig,
):
    """
    Regulator faults are detected primarily by energy appearing in a
    normally near-silent low-frequency gap (100-700 Hz), where the
    fault's injected ripple tone lives but the board's normal
    structural response does not.

    Phase shift is kept only as supporting evidence. It is NOT
    required to fire: raw per-bin FFT phase is extremely sensitive to
    the sub-bin frequency jitter every healthy board already has, so
    on its own it cannot reliably distinguish healthy from faulty here
    and requiring it caused real regulator faults to be missed
    entirely in testing.
    """

    low = config.low_frequency_start_hz
    high = config.low_frequency_end_hz

    baseline_energy = _band_energy(
        baseline,
        low,
        high,
    )

    live_energy = _band_energy(
        live,
        low,
        high,
    )

    energy_ratio = _safe_ratio(
        live_energy,
        baseline_energy,
    )

    energy_detected = (
        energy_ratio
        >=
        config.low_frequency_energy_ratio
    )

    if not energy_detected:
        return 0.0, []

    # The healthy/fault gap here is enormous (roughly 5x vs. 4000x
    # baseline), so any detection at all is very strong evidence.
    # Scale generously above the threshold to reach full confidence
    # quickly rather than needing an extreme ratio.
    energy_score = min(
        1.0,
        (energy_ratio - config.low_frequency_energy_ratio)
        / (config.low_frequency_energy_ratio * 2.0)
        + 0.6,
    )

    score = 0.85 * energy_score

    evidence = [
        (
            f"Low-frequency energy in the "
            f"{low:.0f}-{high:.0f} Hz gap is "
            f"{energy_ratio:.1f}× the healthy baseline."
        ),
    ]

    # Optional supporting evidence only -- never required.
    phase_shift = _phase_difference(
        baseline,
        live,
        low,
        high,
    )

    if phase_shift >= config.phase_shift_threshold:

        phase_score = min(
            1.0,
            phase_shift /
            (config.phase_shift_threshold * 4.0),
        )

        score += 0.15 * phase_score

        evidence.append(
            (
                f"Low-frequency phase also shifted "
                f"({phase_shift:.2f} rad)."
            )
        )

    evidence.append(
        "Pattern is consistent with a regulator fault."
    )

    return min(1.0, score), evidence


# =====================================================================
# CAPACITOR DRIFT
# =====================================================================

def detect_capacitor_drift(
    baseline: Signature,
    live: Signature,
    config: ClassificationConfig,
):

    """
    Capacitor drift is detected by movement of a characteristic
    high-frequency peak.

    Generic spectral deviation alone is NOT sufficient.
    """

    reference_frequencies = [
        15000.0,
        40000.0,
    ]

    best_score = 0.0
    best_shift = 0.0
    best_reference = None
    best_base_peak = None
    best_live_peak = None

    for reference in reference_frequencies:

        if (
            reference
            <
            config.capacitor_min_frequency_hz
        ):
            continue

        baseline_peak = _find_peak(
            baseline,
            reference,
            config.capacitor_search_width_hz,
        )

        live_peak = _find_peak(
            live,
            reference,
            config.capacitor_search_width_hz,
        )

        if (
            baseline_peak is None
            or
            live_peak is None
        ):
            continue

        base_frequency = baseline_peak[0]
        live_frequency = live_peak[0]

        shift_fraction = abs(
            live_frequency -
            base_frequency
        ) / max(
            base_frequency,
            1.0,
        )

        if (
            shift_fraction
            <
            config.capacitor_min_peak_shift_fraction
        ):
            continue

        # Make sure the live peak is actually significant
        # compared with the local baseline peak.
        amplitude_ratio = _safe_ratio(
            live_peak[1],
            baseline_peak[1],
        )

        if amplitude_ratio < 0.30:
            continue

        shift_score = min(
            1.0,
            shift_fraction /
            0.25,
        )

        if shift_score > best_score:

            best_score = shift_score
            best_shift = shift_fraction
            best_reference = reference
            best_base_peak = baseline_peak
            best_live_peak = live_peak

    if best_reference is None:
        return 0.0, []

    evidence = [

        (
            f"Characteristic high-frequency peak near "
            f"{best_reference:.0f} Hz has shifted."
        ),

        (
            f"Baseline peak: "
            f"{best_base_peak[0]:.0f} Hz."
        ),

        (
            f"Live peak: "
            f"{best_live_peak[0]:.0f} Hz."
        ),

        (
            f"Peak displacement: "
            f"{best_shift * 100:.1f}%."
        ),

        (
            "Peak movement is consistent with "
            "capacitor-related signature drift."
        ),
    ]

    return best_score, evidence


# =====================================================================
# MAIN CLASSIFIER
# =====================================================================

def classify_fault(
    baseline: Signature,
    live: Signature,
    comparison: ComparisonResult | None = None,
    config: ClassificationConfig | None = None,
):

    if config is None:
        config = ClassificationConfig()

    if comparison is None:

        comparison = compare_signatures(
            baseline,
            live,
        )

    overall_deviation = float(
        comparison.overall_deviation
    )

    peak_shift = _healthy_peak_shift(
        baseline,
        live,
        config,
    )

    # ---------------------------------------------------------------
    # FAULT DETECTION
    #
    # All fault-specific detectors are run unconditionally, every
    # time, and the best-scoring one wins. A board is only reported
    # HEALTHY if none of them find sufficient evidence.
    #
    # An earlier version of this classifier used a "healthy gate"
    # that checked overall_deviation/peak_shift FIRST and only ran the
    # fault detectors if that gate failed. That was a real bug: some
    # faults (regulator fault, capacitor drift) don't necessarily push
    # the generic whole-spectrum deviation score above a generic
    # threshold even though their own dedicated detector can identify
    # them cleanly and confidently. The old gate could short-circuit
    # straight to "HEALTHY" before those specific detectors ever ran,
    # silently missing real faults. Running every detector every time
    # and trusting the best evidence removes that failure mode.
    # ---------------------------------------------------------------

    open_score, open_evidence = detect_open_trace(
        baseline,
        live,
        comparison,
        config,
    )

    short_score, short_evidence = detect_short_circuit(
        baseline,
        live,
        comparison,
        config,
    )

    regulator_score, regulator_evidence = detect_regulator_fault(
        baseline,
        live,
        config,
    )

    capacitor_score, capacitor_evidence = detect_capacitor_drift(
        baseline,
        live,
        config,
    )

    candidates = [

        (
            "SHORT CIRCUIT",
            short_score,
            short_evidence,
            0,
        ),

        (
            "OPEN TRACE",
            open_score,
            open_evidence,
            1,
        ),

        (
            "REGULATOR FAULT",
            regulator_score,
            regulator_evidence,
            2,
        ),

        (
            "CAPACITOR SIGNATURE DRIFT",
            capacitor_score,
            capacitor_evidence,
            3,
        ),
    ]

    # ---------------------------------------------------------------
    # Sort by score
    # ---------------------------------------------------------------

    candidates.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    (
        best_label,
        best_score,
        best_evidence,
        _,
    ) = candidates[0]

    # ---------------------------------------------------------------
    # HEALTHY
    #
    # None of the fault detectors found meaningful evidence
    # (best_score is effectively zero, not just "below the fault
    # confidence bar"), AND the generic whole-spectrum deviation and
    # characteristic-peak position are both within the range measured
    # across a large healthy-board population. This is checked AFTER
    # the fault detectors, not instead of running them, so a fault
    # that a detector is confident about is never masked by a
    # low generic deviation score.
    # ---------------------------------------------------------------

    no_fault_evidence = best_score < 0.05

    within_generic_bounds = (
        overall_deviation <= config.abnormality_threshold
        and peak_shift <= config.healthy_peak_shift_limit
    )

    if no_fault_evidence and within_generic_bounds:

        confidence = max(
            0.0,
            min(
                1.0,
                1.0
                - (
                    overall_deviation
                    / max(config.abnormality_threshold, 1e-12)
                )
                * 0.5,
            ),
        )

        return FaultClassification(

            label="HEALTHY",

            confidence=confidence,

            overall_deviation=overall_deviation,

            evidence=[

                (
                    "Spectral deviation is within "
                    "the healthy operating region."
                ),

                (
                    f"Maximum characteristic peak shift "
                    f"is {peak_shift * 100:.1f}%."
                ),

                (
                    "No fault-specific signature "
                    "was detected."
                ),
            ],

            dominant_band_low_hz=0.0,

            dominant_band_high_hz=0.0,

            abnormal=False,
        )

    # ---------------------------------------------------------------
    # UNKNOWN
    #
    # Either a detector found weak/ambiguous evidence (some deviation
    # from baseline, but not enough to confidently name a specific
    # fault), or the board is outside normal healthy bounds without
    # matching a known fault pattern.
    # ---------------------------------------------------------------

    if (
        best_score
        <
        config.minimum_fault_confidence
    ):

        return FaultClassification(

            label="UNKNOWN / ABNORMAL SIGNATURE",

            confidence=min(
                1.0,
                max(
                    0.0,
                    overall_deviation,
                ),
            ),

            overall_deviation=overall_deviation,

            evidence=[

                (
                    "The live signature differs from "
                    "the healthy baseline."
                ),

                (
                    "No known fault signature has "
                    "sufficient diagnostic evidence."
                ),
            ],

            dominant_band_low_hz=(
                comparison.worst_band_low_hz
            ),

            dominant_band_high_hz=(
                comparison.worst_band_high_hz
            ),

            abnormal=True,
        )

    # ---------------------------------------------------------------
    # KNOWN FAULT
    # ---------------------------------------------------------------

    return FaultClassification(

        label=best_label,

        confidence=min(
            1.0,
            best_score,
        ),

        overall_deviation=overall_deviation,

        evidence=best_evidence,

        dominant_band_low_hz=(
            comparison.worst_band_low_hz
        ),

        dominant_band_high_hz=(
            comparison.worst_band_high_hz
        ),

        abnormal=True,
    )


# =====================================================================
# PRETTY PRINTER
# =====================================================================

def print_classification(
    name: str,
    result: FaultClassification,
):

    print()
    print("=" * 70)
    print(f"DIAGNOSIS: {name}")
    print("=" * 70)

    print(
        f"Overall deviation : "
        f"{result.overall_deviation:.6f}"
    )

    print(
        f"Classification    : "
        f"{result.label}"
    )

    print(
        f"Confidence        : "
        f"{result.confidence * 100:.1f}%"
    )

    print()
    print("Evidence:")

    for item in result.evidence:
        print(f"  - {item}")

    if result.abnormal:

        print()

        print(
            f"Dominant deviation band: "
            f"{result.dominant_band_low_hz:.0f}-"
            f"{result.dominant_band_high_hz:.0f} Hz"
        )


# =====================================================================
# SYNTHETIC VALIDATION
# =====================================================================

def run_fault_classification_test():

    from baseline_capture import build_average_baseline

    from signal_simulator import (
        AcquisitionConfig,
        BoardProfile,
        generate_known_good,
        generate_open_trace_fault,
        generate_short_circuit_fault,
        generate_regulator_fault,
        generate_capacitor_drift_fault,
    )

    print("=" * 70)
    print("ROBUST FAULT CLASSIFICATION VALIDATION")
    print("=" * 70)

    config = AcquisitionConfig()
    board = BoardProfile()

    # ---------------------------------------------------------------
    # BASELINE
    # ---------------------------------------------------------------

    print()
    print("Building healthy baseline...")

    baseline_signatures = []

    for i in range(10):

        signal = generate_known_good(
            config=config,
            board=board,
            unit_variation=True,
            seed=100 + i,
        )

        signature = compute_signature(
            signal,
            config.sample_rate_hz,
            window_type="hanning",
        )

        baseline_signatures.append(
            signature
        )

    baseline = build_average_baseline(
        baseline_signatures,
        "baseline_library/classifier_baseline.npz",
    )

    # ---------------------------------------------------------------
    # TEST CASES
    # ---------------------------------------------------------------

    test_cases = {

        "HEALTHY BOARD":
            generate_known_good(
                config,
                board,
                unit_variation=True,
                seed=1000,
            ),

        "OPEN TRACE":
            generate_open_trace_fault(
                config,
                board,
                band_index=1,
                seed=1001,
            ),

        "SHORT CIRCUIT":
            generate_short_circuit_fault(
                config,
                board,
                band_indices=(1, 2),
                seed=1002,
            ),

        "REGULATOR FAULT":
            generate_regulator_fault(
                config,
                board,
                seed=1003,
            ),

        "CAPACITOR DRIFT":
            generate_capacitor_drift_fault(
                config,
                board,
                band_index=2,
                seed=1004,
            ),
    }

    expected = {

        "HEALTHY BOARD":
            "HEALTHY",

        "OPEN TRACE":
            "OPEN TRACE",

        "SHORT CIRCUIT":
            "SHORT CIRCUIT",

        "REGULATOR FAULT":
            "REGULATOR FAULT",

        "CAPACITOR DRIFT":
            "CAPACITOR SIGNATURE DRIFT",
    }

    correct = 0

    for test_name, signal in test_cases.items():

        live_signature = compute_signature(
            signal,
            config.sample_rate_hz,
            window_type="hanning",
        )

        comparison = compare_signatures(
            baseline,
            live_signature,
        )

        diagnosis = classify_fault(
            baseline,
            live_signature,
            comparison,
        )

        print_classification(
            test_name,
            diagnosis,
        )

        if diagnosis.label == expected[test_name]:
            correct += 1

    accuracy = (
        100.0 *
        correct /
        len(test_cases)
    )

    print()
    print("=" * 70)
    print("CLASSIFICATION SUMMARY")
    print("=" * 70)

    print(
        f"Correct classifications : "
        f"{correct}/{len(test_cases)}"
    )

    print(
        f"Sanity-test accuracy    : "
        f"{accuracy:.1f}%"
    )

    if correct == len(test_cases):
        print(
            "PASS: All synthetic profiles classified correctly."
        )
    else:
        print(
            "WARNING: Some synthetic profiles were misclassified."
        )

    print("=" * 70)


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    run_fault_classification_test()