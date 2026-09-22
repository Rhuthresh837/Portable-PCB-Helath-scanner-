"""
baseline_capture.py

Baseline capture module for the Portable PCB Health Scanner.

Purpose:
    Simulate the "teach the device a known-good PCB" workflow.

Workflow:
    1. Generate several known-good synthetic PCB signals.
    2. Convert each signal into a frequency-domain Signature.
    3. Store the signatures as a baseline library.
    4. Calculate an average baseline signature.
    5. Store the normal variation of the healthy boards.

The baseline library will later be used by the comparison,
scoring, threshold calibration, and fault classification modules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from signal_simulator import AcquisitionConfig, BoardProfile, generate_known_good
from spectral_analysis import Signature, compute_signature


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASELINE_DIR = Path("baseline_library")


# ---------------------------------------------------------------------------
# Save / load individual signatures
# ---------------------------------------------------------------------------

def save_signature(
    signature: Signature,
    filepath: str | Path,
) -> None:
    """
    Save one frequency-domain Signature to disk.

    The complete signature is stored:
        - frequencies
        - magnitude
        - magnitude_db
        - phase
        - sample rate
        - number of samples
    """

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        filepath,
        frequencies=signature.frequencies,
        magnitude=signature.magnitude,
        magnitude_db=signature.magnitude_db,
        phase=signature.phase,
        sample_rate_hz=signature.sample_rate_hz,
        n_samples=signature.n_samples,
    )


def load_signature(filepath: str | Path) -> Signature:
    """
    Load a previously saved Signature from an .npz file.
    """

    filepath = Path(filepath)

    data = np.load(filepath)

    return Signature(
        frequencies=data["frequencies"],
        magnitude=data["magnitude"],
        magnitude_db=data["magnitude_db"],
        phase=data["phase"],
        sample_rate_hz=float(data["sample_rate_hz"]),
        n_samples=int(data["n_samples"]),
    )


# ---------------------------------------------------------------------------
# Baseline capture
# ---------------------------------------------------------------------------

def capture_baseline(
    n_boards: int = 10,
    config: AcquisitionConfig | None = None,
    board: BoardProfile | None = None,
    baseline_dir: str | Path = BASELINE_DIR,
    window_type: str = "hanning",
    seed: int = 100,
) -> list[Signature]:
    """
    Generate and store multiple known-good PCB signatures.

    Each generated board contains small random manufacturing variations.

    Parameters
    ----------
    n_boards:
        Number of known-good boards/scans to generate.

    config:
        ADC/sampling configuration.

    board:
        PCB frequency-response model.

    baseline_dir:
        Directory where signatures will be stored.

    window_type:
        FFT window used by compute_signature().

    seed:
        Starting random seed.

    Returns
    -------
    list[Signature]
        List containing all captured baseline signatures.
    """

    if config is None:
        config = AcquisitionConfig()

    if board is None:
        board = BoardProfile()

    baseline_dir = Path(baseline_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    signatures = []

    print("=" * 60)
    print("PORTABLE PCB HEALTH SCANNER")
    print("BASELINE CAPTURE")
    print("=" * 60)

    print(f"Number of baseline scans : {n_boards}")
    print(f"Sample rate              : {config.sample_rate_hz:.0f} Hz")
    print(f"Capture duration         : {config.duration_s:.3f} s")
    print(f"FFT window               : {window_type}")
    print()

    for i in range(n_boards):

        # Generate a known-good board with manufacturing variation.
        signal = generate_known_good(
            config=config,
            board=board,
            unit_variation=True,
            seed=seed + i,
        )

        # Convert time-domain signal into frequency-domain signature.
        signature = compute_signature(
            signal,
            config.sample_rate_hz,
            window_type=window_type,
        )

        # Store signature in memory.
        signatures.append(signature)

        # Store individual signature on disk.
        filename = baseline_dir / f"baseline_{i + 1:02d}.npz"

        save_signature(signature, filename)

        print(
            f"Captured baseline {i + 1:02d}/{n_boards} "
            f"-> {filename}"
        )

    print()
    print(f"Baseline library created: {baseline_dir.resolve()}")
    print("=" * 60)

    return signatures


# ---------------------------------------------------------------------------
# Build average baseline
# ---------------------------------------------------------------------------

def build_average_baseline(
    signatures: list[Signature],
    output_file: str | Path,
) -> Signature:
    """
    Build an average healthy-board signature from multiple baseline scans.

    Averaging helps reduce random noise and individual-board variation.
    """

    if not signatures:
        raise ValueError("No baseline signatures supplied.")

    # Verify that all signatures have the same frequency bins.
    reference_frequency = signatures[0].frequencies

    for sig in signatures[1:]:
        if not np.allclose(sig.frequencies, reference_frequency):
            raise ValueError(
                "Baseline signatures have incompatible frequency bins."
            )

    magnitudes = np.array([
        sig.magnitude for sig in signatures
    ])

    phases = np.array([
        sig.phase for sig in signatures
    ])

    # Average magnitude using RMS (power) averaging, NOT a plain
    # arithmetic mean.
    #
    # Why this matters: individual boards have their characteristic
    # peaks sitting at slightly different frequency bins because of
    # normal manufacturing jitter (unit_variation in the simulator).
    # A plain arithmetic mean of magnitude spectra smears each peak
    # across neighbouring bins and *shrinks* its height -- badly, for
    # weaker/higher-frequency components. Measured on this project's
    # own data, a naive mean underestimated the true average band
    # energy of the 40 kHz component by ~4.6x, which made every live
    # scan look artificially "louder" than baseline there and was the
    # single biggest cause of healthy boards being reported as FAIL.
    #
    # Averaging power (magnitude^2) instead of amplitude preserves the
    # true average energy regardless of small per-board frequency
    # jitter, because energy doesn't cancel out or shrink under
    # averaging the way signed/positioned amplitude does.
    mean_magnitude = np.sqrt(
        np.mean(magnitudes ** 2, axis=0)
    )

    # Average phase.
    #
    # For this initial implementation we use the arithmetic mean.
    # Circular phase averaging can be added later when phase scoring
    # becomes more important.
    mean_phase = np.mean(phases, axis=0)

    mean_magnitude_db = (
        20 * np.log10(
            np.maximum(mean_magnitude, 1e-12)
        )
    )

    average_signature = Signature(
        frequencies=reference_frequency.copy(),
        magnitude=mean_magnitude,
        magnitude_db=mean_magnitude_db,
        phase=mean_phase,
        sample_rate_hz=signatures[0].sample_rate_hz,
        n_samples=signatures[0].n_samples,
    )

    save_signature(
        average_signature,
        output_file,
    )

    return average_signature


# ---------------------------------------------------------------------------
# Baseline variation
# ---------------------------------------------------------------------------

def calculate_baseline_variation(
    signatures: list[Signature],
) -> dict[str, np.ndarray]:
    """
    Calculate natural variation across known-good boards.

    This will later be used by the threshold-calibration module.

    Returns:
        mean_magnitude
        std_magnitude
        mean_phase
        std_phase
    """

    if not signatures:
        raise ValueError("No baseline signatures supplied.")

    magnitudes = np.array([
        sig.magnitude for sig in signatures
    ])

    phases = np.array([
        sig.phase for sig in signatures
    ])

    return {
        "mean_magnitude": np.mean(magnitudes, axis=0),
        "std_magnitude": np.std(magnitudes, axis=0),
        "mean_phase": np.mean(phases, axis=0),
        "std_phase": np.std(phases, axis=0),
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    cfg = AcquisitionConfig()
    board = BoardProfile()

    # -------------------------------------------------------
    # Step 1: Capture known-good boards
    # -------------------------------------------------------

    signatures = capture_baseline(
        n_boards=10,
        config=cfg,
        board=board,
        baseline_dir="baseline_library",
        window_type="hanning",
        seed=100,
    )

    # -------------------------------------------------------
    # Step 2: Build average baseline
    # -------------------------------------------------------

    average_file = Path(
        "baseline_library/average_baseline.npz"
    )

    average_signature = build_average_baseline(
        signatures,
        average_file,
    )

    print()
    print("Average baseline created:")
    print(f"  File       : {average_file}")
    print(f"  FFT bins   : {len(average_signature.frequencies)}")

    # -------------------------------------------------------
    # Step 3: Calculate healthy-board variation
    # -------------------------------------------------------

    variation = calculate_baseline_variation(signatures)

    print()
    print("Healthy-board variation calculated.")

    print(
        f"Maximum magnitude standard deviation: "
        f"{np.max(variation['std_magnitude']):.6f}"
    )

    print(
        f"Mean magnitude standard deviation: "
        f"{np.mean(variation['std_magnitude']):.6f}"
    )

    print()
    print("Baseline capture complete.")