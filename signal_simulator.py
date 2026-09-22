"""
signal_simulator.py

Synthetic "fake hardware" module for the Portable PCB Health Scanner.

Generates time-domain signals that stand in for what the STM32's ADC would
capture after injecting a test tone/sweep into a board and reading back the
response. Every downstream module (FFT, scoring, threshold, classification,
GUI) only ever consumes a NumPy array of samples + a known sample rate, so
this module is a drop-in replacement for real hardware acquisition. When
real hardware exists, only the `acquire_from_hardware()` stub at the bottom
needs to be wired up via PySerial -- nothing else in the pipeline changes.

Board model
-----------
A "known-good" board response is modeled as a small set of characteristic
frequency components (like resonances / impedance features at particular
frequencies) plus broadband noise. Each fault type is modeled as a specific,
physically-motivated distortion of that baseline:

- Open trace       -> one or more frequency bands are strongly attenuated
                       (the test signal has nowhere to go / no return path,
                       so energy in that band collapses to near-noise-floor).
- Short circuit     -> impedance collapses across a band, which we model as
                       *flattening* amplitude toward a low, near-constant
                       level across that band (loss of frequency-selective
                       response -- everything looks the same, low, value).
- Regulator fault   -> an extra low-frequency ripple tone appears (e.g. at
                       the switching frequency or a sub-harmonic of it) and
                       a phase shift is introduced relative to baseline.
- Capacitor drift   -> a resonance peak shifts in frequency (models ESR/
                       capacitance drift changing a resonant response) --
                       included as a bonus since it's an Extended-tier target.

None of this claims to be a physically rigorous SPICE-level model. It exists
to produce structurally realistic, labeled signatures so the comparison,
thresholding, and classification code can be written and validated before
real hardware is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
# Board / acquisition configuration
# ---------------------------------------------------------------------------

@dataclass
class AcquisitionConfig:
    """Parameters that describe how a signal is 'sampled'."""
    sample_rate_hz: float = 200_000.0   # ADC sample rate
    duration_s: float = 0.01            # capture window length
    noise_std: float = 0.02             # broadband noise (fraction of amplitude)

    @property
    def n_samples(self) -> int:
        return int(self.sample_rate_hz * self.duration_s)

    def time_vector(self) -> np.ndarray:
        return np.arange(self.n_samples) / self.sample_rate_hz


@dataclass
class BoardProfile:
    """
    Characteristic frequency components of a 'known-good' board response.

    Each entry in `components` is (frequency_hz, amplitude, phase_rad).
    Different boards/board-families can have different profiles; small
    random jitter is added per-unit to simulate manufacturing tolerance.
    """
    components: list = field(default_factory=lambda: [
        (1_000.0, 1.0, 0.0),   # low-frequency structural response
        (5_000.0, 0.6, 0.3),   # mid-band feature (e.g. decoupling network)
        (15_000.0, 0.35, 0.8), # higher-band feature
        (40_000.0, 0.15, 1.2), # fine-detail / high-frequency rolloff region
    ])


# ---------------------------------------------------------------------------
# Core synthesis
# ---------------------------------------------------------------------------

def _synthesize(
    t: np.ndarray,
    components: list,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sum sinusoidal components and add Gaussian noise."""
    signal = np.zeros_like(t)
    for freq, amp, phase in components:
        signal += amp * np.sin(2 * np.pi * freq * t + phase)
    signal += rng.normal(0.0, noise_std, size=t.shape)
    return signal


def _jitter_components(components: list, rng: np.random.Generator, jitter: float = 0.03) -> list:
    """Apply small random per-unit variation to simulate manufacturing tolerance."""
    jittered = []
    for freq, amp, phase in components:
        f = freq * (1 + rng.uniform(-jitter, jitter))
        a = amp * (1 + rng.uniform(-jitter, jitter))
        jittered.append((f, a, phase))
    return jittered


# ---------------------------------------------------------------------------
# Public API: generate labeled synthetic scans
# ---------------------------------------------------------------------------

def generate_known_good(
    config: AcquisitionConfig = AcquisitionConfig(),
    board: BoardProfile = BoardProfile(),
    unit_variation: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate a synthetic 'known-good' board response.

    unit_variation: if True, applies small random jitter to simulate normal
    manufacturing tolerance between individual boards of the same design
    (useful for building a population to calibrate thresholds against).
    """
    rng = np.random.default_rng(seed)
    t = config.time_vector()
    components = _jitter_components(board.components, rng) if unit_variation else board.components
    return _synthesize(t, components, config.noise_std, rng)


def generate_open_trace_fault(
    config: AcquisitionConfig = AcquisitionConfig(),
    board: BoardProfile = BoardProfile(),
    band_index: int = 1,
    attenuation_db: float = -30.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate an open trace: the component at `band_index` is heavily
    attenuated, since the test signal has no return path through that
    part of the circuit anymore.
    """
    rng = np.random.default_rng(seed)
    t = config.time_vector()
    components = _jitter_components(board.components, rng)

    freq, amp, phase = components[band_index]
    attenuation_factor = 10 ** (attenuation_db / 20)
    components[band_index] = (freq, amp * attenuation_factor, phase)

    return _synthesize(t, components, config.noise_std, rng)


def generate_short_circuit_fault(
    config: AcquisitionConfig = AcquisitionConfig(),
    board: BoardProfile = BoardProfile(),
    band_indices: tuple = (1, 2),
    flatten_level: float = 0.1,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate a short circuit: impedance collapses across a band, modeled as
    flattening the affected components down toward a common low amplitude
    (loss of frequency-selective response).
    """
    rng = np.random.default_rng(seed)
    t = config.time_vector()
    components = _jitter_components(board.components, rng)

    for idx in band_indices:
        freq, amp, phase = components[idx]
        components[idx] = (freq, flatten_level, phase)

    return _synthesize(t, components, config.noise_std, rng)


def generate_regulator_fault(
    config: AcquisitionConfig = AcquisitionConfig(),
    board: BoardProfile = BoardProfile(),
    ripple_freq_hz: float = 300.0,
    ripple_amp: float = 0.4,
    phase_shift_rad: float = 1.5,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate a faulty regulator: an extra low-frequency ripple tone appears
    (e.g. at or near the switching frequency), and existing components pick
    up an added phase shift relative to baseline.
    """
    rng = np.random.default_rng(seed)
    t = config.time_vector()
    components = _jitter_components(board.components, rng)

    # Add phase shift to existing components
    shifted = [(f, a, p + phase_shift_rad) for f, a, p in components]
    # Inject the extra ripple tone
    shifted.append((ripple_freq_hz, ripple_amp, 0.0))

    return _synthesize(t, shifted, config.noise_std, rng)


def generate_capacitor_drift_fault(
    config: AcquisitionConfig = AcquisitionConfig(),
    board: BoardProfile = BoardProfile(),
    band_index: int = 2,
    freq_shift_fraction: float = -0.25,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate capacitor degradation as a resonance-frequency shift on one
    component (bonus / Extended-tier fault type -- explicitly a heuristic
    stand-in for true ESR drift, not a lab-grade measurement).
    """
    rng = np.random.default_rng(seed)
    t = config.time_vector()
    components = _jitter_components(board.components, rng)

    freq, amp, phase = components[band_index]
    components[band_index] = (freq * (1 + freq_shift_fraction), amp, phase)

    return _synthesize(t, components, config.noise_std, rng)


# ---------------------------------------------------------------------------
# Dataset helpers (useful for threshold calibration + ML training later)
# ---------------------------------------------------------------------------

FAULT_GENERATORS = {
    "good": generate_known_good,
    "open_trace": generate_open_trace_fault,
    "short_circuit": generate_short_circuit_fault,
    "regulator_fault": generate_regulator_fault,
    "capacitor_drift": generate_capacitor_drift_fault,
}


def generate_labeled_batch(
    n_per_class: int = 20,
    config: AcquisitionConfig = AcquisitionConfig(),
    board: BoardProfile = BoardProfile(),
    base_seed: int = 0,
) -> list[tuple[str, np.ndarray]]:
    """
    Generate a batch of (label, signal) pairs across all classes.
    Useful for threshold calibration (the 'good' subset) and later for
    training the ML classifier (all classes, with feature extraction).
    """
    dataset = []
    seed_counter = base_seed
    for label, generator in FAULT_GENERATORS.items():
        for _ in range(n_per_class):
            signal = generator(config=config, board=board, seed=seed_counter)
            dataset.append((label, signal))
            seed_counter += 1
    return dataset


# ---------------------------------------------------------------------------
# Hardware acquisition stub (fill in later, nothing else needs to change)
# ---------------------------------------------------------------------------

def acquire_from_hardware(port: str, config: AcquisitionConfig) -> np.ndarray:
    """
    Placeholder for real acquisition via PySerial once hardware exists.
    Should return a NumPy array of the same shape/meaning as the
    generate_* functions above: raw time-domain samples at
    config.sample_rate_hz for config.duration_s seconds.
    """
    raise NotImplementedError(
        "Hardware not connected yet -- use the generate_* functions in "
        "this module as a stand-in until the STM32 acquisition path is wired up."
    )


# ---------------------------------------------------------------------------
# Quick self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = AcquisitionConfig()
    board = BoardProfile()

    good = generate_known_good(cfg, board, seed=1)
    open_fault = generate_open_trace_fault(cfg, board, seed=2)
    short_fault = generate_short_circuit_fault(cfg, board, seed=3)
    reg_fault = generate_regulator_fault(cfg, board, seed=4)
    cap_fault = generate_capacitor_drift_fault(cfg, board, seed=5)

    print(f"Samples per capture: {cfg.n_samples}")
    for label, sig in [
        ("good", good),
        ("open_trace", open_fault),
        ("short_circuit", short_fault),
        ("regulator_fault", reg_fault),
        ("capacitor_drift", cap_fault),
    ]:
        print(f"{label:16s} shape={sig.shape} rms={np.sqrt(np.mean(sig**2)):.4f}")

    batch = generate_labeled_batch(n_per_class=5)
    print(f"\nGenerated labeled batch: {len(batch)} signals "
          f"({len(FAULT_GENERATORS)} classes x 5 each)")