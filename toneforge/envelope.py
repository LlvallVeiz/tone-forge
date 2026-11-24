"""ADSR amplitude envelopes.

ADSR shapes how loud a note is over time, in four stages:

    A  Attack:  silence -> peak (``attack`` seconds)
    D  Decay:   peak -> sustain level (``decay`` seconds)
    S  Sustain: holds at ``sustain_level`` while the note is held
    R  Release: sustain level -> silence (``release`` seconds)

The envelope always starts and ends at exactly 0, so applying it never
creates a click at the start or end of a buffer.
"""

from __future__ import annotations

import numpy as np

__all__ = ["adsr_envelope", "apply_adsr"]


def adsr_envelope(n_samples: int, sample_rate: int,
                  attack: float = 0.01, decay: float = 0.1,
                  sustain_level: float = 0.7, release: float = 0.1) -> np.ndarray:
    """Build an ADSR envelope of ``n_samples`` samples (values in [0, 1]).

    The note is considered "held" for the whole buffer except the final
    ``release`` seconds, so the sustain section fills the middle and the
    release ramps down to 0 at the last sample.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not 0.0 <= sustain_level <= 1.0:
        raise ValueError("sustain_level must be in [0, 1]")
    for name, value in (("attack", attack), ("decay", decay),
                        ("release", release)):
        if value < 0:
            raise ValueError(f"{name} must be >= 0")

    envelope = np.zeros(n_samples, dtype=np.float64)

    n_attack = int(round(attack * sample_rate))
    n_decay = int(round(decay * sample_rate))
    n_release = int(round(release * sample_rate))
    # The release section is clamped so it never overlaps the attack/decay.
    n_release = min(n_release, n_samples)
    n_sustain = max(n_samples - n_attack - n_decay - n_release, 0)

    idx = 0
    # Attack: linear ramp 0 -> 1. Endpoint excluded so the peak is hit
    # exactly once, at the start of the decay section.
    if n_attack > 0:
        envelope[idx:idx + n_attack] = np.linspace(
            0.0, 1.0, n_attack, endpoint=False)
        idx += n_attack
    # Decay: linear ramp 1 -> sustain_level.
    if n_decay > 0:
        envelope[idx:idx + n_decay] = np.linspace(
            1.0, sustain_level, n_decay, endpoint=False)
        idx += n_decay
    # Sustain: flat at sustain_level.
    if n_sustain > 0:
        envelope[idx:idx + n_sustain] = sustain_level
        idx += n_sustain
    # Release: linear ramp sustain_level -> 0, ending exactly at 0.
    if n_release > 0:
        envelope[idx:idx + n_release] = np.linspace(
            sustain_level, 0.0, n_release, endpoint=False)
        idx += n_release

    # Any leftover samples (rounding) hold the sustain level, except the
    # very last sample, which must be 0 so the envelope ends in silence.
    if idx < n_samples:
        envelope[idx:] = sustain_level
    envelope[-1] = 0.0
    envelope[0] = 0.0
    return envelope


def apply_adsr(samples: np.ndarray, sample_rate: int,
               attack: float = 0.01, decay: float = 0.1,
               sustain_level: float = 0.7, release: float = 0.1) -> np.ndarray:
    """Multiply ``samples`` by an ADSR envelope of the same length."""
    env = adsr_envelope(len(samples), sample_rate,
                        attack=attack, decay=decay,
                        sustain_level=sustain_level, release=release)
    return samples * env
