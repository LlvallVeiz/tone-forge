"""Waveform generators.

Every function returns a 1-D ``numpy`` float array of samples in [-1, 1]
for a tone of the given frequency, duration, sample rate and amplitude.

The four classic waveforms:
- sine:     smooth fundamental only (pure tone)
- square:   odd harmonics only, sounds hollow/buzzy
- sawtooth: all harmonics, sounds bright/brassy
- triangle: odd harmonics decaying quickly, sounds soft/flute-like
"""

from __future__ import annotations

import numpy as np

__all__ = ["sine", "square", "sawtooth", "triangle", "WAVEFORMS"]


def _time_axis(duration: float, sample_rate: int) -> np.ndarray:
    """Sample times 0, 1/sr, 2/sr, ... for ``duration`` seconds."""
    n = int(round(duration * sample_rate))
    return np.arange(n, dtype=np.float64) / sample_rate


def _validate(frequency: float, duration: float, sample_rate: int,
              amplitude: float) -> None:
    if frequency <= 0:
        raise ValueError("frequency must be positive")
    if duration <= 0:
        raise ValueError("duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not 0.0 < amplitude <= 1.0:
        raise ValueError("amplitude must be in (0, 1]")


def sine(frequency: float, duration: float, sample_rate: int = 44100,
         amplitude: float = 0.8) -> np.ndarray:
    """Pure sine wave: ``amplitude * sin(2*pi*f*t)``."""
    _validate(frequency, duration, sample_rate, amplitude)
    t = _time_axis(duration, sample_rate)
    return amplitude * np.sin(2.0 * np.pi * frequency * t)


def square(frequency: float, duration: float, sample_rate: int = 44100,
           amplitude: float = 0.8) -> np.ndarray:
    """Square wave: sign of the sine (``+amplitude`` / ``-amplitude``).

    Equivalent to the sum of odd harmonics 1, 3, 5, ... with amplitudes
    1/n. Computed via ``sign(sin(...))``; the exact zero crossings of
    the sine land on 0, which keeps the mean at zero.
    """
    _validate(frequency, duration, sample_rate, amplitude)
    t = _time_axis(duration, sample_rate)
    return amplitude * np.sign(np.sin(2.0 * np.pi * frequency * t))


def sawtooth(frequency: float, duration: float, sample_rate: int = 44100,
             amplitude: float = 0.8) -> np.ndarray:
    """Sawtooth wave: linear ramp from -amplitude to +amplitude each period.

    ``2 * frac(f*t) - 1`` rises linearly from -1 to +1 once per cycle.
    Contains all harmonics 1, 2, 3, ... with amplitudes 1/n.
    """
    _validate(frequency, duration, sample_rate, amplitude)
    t = _time_axis(duration, sample_rate)
    phase = (frequency * t) % 1.0  # fractional part of the cycle count
    return amplitude * (2.0 * phase - 1.0)


def triangle(frequency: float, duration: float, sample_rate: int = 44100,
             amplitude: float = 0.8) -> np.ndarray:
    """Triangle wave: linear up/down ramp each period.

    Built from the sawtooth phase: ``2*|2*phase - 1| - 1`` folds the
    ramp into a triangle. Contains only odd harmonics with amplitudes
    1/n**2, so it sounds much softer than the square wave.
    """
    _validate(frequency, duration, sample_rate, amplitude)
    t = _time_axis(duration, sample_rate)
    phase = (frequency * t) % 1.0
    return amplitude * (2.0 * np.abs(2.0 * phase - 1.0) - 1.0)


WAVEFORMS = {
    "sine": sine,
    "square": square,
    "sawtooth": sawtooth,
    "triangle": triangle,
}
