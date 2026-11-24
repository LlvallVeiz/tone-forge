"""FFT analysis: find the dominant frequency of a sample buffer.

A Hann window is applied before the real FFT (``numpy.fft.rfft``) to
reduce spectral leakage from the abrupt start/end of a finite buffer.
The peak of the magnitude spectrum is the dominant frequency; with a
pure tone it lands within one FFT bin of the true frequency
(bin width = sample_rate / n_samples).
"""

from __future__ import annotations

import numpy as np

__all__ = ["peak_frequency", "magnitude_spectrum", "SpectrumSummary"]


class SpectrumSummary:
    """Summary of a magnitude spectrum."""

    def __init__(self, frequencies: np.ndarray, magnitudes: np.ndarray):
        self.frequencies = frequencies
        self.magnitudes = magnitudes

    @property
    def peak_frequency(self) -> float:
        """Frequency (Hz) of the largest magnitude bin."""
        return float(self.frequencies[int(np.argmax(self.magnitudes))])

    @property
    def peak_magnitude(self) -> float:
        """Magnitude of the largest bin."""
        return float(np.max(self.magnitudes))

    @property
    def total_energy(self) -> float:
        """Sum of squared magnitudes (Parseval: proportional to signal power)."""
        return float(np.sum(self.magnitudes ** 2))

    @property
    def bin_width(self) -> float:
        """Frequency spacing between adjacent bins, in Hz."""
        if len(self.frequencies) < 2:
            return 0.0
        return float(self.frequencies[1] - self.frequencies[0])


def magnitude_spectrum(samples: np.ndarray,
                       sample_rate: int) -> SpectrumSummary:
    """Magnitude spectrum of ``samples`` (Hann window + rfft)."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1 or len(samples) == 0:
        raise ValueError("samples must be a non-empty 1-D array")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    windowed = samples * np.hanning(len(samples))
    spectrum = np.fft.rfft(windowed)
    magnitudes = np.abs(spectrum)
    frequencies = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    return SpectrumSummary(frequencies, magnitudes)


def peak_frequency(samples: np.ndarray, sample_rate: int) -> float:
    """Dominant frequency (Hz) of ``samples``."""
    return magnitude_spectrum(samples, sample_rate).peak_frequency
