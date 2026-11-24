"""16-bit PCM WAV input/output via the standard library ``wave`` module.

Float samples in [-1, 1] are stored as signed 16-bit integers
(``int16``), the most widely supported WAV format:

    int16 = round(float * 32767)

clipped to [-32768, 32767] first. Reading converts back to float, so a
write -> read round-trip preserves the sample rate, the sample count,
and the samples themselves to within one int16 step (~3e-5).
"""

from __future__ import annotations

import wave

import numpy as np

__all__ = ["write_wav", "read_wav"]

_INT16_MAX = 32767


def write_wav(path: str, samples: np.ndarray, sample_rate: int = 44100) -> None:
    """Write mono float samples in [-1, 1] to a 16-bit PCM WAV file."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError("only mono (1-D) samples are supported")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.round(clipped * _INT16_MAX).astype("<i2")
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 2 bytes = 16 bits
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a mono 16-bit PCM WAV file.

    Returns ``(samples, sample_rate)`` with samples as floats in [-1, 1].
    """
    with wave.open(path, "rb") as wav:
        n_channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        sample_rate = wav.getframerate()
        n_frames = wav.getnframes()
        raw = wav.readframes(n_frames)
    if sampwidth != 2:
        raise ValueError(f"only 16-bit WAV files are supported, got "
                         f"{sampwidth * 8}-bit")
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    if n_channels > 1:
        # Down-mix multi-channel audio to mono by averaging channels.
        pcm = pcm.reshape(-1, n_channels).mean(axis=1)
    return pcm / _INT16_MAX, sample_rate
