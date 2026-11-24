"""tone-forge: a small music/audio synthesis toolkit.

Generate tones, melodies and WAV files from pure Python + numpy, and
analyse them with a simple FFT. No external audio libraries are needed:
WAV I/O uses the standard library ``wave`` module.
"""

__all__ = [
    "waveforms",
    "notes",
    "envelope",
    "melody",
    "wav_io",
    "analyze",
    "cli",
]

__version__ = "0.1.0"
