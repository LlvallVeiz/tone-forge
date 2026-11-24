"""Melody sequencer: turn a list of notes into one sample buffer.

A melody is a list of ``(note, beats)`` tuples, where ``note`` is a note
name (``"C4"``), a MIDI number, or a frequency in Hz, and ``beats`` is
how long the note lasts. Notes play one after another; a chord is a
list of notes in place of a single note, e.g. ``(["C4", "E4"], 2)``.
"""

from __future__ import annotations

import numpy as np

from . import envelope, notes, waveforms

__all__ = ["render_melody", "parse_melody_string"]

_DEFAULT_WAVEFORM = "sine"


def _note_to_frequency(note) -> float:
    """Accept a note name, MIDI number, or raw frequency in Hz."""
    if isinstance(note, str):
        return notes.midi_to_frequency(notes.note_name_to_midi(note))
    if isinstance(note, (int, np.integer)):
        return notes.midi_to_frequency(int(note))
    # Anything else is treated as a frequency in Hz.
    frequency = float(note)
    if frequency <= 0:
        raise ValueError("frequency must be positive")
    return frequency


def _render_note(frequency: float, duration: float, sample_rate: int,
                 waveform: str, amplitude: float,
                 attack: float, decay: float,
                 sustain_level: float, release: float) -> np.ndarray:
    generator = waveforms.WAVEFORMS[waveform]
    samples = generator(frequency, duration, sample_rate, amplitude)
    return envelope.apply_adsr(samples, sample_rate,
                               attack=attack, decay=decay,
                               sustain_level=sustain_level,
                               release=release)


def render_melody(events, bpm: float = 120.0, sample_rate: int = 44100,
                  waveform: str = _DEFAULT_WAVEFORM, amplitude: float = 0.8,
                  attack: float = 0.01, decay: float = 0.1,
                  sustain_level: float = 0.7, release: float = 0.1) -> np.ndarray:
    """Render ``events`` (``(note, beats)`` pairs) into a float buffer.

    ``beats`` are quarter-note beats at ``bpm``; one beat lasts
    ``60 / bpm`` seconds. Chord notes (a list in place of a single note)
    are mixed together. The final mix is normalised by its peak so the
    output always stays within [-1, 1] — overlapping notes can never
    clip.
    """
    if bpm <= 0:
        raise ValueError("bpm must be positive")
    if waveform not in waveforms.WAVEFORMS:
        raise ValueError(f"unknown waveform: {waveform!r}")

    seconds_per_beat = 60.0 / bpm
    segments = []
    for note, beats in events:
        if beats <= 0:
            raise ValueError("beats must be positive")
        duration = beats * seconds_per_beat
        chord = note if isinstance(note, (list, tuple)) else [note]
        # Mix chord notes at equal level, then normalise the chord.
        mixed = np.zeros(int(round(duration * sample_rate)),
                         dtype=np.float64)
        for single in chord:
            freq = _note_to_frequency(single)
            mixed += _render_note(freq, duration, sample_rate, waveform,
                                  amplitude, attack, decay,
                                  sustain_level, release)
        peak = np.max(np.abs(mixed))
        if peak > 1.0:
            mixed /= peak
        segments.append(mixed)

    if not segments:
        return np.zeros(0, dtype=np.float64)
    out = np.concatenate(segments)
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out /= peak
    return out


def parse_melody_string(text: str) -> list[tuple[str, float]]:
    """Parse e.g. ``"C4:1 E4:1 G4:2"`` into ``[("C4", 1.0), ...]``.

    Each token is ``NOTE:BEATS``; ``:BEATS`` may be omitted (default 1).
    Chords use ``+``: ``"C4+E4+G4:2"`` plays a C-major triad for 2 beats.
    """
    events = []
    for token in text.split():
        if ":" in token:
            note_part, beats_part = token.rsplit(":", 1)
            beats = float(beats_part)
        else:
            note_part, beats = token, 1.0
        chord = note_part.split("+")
        note = chord if len(chord) > 1 else chord[0]
        events.append((note, beats))
    return events
