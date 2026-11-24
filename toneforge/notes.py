"""Musical notes: MIDI numbers, note names, and frequencies.

Equal temperament (the tuning used by virtually all modern music):
each semitone multiplies the frequency by the twelfth root of 2, so

    f = 440 * 2 ** ((midi - 69) / 12)

with A4 = MIDI 69 = 440 Hz as the reference pitch. MIDI 60 is middle C
(C4, ~261.63 Hz).
"""

from __future__ import annotations

import math
import re

__all__ = [
    "A4_FREQUENCY",
    "A4_MIDI",
    "midi_to_frequency",
    "frequency_to_midi",
    "note_name_to_midi",
    "midi_to_note_name",
    "frequency_to_note",
]

A4_FREQUENCY = 440.0
A4_MIDI = 69

# Semitone offsets from C within an octave.
_NOTE_OFFSETS = {
    "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
}

# Sharp spelling used when converting a MIDI number back to a name.
_SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F",
                "F#", "G", "G#", "A", "A#", "B"]

# Matches e.g. "A4", "C#5", "Bb3", "f#2" (letter, optional accidental, octave).
_NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")


def midi_to_frequency(midi: int) -> float:
    """Frequency in Hz of a MIDI note number (equal temperament, A4=440 Hz)."""
    return A4_FREQUENCY * 2.0 ** ((midi - A4_MIDI) / 12.0)


def frequency_to_midi(frequency: float) -> float:
    """Fractional MIDI number of a frequency (69.0 == 440 Hz exactly)."""
    if frequency <= 0:
        raise ValueError("frequency must be positive")
    # Inverse of midi_to_frequency: solve 440 * 2**((m-69)/12) = f.
    return A4_MIDI + 12.0 * math.log2(frequency / A4_FREQUENCY)


def note_name_to_midi(name: str) -> int:
    """Parse a note name like ``"A4"``, ``"C#5"`` or ``"Bb3"`` to a MIDI number.

    Sharps (``#``) raise the note by one semitone, flats (``b``) lower it
    by one. Octave numbers follow scientific pitch notation (C4 = middle C).
    """
    match = _NOTE_RE.match(name.strip())
    if match is None:
        raise ValueError(f"invalid note name: {name!r} "
                         "(expected e.g. 'A4', 'C#5', 'Bb3')")
    letter, accidental, octave_str = match.groups()
    semitone = _NOTE_OFFSETS[letter.upper()]
    if accidental == "#":
        semitone += 1
    elif accidental == "b":
        semitone -= 1
    octave = int(octave_str)
    # MIDI 0 is C-1, so MIDI = (octave + 1) * 12 + semitone.
    return (octave + 1) * 12 + semitone


def midi_to_note_name(midi: int) -> str:
    """MIDI number to a note name, spelled with sharps (e.g. 61 -> 'C#4')."""
    if not 0 <= midi <= 127:
        raise ValueError("MIDI note must be in 0..127")
    octave = midi // 12 - 1
    return f"{_SHARP_NAMES[midi % 12]}{octave}"


def frequency_to_note(frequency: float) -> tuple[str, float]:
    """Nearest note name for a frequency, plus the detune in cents.

    Returns ``(name, cents)`` where cents is the deviation from the
    equal-tempered pitch (+100 cents = one semitone sharp). 440 Hz
    returns ``("A4", 0.0)``.
    """
    midi_float = frequency_to_midi(frequency)
    nearest = int(round(midi_float))
    cents = (midi_float - nearest) * 100.0
    return midi_to_note_name(nearest), cents
