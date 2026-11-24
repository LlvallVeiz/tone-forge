# tone-forge

A small music/audio synthesis toolkit in pure Python (stdlib `wave` + numpy).
Generate tones and melodies, save them as 16-bit PCM WAV files, and analyse
them with an FFT — no audio libraries required.

## Features

- **Waveforms** — sine, square, sawtooth and triangle generators returning
  numpy float arrays in [-1, 1].
- **Notes** — MIDI numbers, note names (`"A4"`, `"C#5"`, `"Bb3"`) and
  frequencies, using equal temperament.
- **ADSR envelope** — attack / decay / sustain / release shaping that
  always starts and ends at zero (no clicks).
- **Melody sequencer** — render `(note, beats)` sequences (with chords) at
  any BPM; the mix is normalised so it can never clip.
- **WAV I/O** — 16-bit PCM read/write via the stdlib `wave` module.
- **FFT analysis** — Hann-windowed `rfft` peak-frequency detection, with the
  nearest note name and detune in cents.
- **CLI** — `tone`, `melody` and `analyze` subcommands.

## Install

```bash
pip install numpy
```

Python 3.10+ is required. Nothing else.

## CLI examples

```bash
# A 1-second 440 Hz sine tone (A4)
python -m toneforge.cli tone 440 1 a4.wav

# A bright sawtooth tone, quieter, with a slower attack
python -m toneforge.cli tone 523.25 2 c5.wav --waveform sawtooth \
    --amplitude 0.5 --attack 0.05

# A C-major arpeggio at 120 BPM (NOTE:BEATS, chords joined with +)
python -m toneforge.cli melody "C4:1 E4:1 G4:1 C5:2" arpeggio.wav
python -m toneforge.cli melody "C4+E4+G4:2 G4:1" chord.wav --bpm 90

# Analyse a file: peak frequency, nearest note, spectrum summary
python -m toneforge.cli analyze a4.wav
```

## Library usage

```python
from toneforge import waveforms, notes, melody, wav_io, analyze

# Middle C as a 0.5 s triangle wave
samples = waveforms.triangle(notes.midi_to_frequency(60), 0.5, 44100, 0.8)

# "Ode to Joy" opening, rendered at 120 BPM
song = melody.render_melody(
    [("E4", 1), ("E4", 1), ("F4", 1), ("G4", 1),
     ("G4", 1), ("F4", 1), ("E4", 1), ("D4", 1)],
    bpm=120)
wav_io.write_wav("ode.wav", song, 44100)

# What frequency dominates this file?
samples, sr = wav_io.read_wav("ode.wav")
print(analyze.peak_frequency(samples, sr))
print(notes.frequency_to_note(440.0))  # ('A4', 0.0)
```

## The math

### Equal temperament

In equal temperament every semitone multiplies the frequency by the
twelfth root of 2, so a MIDI note number `m` has frequency

```
f = 440 * 2 ** ((m - 69) / 12)
```

with A4 = MIDI 69 = 440 Hz as the reference. MIDI 60 (middle C, C4) is
therefore 440 * 2**(-9/12) ≈ 261.63 Hz.

### ADSR envelope

A note's loudness over time follows four stages:

```
amplitude
  ^  A       D          S           R
  | / \     / \        /‾‾‾‾‾‾‾‾‾\ /
  |/   \   /   \______/           X
  +-----+-------+---------------+----> time
  0  attack  decay  sustain   release
```

- **Attack** — ramp from silence to full volume.
- **Decay** — fall from full volume to the sustain level.
- **Sustain** — hold at the sustain level while the note is held.
- **Release** — fade back to silence when the note ends.

The envelope starts and ends at exactly 0, so a shaped note never
clicks at its boundaries.

### FFT peak detection

`analyze` applies a Hann window (to reduce spectral leakage from the
finite buffer edges) and then `numpy.fft.rfft`. The bin with the largest
magnitude is the dominant frequency; the bin width is
`sample_rate / n_samples`, so a 1-second clip at 44.1 kHz resolves
frequencies to about 1 Hz.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover MIDI/frequency conversion, note-name parsing (sharps and
flats), WAV round-trips, FFT peak accuracy within one bin, ADSR
start/end at zero, and melody mixing staying within [-1, 1].

## Project structure

```
toneforge/
  waveforms.py   sine / square / sawtooth / triangle generators
  notes.py       MIDI <-> frequency <-> note-name conversion
  envelope.py    ADSR envelope
  melody.py      (note, beats) sequencer with chord mixing
  wav_io.py      16-bit PCM WAV read/write (stdlib wave)
  analyze.py     FFT peak-frequency analysis
  cli.py         tone / melody / analyze command-line interface
tests/
  test_toneforge.py
```
