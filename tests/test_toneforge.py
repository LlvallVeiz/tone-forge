"""Tests for tone-forge.

Covers the core guarantees of each module:
- notes: MIDI <-> frequency conversion and note-name parsing
- waveforms: ranges, sample counts, basic shape properties
- envelope: ADSR starts and ends at zero
- melody: mixed output never clips
- wav_io: round-trip preserves sample rate, count and values
- analyze: FFT peak matches the generated frequency within one bin
"""

import os
import sys
import tempfile
import unittest

import numpy as np

# Allow running the suite from the project root (or anywhere).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toneforge import analyze, envelope, melody, notes, wav_io, waveforms

SR = 44100


class TestNotes(unittest.TestCase):
    def test_midi_69_is_440_hz(self):
        self.assertAlmostEqual(notes.midi_to_frequency(69), 440.0, places=9)

    def test_midi_60_is_middle_c(self):
        # C4 = 440 * 2**(-9/12) ≈ 261.6256 Hz.
        self.assertAlmostEqual(notes.midi_to_frequency(60), 261.6256,
                               places=3)

    def test_octave_doubles_frequency(self):
        self.assertAlmostEqual(notes.midi_to_frequency(81),
                               2 * notes.midi_to_frequency(69), places=9)

    def test_frequency_to_midi_round_trip(self):
        for midi in (21, 60, 69, 72, 108):
            self.assertAlmostEqual(
                notes.frequency_to_midi(notes.midi_to_frequency(midi)),
                midi, places=9)

    def test_note_name_parsing(self):
        self.assertEqual(notes.note_name_to_midi("A4"), 69)
        self.assertEqual(notes.note_name_to_midi("C4"), 60)
        self.assertEqual(notes.note_name_to_midi("C0"), 12)
        self.assertEqual(notes.note_name_to_midi("C-1"), 0)

    def test_note_name_sharps_and_flats(self):
        self.assertEqual(notes.note_name_to_midi("C#5"), 73)
        self.assertEqual(notes.note_name_to_midi("F#3"), 54)
        self.assertEqual(notes.note_name_to_midi("Bb3"), 58)
        self.assertEqual(notes.note_name_to_midi("Eb4"), 63)
        # Enharmonic equivalents: C# == Db, Bb == A#.
        self.assertEqual(notes.note_name_to_midi("C#4"),
                         notes.note_name_to_midi("Db4"))
        self.assertEqual(notes.note_name_to_midi("Bb3"),
                         notes.note_name_to_midi("A#3"))

    def test_note_name_invalid(self):
        for bad in ("H4", "C", "A#", "4A", "C##4", ""):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    notes.note_name_to_midi(bad)

    def test_frequency_to_note(self):
        name, cents = notes.frequency_to_note(440.0)
        self.assertEqual(name, "A4")
        self.assertAlmostEqual(cents, 0.0, places=6)
        name, cents = notes.frequency_to_note(261.63)
        self.assertEqual(name, "C4")
        self.assertLess(abs(cents), 1.0)

    def test_midi_to_note_name(self):
        self.assertEqual(notes.midi_to_note_name(69), "A4")
        self.assertEqual(notes.midi_to_note_name(60), "C4")
        self.assertEqual(notes.midi_to_note_name(61), "C#4")


class TestWaveforms(unittest.TestCase):
    def test_sample_count(self):
        for gen in (waveforms.sine, waveforms.square,
                    waveforms.sawtooth, waveforms.triangle):
            with self.subTest(gen=gen.__name__):
                self.assertEqual(len(gen(440.0, 0.5, SR)), SR // 2)

    def test_range_within_unit(self):
        for gen in (waveforms.sine, waveforms.square,
                    waveforms.sawtooth, waveforms.triangle):
            with self.subTest(gen=gen.__name__):
                samples = gen(440.0, 0.1, SR, amplitude=0.8)
                self.assertTrue(np.all(np.abs(samples) <= 0.8 + 1e-12))

    def test_sine_is_periodic(self):
        # One period of 440 Hz at 44100 Hz is 100.227... samples; compare
        # against the analytic value instead of exact periodicity.
        samples = waveforms.sine(440.0, 0.01, SR)
        t = np.arange(len(samples)) / SR
        np.testing.assert_allclose(samples, 0.8 * np.sin(2 * np.pi * 440 * t))

    def test_square_is_two_level(self):
        samples = waveforms.square(440.0, 0.1, SR, amplitude=0.5)
        levels = np.unique(np.round(samples, 9))
        self.assertEqual(set(levels.tolist()), {-0.5, 0.0, 0.5})

    def test_sawtooth_ramps(self):
        # Over one period the sawtooth rises monotonically from -a to +a.
        samples = waveforms.sawtooth(100.0, 0.01, SR, amplitude=1.0)
        period = SR // 100
        first_period = samples[:period]
        self.assertLess(first_period[0], first_period[-1])
        self.assertTrue(np.all(np.diff(first_period) >= -1e-12))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            waveforms.sine(0, 1.0, SR)
        with self.assertRaises(ValueError):
            waveforms.sine(440.0, 0.0, SR)
        with self.assertRaises(ValueError):
            waveforms.sine(440.0, 1.0, SR, amplitude=1.5)


class TestEnvelope(unittest.TestCase):
    def test_starts_and_ends_at_zero(self):
        env = envelope.adsr_envelope(SR, SR)
        self.assertEqual(env[0], 0.0)
        self.assertEqual(env[-1], 0.0)

    def test_range_and_peak(self):
        env = envelope.adsr_envelope(SR, SR, attack=0.05, decay=0.1,
                                     sustain_level=0.6, release=0.2)
        self.assertTrue(np.all((env >= 0.0) & (env <= 1.0)))
        self.assertAlmostEqual(env.max(), 1.0, places=9)

    def test_sustain_level_held(self):
        env = envelope.adsr_envelope(SR, SR, attack=0.01, decay=0.01,
                                     sustain_level=0.5, release=0.01)
        mid = env[SR // 2]
        self.assertAlmostEqual(mid, 0.5, places=9)

    def test_apply_adsr_scales_and_zeroes_ends(self):
        samples = np.ones(SR)
        shaped = envelope.apply_adsr(samples, SR)
        self.assertEqual(shaped[0], 0.0)
        self.assertEqual(shaped[-1], 0.0)
        self.assertTrue(np.all(np.abs(shaped) <= 1.0))


class TestMelody(unittest.TestCase):
    def test_length_matches_bpm(self):
        # 4 beats at 120 BPM = 2 seconds.
        out = melody.render_melody([("C4", 1), ("E4", 1), ("G4", 1),
                                    ("C5", 1)], bpm=120.0, sample_rate=SR)
        self.assertEqual(len(out), 2 * SR)

    def test_mixed_output_never_clips(self):
        # A loud 3-note chord at full amplitude, repeated: worst case
        # for clipping if normalisation were missing.
        events = [(["C4", "E4", "G4"], 2), (["C4", "E4", "G4"], 2),
                  (["A4", "C5", "E5"], 2)]
        out = melody.render_melody(events, bpm=120.0, sample_rate=SR,
                                   amplitude=1.0, waveform="square")
        self.assertTrue(np.all(np.abs(out) <= 1.0 + 1e-12))
        self.assertGreater(np.max(np.abs(out)), 0.5)

    def test_melody_parsing(self):
        events = melody.parse_melody_string("C4:1 E4:1 G4:2")
        self.assertEqual(events, [("C4", 1.0), ("E4", 1.0), ("G4", 2.0)])
        events = melody.parse_melody_string("C4+E4+G4:2 G4")
        self.assertEqual(events, [(["C4", "E4", "G4"], 2.0), ("G4", 1.0)])

    def test_midi_and_frequency_notes(self):
        out = melody.render_melody([(69, 1), (440.0, 1)], bpm=120.0,
                                   sample_rate=SR)
        self.assertEqual(len(out), SR)  # 2 beats at 120 BPM = 1 second
        self.assertTrue(np.all(np.abs(out) <= 1.0))


class TestWavIO(unittest.TestCase):
    def test_round_trip(self):
        original = waveforms.sine(440.0, 0.5, SR, amplitude=0.8)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tone.wav")
            wav_io.write_wav(path, original, SR)
            samples, rate = wav_io.read_wav(path)
        self.assertEqual(rate, SR)
        self.assertEqual(len(samples), len(original))
        # int16 quantisation: one step is 1/32767 ≈ 3.05e-5.
        np.testing.assert_allclose(samples, original, atol=1.0 / 32767 + 1e-9)

    def test_round_trip_different_sample_rate(self):
        original = waveforms.sine(220.0, 0.25, 22050, amplitude=0.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "low.wav")
            wav_io.write_wav(path, original, 22050)
            samples, rate = wav_io.read_wav(path)
        self.assertEqual(rate, 22050)
        self.assertEqual(len(samples), len(original))

    def test_silence_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "silence.wav")
            wav_io.write_wav(path, np.zeros(1000), SR)
            samples, rate = wav_io.read_wav(path)
        self.assertEqual(rate, SR)
        self.assertTrue(np.all(samples == 0.0))


class TestAnalyze(unittest.TestCase):
    def test_fft_peak_matches_generated_frequency(self):
        freq = 440.0
        samples = waveforms.sine(freq, 1.0, SR, amplitude=0.8)
        summary = analyze.magnitude_spectrum(samples, SR)
        # Bin width is 1 Hz for a 1-second clip; allow one bin of error.
        self.assertLessEqual(abs(summary.peak_frequency - freq),
                             summary.bin_width)

    def test_fft_peak_other_frequencies(self):
        for freq in (220.0, 523.25, 1000.0):
            with self.subTest(freq=freq):
                samples = waveforms.sine(freq, 1.0, SR, amplitude=0.8)
                peak = analyze.peak_frequency(samples, SR)
                self.assertLessEqual(abs(peak - freq), 1.0)

    def test_fft_peak_on_wav_file(self):
        original = waveforms.sine(880.0, 1.0, SR, amplitude=0.8)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a5.wav")
            wav_io.write_wav(path, original, SR)
            samples, rate = wav_io.read_wav(path)
        peak = analyze.peak_frequency(samples, rate)
        self.assertLessEqual(abs(peak - 880.0), 1.0)

    def test_spectrum_summary(self):
        samples = waveforms.sine(440.0, 1.0, SR, amplitude=0.8)
        summary = analyze.magnitude_spectrum(samples, SR)
        self.assertAlmostEqual(summary.bin_width, 1.0, places=9)
        self.assertGreater(summary.peak_magnitude, 0.0)
        self.assertGreater(summary.total_energy, 0.0)


if __name__ == "__main__":
    unittest.main()
