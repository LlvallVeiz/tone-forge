"""Command-line interface for tone-forge.

Subcommands:
    tone    generate a single tone (sine/square/sawtooth/triangle) to a .wav
    melody  render a note sequence like "C4:1 E4:1 G4:2" to a .wav
    analyze print the peak frequency (and nearest note) of a .wav file

Run as a module: ``python -m toneforge.cli <subcommand> ...``
"""

from __future__ import annotations

import argparse
import sys

from . import analyze, melody, notes, wav_io, waveforms


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-rate", type=int, default=44100,
                        help="samples per second (default 44100)")
    parser.add_argument("--amplitude", type=float, default=0.8,
                        help="peak amplitude in (0, 1] (default 0.8)")


def _add_adsr(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--attack", type=float, default=0.01,
                        help="ADSR attack in seconds (default 0.01)")
    parser.add_argument("--decay", type=float, default=0.1,
                        help="ADSR decay in seconds (default 0.1)")
    parser.add_argument("--sustain", type=float, default=0.7,
                        help="ADSR sustain level 0..1 (default 0.7)")
    parser.add_argument("--release", type=float, default=0.1,
                        help="ADSR release in seconds (default 0.1)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tone-forge",
        description="Generate and analyse tones, melodies and WAV files.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_tone = sub.add_parser("tone", help="generate a single tone")
    p_tone.add_argument("frequency", type=float, help="frequency in Hz")
    p_tone.add_argument("duration", type=float, help="duration in seconds")
    p_tone.add_argument("output", help="output .wav file")
    p_tone.add_argument("--waveform", choices=sorted(waveforms.WAVEFORMS),
                        default="sine", help="waveform (default sine)")
    _add_common(p_tone)
    _add_adsr(p_tone)

    p_melody = sub.add_parser("melody", help="render a note sequence")
    p_melody.add_argument("notes",
                          help='e.g. "C4:1 E4:1 G4:2" (NOTE:BEATS, '
                               'chords with +, e.g. "C4+E4+G4:2")')
    p_melody.add_argument("output", help="output .wav file")
    p_melody.add_argument("--bpm", type=float, default=120.0,
                          help="tempo in beats per minute (default 120)")
    p_melody.add_argument("--waveform", choices=sorted(waveforms.WAVEFORMS),
                          default="sine", help="waveform (default sine)")
    _add_common(p_melody)
    _add_adsr(p_melody)

    p_analyze = sub.add_parser("analyze", help="analyse a .wav file")
    p_analyze.add_argument("input", help="input .wav file")
    return parser


def _run_tone(args: argparse.Namespace) -> int:
    from . import envelope
    generator = waveforms.WAVEFORMS[args.waveform]
    samples = generator(args.frequency, args.duration,
                        args.sample_rate, args.amplitude)
    samples = envelope.apply_adsr(samples, args.sample_rate,
                                  attack=args.attack, decay=args.decay,
                                  sustain_level=args.sustain,
                                  release=args.release)
    wav_io.write_wav(args.output, samples, args.sample_rate)
    print(f"wrote {args.output}: {args.waveform} {args.frequency} Hz, "
          f"{args.duration} s, {args.sample_rate} Hz, "
          f"{len(samples)} samples")
    return 0


def _run_melody(args: argparse.Namespace) -> int:
    events = melody.parse_melody_string(args.notes)
    samples = melody.render_melody(
        events, bpm=args.bpm, sample_rate=args.sample_rate,
        waveform=args.waveform, amplitude=args.amplitude,
        attack=args.attack, decay=args.decay,
        sustain_level=args.sustain, release=args.release)
    wav_io.write_wav(args.output, samples, args.sample_rate)
    seconds = len(samples) / args.sample_rate
    print(f"wrote {args.output}: {len(events)} notes at {args.bpm} BPM, "
          f"{seconds:.2f} s, {args.sample_rate} Hz, {len(samples)} samples")
    return 0


def _run_analyze(args: argparse.Namespace) -> int:
    samples, sample_rate = wav_io.read_wav(args.input)
    summary = analyze.magnitude_spectrum(samples, sample_rate)
    note, cents = notes.frequency_to_note(summary.peak_frequency)
    print(f"file: {args.input}")
    print(f"sample rate: {sample_rate} Hz, samples: {len(samples)}, "
          f"duration: {len(samples) / sample_rate:.2f} s")
    print(f"peak frequency: {summary.peak_frequency:.2f} Hz "
          f"(FFT bin width {summary.bin_width:.2f} Hz)")
    print(f"nearest note: {note} ({cents:+.1f} cents)")
    print(f"peak magnitude: {summary.peak_magnitude:.1f}, "
          f"total energy: {summary.total_energy:.1f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "tone":
        return _run_tone(args)
    if args.command == "melody":
        return _run_melody(args)
    if args.command == "analyze":
        return _run_analyze(args)
    return 2  # unreachable: subparsers are required


if __name__ == "__main__":
    sys.exit(main())
