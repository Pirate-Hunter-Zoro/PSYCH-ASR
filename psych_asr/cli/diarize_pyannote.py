"""Stage 1b, baseline arm: pyannote community-1 -> RTTM.

    python -m psych_asr.cli.diarize_pyannote <audio.wav> --num-speakers 2

Audio in, RTTM out. Nothing else. All joining and rendering lives in Stage 1c so that every
arm of the bake-off shares one join implementation and one renderer -- otherwise the
comparison would partly measure the glue.

Runs in asr_env on a GPU. Also writes <stem>.<arm>.exclusive.rttm: the same diarization
with the per-frame speaker count clamped to one, which makes overlapping speech recoverable
as a SET DIFFERENCE rather than reconstructed from turn intersections.
"""

from argparse import ArgumentParser
from pathlib import Path

from .. import config
from ..artifacts.naming import exclusive_rttm_path, rttm_path
from ..artifacts.rttm_io import format_turn_summary, read_rttm, summarize_turns, write_rttm_from_annotation
from ..asr.align import load_audio
from ..diarize import pyannote_arm
from ._common import add_arm, add_num_speakers, add_output_dir, prepare_output_dir, report


def build_parser():
    parser = ArgumentParser(description="Stage 1b baseline: diarize one WAV with pyannote community-1.")
    parser.add_argument("audio", type=str)
    add_output_dir(parser)
    parser.add_argument("--model-dir", type=str, default=str(config.PYANNOTE_MODEL_DIR),
                        help="staged community-1 directory (default: %(default)s)")
    add_num_speakers(parser)
    add_arm(parser, default=config.ARM_BASELINE)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    audio_path = Path(args.audio)
    output_dir = prepare_output_dir(args.outdir)
    uri = audio_path.stem

    # The same decode Stage 1a used, so the diarizer sees exactly the waveform the words
    # were aligned against.
    decoded_audio = load_audio(audio_path)
    pipeline = pyannote_arm.load_pipeline(args.model_dir)
    output = pyannote_arm.diarize(pipeline, decoded_audio, args.num_speakers)

    written = rttm_path(output_dir, uri, args.arm)
    num_turns = write_rttm_from_annotation(output.speaker_diarization, uri, written)

    # The overlap-free view, kept alongside. Its own set difference against the main RTTM is
    # where every Stage 3a overlap/interruption feature comes from.
    exclusive = pyannote_arm.exclusive_annotation(output)
    if exclusive is not None:
        exclusive_written = exclusive_rttm_path(output_dir, uri, args.arm)
        num_exclusive = write_rttm_from_annotation(exclusive, uri, exclusive_written)
        report(f"Wrote {exclusive_written} ({num_exclusive} turns, overlap-free view)")

    report(f"Wrote {written} ({num_turns} turns)")
    report(format_turn_summary(args.arm, uri, summarize_turns(read_rttm(written))))


if __name__ == "__main__":
    main()
