"""Stage 1c: join one arm's RTTM onto the shared aligned transcript, then render.

    python -m psych_asr.cli.join_speakers <stem>.aligned.json <stem>.<arm>.rttm

IN:  <stem>.aligned.json from Stage 1a + <stem>.<arm>.rttm from Stage 1b
OUT: <stem>.<arm>.diarized.json + <stem>.<arm>.transcript.txt

CPU only -- no model loads, no GPU. Runs in asr_env for whisperx.assign_word_speakers.

The arm is derived from the RTTM's OWN FILENAME by default, which is why adding a fifth arm
needs no change here.
"""

from argparse import ArgumentParser
from pathlib import Path

from ..artifacts.naming import RTTM_SUFFIX, arm_from, diarized_path, stem_from_aligned, transcript_path
from ..artifacts.transcripts import load_transcript, save_transcript
from ..join import format_join_summary, join_arm
from ..transcript.render import write_readable_transcript
from ..transcript.summary import format_summary
from ._common import report


def build_parser():
    parser = ArgumentParser(description="Stage 1c: join a diarization RTTM onto the aligned transcript.")
    parser.add_argument("aligned", type=str, help="path to <stem>.aligned.json from Stage 1a")
    parser.add_argument("rttm", type=str, help="path to <stem>.<arm>.rttm from Stage 1b")
    parser.add_argument("--arm", type=str, default=None,
                        help="arm name for the output filenames (default: inferred from the RTTM filename)")
    parser.add_argument("--outdir", type=str, default=None,
                        help="where to write (default: beside the aligned JSON)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    aligned_file = Path(args.aligned)
    rttm_file = Path(args.rttm)

    stem = stem_from_aligned(aligned_file)
    arm = args.arm or arm_from(rttm_file, stem, RTTM_SUFFIX)

    output_dir = Path(args.outdir) if args.outdir else aligned_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    aligned = load_transcript(aligned_file)
    augmented, num_turns = join_arm(aligned, rttm_file, arm)

    output_path = save_transcript(augmented, diarized_path(output_dir, stem, arm))
    report(format_join_summary(arm, augmented, num_turns) + [f"Wrote {output_path}"])

    # Talk-time split is the cheapest diarization sanity check there is: two people in a
    # room do not split 97/3, so that number falsifies a collapsed clustering from the log
    # alone, before anyone opens the audio.
    label = f"{stem} [{arm}]"
    readable_path = transcript_path(output_dir, stem, arm)
    summary = write_readable_transcript(augmented, readable_path, label)
    report(format_summary(label, summary) + [f"Wrote {readable_path}"])


if __name__ == "__main__":
    main()
