"""Render an existing Stage 1 machine artifact as a readable transcript.

    python -m psych_asr.cli.render_transcript <stem>.<arm>.diarized.json

Stage 1 already writes the .txt as its last step, so this exists for re-rendering after a
change to the format without paying for another GPU job. CPU only, sub-second, no models.

OUTPUT IS PHI. It is written beside the input under data/stage1/, which is gitignored
wholesale. Do not write it anywhere else.
"""

from argparse import ArgumentParser
from pathlib import Path

from ..artifacts.naming import TRANSCRIPT_SUFFIX, stem_from_diarized
from ..artifacts.transcripts import load_transcript
from ..transcript.render import write_readable_transcript
from ..transcript.summary import format_summary
from ._common import report


def build_parser():
    parser = ArgumentParser(description="Render a Stage 1 .diarized.json as a readable transcript.")
    parser.add_argument("transcript", type=str, help="path to a <stem>.diarized.json")
    parser.add_argument("--outdir", type=str, default=None,
                        help="where to write the .txt (default: beside the input JSON)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    transcript_file = Path(args.transcript)
    # Whatever precedes ".diarized.json" -- arm included when there is one -- names the
    # output, so re-rendering one arm never overwrites another's copy.
    label = stem_from_diarized(transcript_file)
    output_dir = Path(args.outdir) if args.outdir else transcript_file.parent
    output_path = output_dir / f"{label}{TRANSCRIPT_SUFFIX}"

    summary = write_readable_transcript(load_transcript(transcript_file), output_path, label)
    report(format_summary(label, summary) + [f"Wrote {output_path}"])


if __name__ == "__main__":
    main()
