"""The regression gate: does 1a -> 1b(community-1) -> 1c reproduce the single-job fixture?

    python -m psych_asr.cli.check_split_regression <fixture>.diarized.json <candidate>.community-1.diarized.json

CPU only. Runs in asr_env; loads no models.

Exits 1 when the STRUCTURE differs -- that means the split changed behavior and is a bug,
not a finding, and no challenger arm means anything until it is fixed. A field-level
difference with matching structure is reported and does NOT fail: Whisper decodes in
float16 on a GPU and is not bit-reproducible across runs.
"""

from argparse import ArgumentParser
from pathlib import Path

from ..artifacts.transcripts import load_transcript
from ..evaluate.regression import compare_fields, compare_structures, structure_of
from ._common import report


def build_parser():
    parser = ArgumentParser(description="Verify the 1a/1b/1c split reproduces the single-job fixture.")
    parser.add_argument("reference", type=str, help="the fixture <stem>.diarized.json from job 2032471")
    parser.add_argument("candidate", type=str, help="the split's <stem>.community-1.diarized.json")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    reference = load_transcript(args.reference)
    candidate = load_transcript(args.candidate)

    rule = "=" * 72
    report([rule, "1a/1b/1c SPLIT REGRESSION CHECK", rule,
            f"reference: {Path(args.reference).name}",
            f"candidate: {Path(args.candidate).name}", "",
            f"{'':<20}{'reference':>26}{'candidate':>26}"])

    structure_ok, rows = compare_structures(structure_of(reference), structure_of(candidate))
    for key, left, right, matched in rows:
        report(f"{key:<20}{str(left):>26}{str(right):>26}  {'ok' if matched else 'MISMATCH'}")

    report(rule)
    if not structure_ok:
        report(["STRUCTURE DIFFERS. The split changed behavior -- this is a bug, not a finding.",
                "Do not add a challenger arm until it is fixed; a difference between arms",
                "could not be attributed to the model afterwards."])
        raise SystemExit(1)

    report("Structure matches the fixture.")
    differences = compare_fields(reference, candidate)
    if not differences:
        report("Exact field comparison also matches: the split is byte-identical to the fixture.")
    else:
        report([f"Structure matches but {len(differences)}+ individual fields differ. Read these before",
                "treating it as a failure -- Whisper decodes in float16 on the GPU and is not",
                "bit-reproducible across runs, so scattered decode noise is expected. A systematic",
                "shift (every timestamp offset, every speaker swapped) is not."]
               + [f"  {difference}" for difference in differences])
    report(rule)


if __name__ == "__main__":
    main()
