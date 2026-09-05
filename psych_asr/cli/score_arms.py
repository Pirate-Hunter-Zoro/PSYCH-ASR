"""Score every diarization arm against a hand-corrected reference RTTM.

    python -m psych_asr.cli.score_arms <reference.rttm> [--uem <file>]

Runs in diar_eval_env: pyannote.metrics, deliberately no torch. ONE scorer, ONE collar
setting, EVERY arm -- or the comparison measures the scorer instead of the models.

IT CANNOT PRODUCE A REAL NUMBER UNTIL THE REFERENCE EXISTS. Scoring the baseline against
itself returns 0.00% DER at both collars, which checks that the implementation is not lying
and nothing else.

Writes <stem>.arm_scores.json: metrics only, no transcript text. That is why this one file
inside data/ is deliberately left readable by the assistant's guard.
"""

import json
from argparse import ArgumentParser
from pathlib import Path

from pyannote.database.util import load_rttm

from .. import config
from ..artifacts.naming import diarized_path, find_arm_rttms, scores_path
from ..evaluate import score
from ._common import report

NO_UEM_WARNING = (
    "WARNING: no --uem given. DER is being computed over the union of reference and\n"
    "         hypothesis extents, so an arm is scored on audio the reference never\n"
    "         annotated. Fine for a smoke test, NOT for a reported number.\n"
)


def build_parser():
    parser = ArgumentParser(description="Score every diarization arm against a reference RTTM.")
    parser.add_argument("reference", type=str, help="hand-corrected reference RTTM from Stage 2")
    parser.add_argument("--stage1-dir", type=str, default=str(config.STAGE1_DIR))
    parser.add_argument("--stem", type=str, default=None,
                        help="session stem (default: inferred from the reference filename)")
    parser.add_argument("--backchannel-source", type=str, default=config.ARM_BASELINE,
                        help="which arm's joined transcript defines the backchannel spans. ONE arm "
                             "defines them for ALL arms, so every arm is scored on the same set; "
                             "letting each arm nominate its own would change the denominator per arm "
                             "and make the percentages incomparable")
    parser.add_argument("--uem", type=str, default=None,
                        help="evaluation-region UEM. Without it pyannote.metrics approximates the "
                             "region as the union of reference and hypothesis extents, which scores "
                             "arms over stretches the reference never annotated")
    return parser


def load_uem(path, stem):
    """IN: a UEM path (or None) + the stem   OUT: the UEM timeline, or None with a warning."""
    if not path:
        report(NO_UEM_WARNING)
        return None
    from pyannote.database.util import load_uem as _load_uem

    return _load_uem(path)[stem]


def main(argv=None):
    args = build_parser().parse_args(argv)

    reference_path = Path(args.reference)
    stage1_dir = Path(args.stage1_dir)
    stem = args.stem or reference_path.name.split(".")[0]

    references = load_rttm(str(reference_path))
    if stem not in references:
        raise SystemExit(f"{reference_path} holds no annotation for uri '{stem}' (found {sorted(references)}).")
    reference = references[stem]

    uem = load_uem(args.uem, stem)

    # ONE backchannel span set for every arm. The spans come from a transcript rather than
    # the reference RTTM because identifying a backchannel needs the WORDS, and an RTTM
    # carries none. Which arm supplies them barely matters (the arms agree on >99% of
    # words); that it is the SAME arm for all of them is what makes the percentages
    # comparable.
    backchannel_path = diarized_path(stage1_dir, stem, args.backchannel_source)
    if backchannel_path.exists():
        spans = score.backchannel_spans(backchannel_path)
        report(f"backchannel spans: {len(spans)}, taken from '{args.backchannel_source}'")
    else:
        spans = []
        report(f"backchannel spans: none -- {backchannel_path.name} not found, so that column is n/a")

    reference_shares = score.talk_time_shares(reference)
    report(["reference talk-time: " + "  ".join(f"{k}:{v * 100:.1f}%" for k, v in sorted(reference_shares.items())), ""])

    rule = "=" * 96
    report([rule,
            f"{'arm':<24}{'collar':>7}{'DER':>9}{'miss':>9}{'FA':>9}{'conf':>9}{'ratio err':>11}{'backchannel':>13}",
            rule])

    results = {}
    for arm, rttm_file in find_arm_rttms(stage1_dir, stem):
        hypotheses = load_rttm(str(rttm_file))
        if stem not in hypotheses:
            report(f"{arm:<24}  (no annotation for '{stem}' -- skipped)")
            continue

        scored = score.score_arm(reference, hypotheses[stem], reference_shares, spans, uem=uem)
        results[arm] = scored
        report(_format_arm_rows(arm, scored))

    report([rule,
            "DER is overlap-INCLUSIVE at both collars. 'ratio err' is the largest absolute error in any",
            "reference speaker's share of speech. 'backchannel' is whole listener-token turns landing on",
            "the right person. Agreement between arms is not accuracy -- only this reference is."])

    output_path = scores_path(stage1_dir, stem)
    with open(output_path, "w") as f:
        json.dump({"stem": stem, "reference": str(reference_path), "uem": args.uem,
                   "arms": _serializable(results)}, f, indent=2)
    report(f"\nWrote {output_path}")


def _backchannel_text(scored):
    """IN: one arm's score dict   OUT: "correct/total pct%", or "n/a" when no spans exist."""
    if not scored["backchannel"]:
        return "n/a"
    correct, total = scored["backchannel"]
    return f"{correct}/{total} {correct / total * 100:.0f}%"


def _format_arm_rows(arm, scored):
    """IN: arm name + its score dict   OUT: one table row per collar.

    The per-arm columns that do not depend on the collar print once, on the first row, so
    the eye reads a two-line block per arm rather than a repeated number.
    """
    rows = []
    for index, (collar, detail) in enumerate(scored["der"].items()):
        first = index == 0
        extra = (f"{scored['talk_time_ratio_error'] * 100:>10.1f}%{_backchannel_text(scored):>13}"
                 if first else "")
        rows.append(
            f"{arm if first else '':<24}{float(collar):>7}"
            f"{detail['der'] * 100:>8.2f}%"
            f"{detail['missed_detection']:>8.1f}s"
            f"{detail['false_alarm']:>8.1f}s"
            f"{detail['confusion']:>8.1f}s{extra}"
        )
    return rows


def _serializable(results):
    """IN: the per-arm score dicts   OUT: the same with the backchannel tuple named.

    A bare [correct, total] pair in the JSON is a shape whoever reads it has to guess at;
    naming the two fields costs nothing and the file is the reportable artifact.
    """
    out = {}
    for arm, scored in results.items():
        record = dict(scored)
        if scored["backchannel"]:
            correct, total = scored["backchannel"]
            record["backchannel"] = {"correct": correct, "total": total,
                                     "accuracy": correct / total if total else None}
        out[arm] = record
    return out


if __name__ == "__main__":
    main()
