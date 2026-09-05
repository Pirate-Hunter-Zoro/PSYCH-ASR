"""Word-level diff of every diarization arm's transcript.

    python -m psych_asr.cli.compare_arms

CPU only, runs in asr_env, loads no models. Because Stage 1a ran once, every arm sits on
the identical word sequence with identical word timings, so comparing four 50-minute
transcripts is an exact diff over ONE COLUMN rather than a reading task.

Agreement is NOT accuracy: four arms can agree and all be wrong. What this produces is the
map of WHERE they disagree, which is what makes the human listening pass affordable. Only
the hand-corrected reference RTTM from Stage 2 scores the arms.

BOTH THE STDOUT AND THE JSON CARRY VERBATIM TRANSCRIPT TEXT and are therefore PHI. The JSON
is written under data/stage1/, which is gitignored wholesale, and the assistant's read
guard refuses both this artifact and the running of this entry point.
"""

import json
from argparse import ArgumentParser
from pathlib import Path

from .. import config
from ..artifacts.naming import comparison_path, find_arm_transcripts, find_sole_stem
from ..artifacts.transcripts import load_transcript, word_label_columns, word_start_time
from ..evaluate import compare
from ..transcript.summary import format_timestamp
from ._common import report


def build_parser():
    parser = ArgumentParser(description="Word-level diff of every diarization arm's transcript.")
    parser.add_argument("--stage1-dir", type=str, default=str(config.STAGE1_DIR))
    parser.add_argument("--stem", type=str, default=None,
                        help="session stem (default: inferred from the single .aligned.json present)")
    parser.add_argument("--baseline", type=str, default=compare.DEFAULT_BASELINE,
                        help="arm whose speaker names every other arm is mapped onto")
    parser.add_argument("--max-runs-listed", type=int, default=25,
                        help="how many disagreement regions to print in full; all of them are "
                             "written to the JSON")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    stage1_dir = Path(args.stage1_dir)
    stem = args.stem or find_sole_stem(stage1_dir)

    found = find_arm_transcripts(stage1_dir, stem)
    if len(found) < 2:
        raise SystemExit(
            f"Need at least two arms to compare; found {sorted(arm for arm, _ in found)} for stem '{stem}'."
        )

    transcripts, words_by_arm, raw_labels = {}, {}, {}
    for arm, path in found:
        transcripts[arm] = load_transcript(path)
        words_by_arm[arm], raw_labels[arm] = word_label_columns(transcripts[arm])

    ordered_arms, reference_arm = compare.order_arms(list(transcripts), args.baseline)
    num_words = compare.require_identical_words(words_by_arm, reference_arm)
    words = words_by_arm[reference_arm]

    rule = "=" * 78
    report([rule, f"DIARIZATION ARM COMPARISON — {stem}", rule,
            f"Words in common : {num_words}", f"Baseline arm    : {reference_arm}", ""])

    labels_by_arm, mappings = compare.canonicalize_labels(raw_labels, ordered_arms, reference_arm)
    for arm in ordered_arms[1:]:
        rendered = ", ".join(f"{source} -> {target}" for source, target in sorted(mappings[arm].items()))
        report(f"relabel {arm:<24}: {rendered}")

    report(["", f"{'arm':<24}{'labels':>8}{'unlabeled':>11}   talk-time split"])
    per_arm_summary = {}
    for arm in ordered_arms:
        shape = compare.per_arm_shape(labels_by_arm[arm])
        per_arm_summary[arm] = shape
        split = "  ".join(
            f"{label}:{share * 100:.0f}%"
            for label, share in sorted(shape["word_share"].items(), key=lambda kv: -kv[1])
        ) or "(none)"
        report(f"{arm:<24}{shape['distinct_labels']:>8}{shape['unlabeled_words']:>11}   {split}")

    pairwise = compare.pairwise_agreement(labels_by_arm, ordered_arms)
    report(["", "pairwise word-level agreement (over words both arms labeled)",
            f"{'':<24}" + "".join(f"{arm[:11]:>13}" for arm in ordered_arms)])
    for left in ordered_arms:
        report(f"{left:<24}" + "".join(f"{pairwise[left][right]:>12.1f}%" for right in ordered_arms))

    runs = compare.disagreement_runs(labels_by_arm, ordered_arms, num_words)
    interesting = [run for run in runs if run[1] - run[0] >= compare.MIN_INTERESTING_RUN]
    disputed_words = sum(end - start for start, end in runs)

    report([
        "",
        f"Disagreement regions      : {len(runs)}  ({disputed_words} words, "
        f"{disputed_words / num_words * 100:.1f}% of the transcript)",
        f"Of those, >= {compare.MIN_INTERESTING_RUN} words long : {len(interesting)}  "
        f"(shorter runs are boundary jitter at a speaker change)",
        rule,
    ])

    baseline_transcript = transcripts[reference_arm]
    records = compare.build_region_records(
        interesting, words, labels_by_arm, ordered_arms,
        start_time_of=lambda index: word_start_time(baseline_transcript, index),
    )

    longest_first = sorted(records, key=lambda r: -(r["word_range"][1] - r["word_range"][0]))
    for record in longest_first[: args.max_runs_listed]:
        start, end = record["word_range"]
        stamp = format_timestamp(record["start_time"]) if record["start_time"] is not None else "--:--"
        report(f"[{stamp}] words {start}-{end} ({end - start})")
        for arm, counts in compare.summarize_region(record, ordered_arms).items():
            said = "  ".join(f"{label}x{count}" for label, count in
                             sorted(counts.items(), key=lambda kv: -kv[1]))
            report(f"    {arm:<24}{said}")
        report([f"    \"{record['disputed_text']}\"", ""])

    output_path = comparison_path(stage1_dir, stem)
    with open(output_path, "w") as f:
        json.dump({
            "stem": stem,
            "baseline_arm": reference_arm,
            "arms": ordered_arms,
            "num_words": num_words,
            "label_mappings": mappings,
            "per_arm": per_arm_summary,
            "pairwise_agreement": pairwise,
            "num_disagreement_regions": len(runs),
            "disputed_words": disputed_words,
            "regions": records,
        }, f, indent=2, ensure_ascii=False)
    report(f"Wrote {output_path}  ({len(records)} regions with context, for downstream triage)")


if __name__ == "__main__":
    main()
