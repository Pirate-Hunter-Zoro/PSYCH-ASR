"""Stage 2: grade every arm's transcript against the corrected reference, her way.

    python -m psych_asr.cli.grade_arms
    python -m psych_asr.cli.grade_arms --arm community-1        # one cell only
    python -m psych_asr.cli.grade_arms --details                # also write the PHI spans
    python -m psych_asr.cli.grade_arms --dry-run                # report only, writes nothing

CPU only, sub-second per arm, no models and no third-party imports -- so it runs on the
login node in whichever env happens to be active, and every cell of the model grid can be
graded the moment its transcript lands.

WHAT IT READS
    data/stage2/<stem>.corrected.turns.json     the reference: the baseline plus the
                                                annotator's 117 corrections
    data/stage1/<stem>.<arm>.diarized.json      one per arm, discovered by glob
    data/stage2/<stem>.correction_report.json   optional, and only for the calibration line

WHAT IT WRITES, into data/stage2/
    <stem>.<arm>.error_profile.json   that arm's profile: counts and rates, NO text
    <stem>.error_profiles.json        every arm in one table, NO text
    <stem>.<arm>.error_detail.json    --details only. The spans, and therefore PHI.

THE CALIBRATION LINE IS THE POINT OF PRINTING THE ARM THE LOG WAS ANNOTATED AGAINST.
The reference IS that arm plus her corrections, so grading it re-derives her sheet from
the other direction. The two tallies will not be identical -- she logged one row per thing
she decided had happened, and this logs one finding per contiguous difference -- but a
profile that is not close is a classifier disagreeing with the annotator about what counts
as an omission, and that is worth knowing before any other cell is believed.

EVERY ARM IS GRADED ON THE SAME REFERENCE AND THE SAME DEFINITIONS, which is the only
reason the cells of the grid can be compared to each other at all.
"""

from argparse import ArgumentParser
from json import dump, load
from pathlib import Path

from .. import config
from ..artifacts.naming import (
    corrected_turns_path,
    correction_report_path,
    error_detail_path,
    error_profile_path,
    error_profiles_path,
    find_arm_transcripts,
    find_sole_stem,
)
from ..artifacts.transcripts import load_transcript
from ..evaluate import grade as grading
from ..transcript.turns import group_into_turns
from ._common import add_arm, prepare_output_dir, report


def build_parser():
    parser = ArgumentParser(
        description="Grade every diarization arm against the hand-corrected reference, "
                    "in the annotator's own error categories.")
    parser.add_argument("--stage1", type=str, default=str(config.STAGE1_DIR),
                        help="directory holding the arm transcripts (default: %(default)s)")
    parser.add_argument("--stage2", type=str, default=str(config.STAGE2_DIR),
                        help="directory holding the corrected reference, and where the "
                             "profiles go (default: %(default)s)")
    parser.add_argument("--stem", type=str, default=None,
                        help="session stem; inferred from the sole .aligned.json when absent")
    add_arm(parser, default=None,
            help_text="grade only this arm; every arm found is graded when absent")
    parser.add_argument("--transcript", type=str, default=None,
                        help="grade one transcript at this exact path instead of discovering "
                             "them; requires --arm, which names the cell in the output")
    parser.add_argument("--details", action="store_true",
                        help="also write the span-by-span detail file. IT CARRIES BOTH "
                             "TRANSCRIPTS' WORDS and is PHI; the counts do not need it")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the profiles, write nothing")
    return parser


def load_reference(stage2_dir, stem):
    """IN: the Stage 2 directory + stem   OUT: the corrected reference's turn list.

    Raises SystemExit naming the missing file rather than grading against nothing: without
    the reference there is no answer key, and every number this job produces would be a
    comparison against an empty transcript that looked like a catastrophic arm.
    """
    path = corrected_turns_path(stage2_dir, stem)
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist. The reference is built by "
            "python -m psych_asr.cli.apply_corrections; run that first."
        )
    with path.open() as handle:
        return load(handle).get("turns", [])


def candidate_turns(path):
    """IN: an arm's .diarized.json   OUT: its turn list, grouped the one canonical way.

    Through group_into_turns rather than by reading the segments, because the reference is a
    turn list and comparing a turn list to a segment list would count every sentence
    boundary alignment inserted as a turn boundary the diarizer never claimed.
    """
    return group_into_turns(load_transcript(path).get("segments", []))


def annotator_tally(stage2_dir, stem):
    """IN: the Stage 2 directory + stem   OUT: the sheet's own numbers, or None.

    Read out of the correction report, which carries counts and spreadsheet row numbers and
    no text. Absent is normal -- a session graded before its log was applied has no sheet to
    compare against -- so this returns None rather than raising.
    """
    path = correction_report_path(stage2_dir, stem)
    if not path.exists():
        return None
    with path.open() as handle:
        data = load(handle)
    return {
        "arm_annotated": data.get("arm_annotated"),
        "rows": data.get("corrections_read"),
        "rows_changing_turn_structure": data.get("rows_changing_turn_structure"),
        "rows_changing_words_only": data.get("rows_changing_words_only"),
        "by_error": {error: sum(counts.values())
                     for error, counts in (data.get("by_error") or {}).items()},
    }


def format_profile(arm, profile):
    """IN: an arm name + its profile   OUT: log lines. NO transcript text."""
    words = profile["words"]
    lines = [
        "-" * 78,
        f"{arm}",
        "-" * 78,
        f"Reference      : {profile['reference_words']} words in "
        f"{profile['reference_turns']} turns",
        f"This arm       : {profile['candidate_words']} words in "
        f"{profile['candidate_turns']} turns",
        f"What was said  : {words['matched']} matched, {words['substituted']} substituted, "
        f"{words['omitted']} omitted, {words['inserted']} invented"
        f"   -> WER {profile['word_error_rate'] * 100:.1f}%",
        f"Who said it    : {profile['misattributed_words']} of {words['matched']} matched "
        f"words under the wrong person"
        f"   -> {profile['speaker_error_rate'] * 100:.1f}%"
        + (f"   ({profile['unlabeled_matched_words']} of them given no speaker at all)"
           if profile["unlabeled_matched_words"] else ""),
        f"Speaker roles  : "
        + ("  ".join(f"{label} -> {role}"
                     for label, role in sorted(profile["role_mapping"].items()))
           or "(nothing to map)")
        + ("" if profile["role_mapping_complete"]
           else "   INCOMPLETE -- a cluster never agreed with the reference anywhere"),
        "",
        f"{'Error':<22}{'findings':>10}{'ref words':>11}",
    ]
    for error in grading.ERROR_LABELS:
        counts = profile["by_error"][error]
        lines.append(f"{error:<22}{counts['findings']:>10}{counts['reference_words']:>11}")
    lines += ["", "What it did to the turn structure (the annotator's Add Turn? column):"]
    for structure in (grading.MISSED_TURN, grading.WELDED_TURN, grading.INVENTED_TURN):
        lines.append(f"  {structure:<40}{profile['by_structure'][structure]:>6}")
    lines.append(f"  {'TURNS WRONG — ticked':<40}"
                 f"{profile['findings_changing_turn_structure']:>6}")
    lines.append(f"  {grading.WORDS_ONLY:<40}"
                 f"{profile['findings_changing_words_only']:>6}")
    return lines


def format_calibration(tally, profile):
    """IN: the sheet's own numbers + the profile of the arm it was annotated against
    OUT: the side-by-side lines. Counts only.
    """
    lines = [
        "=" * 78,
        f"CALIBRATION — this arm IS what the annotator read, so her sheet and this profile "
        f"describe the same errors",
        "=" * 78,
        f"{'':<22}{'her rows':>10}{'findings':>10}",
        f"{'Total':<22}{tally['rows']:>10}{profile['findings']:>10}",
        f"{'Turns wrong':<22}{tally['rows_changing_turn_structure']:>10}"
        f"{profile['findings_changing_turn_structure']:>10}",
        f"{'Words only':<22}{tally['rows_changing_words_only']:>10}"
        f"{profile['findings_changing_words_only']:>10}",
        "",
    ]
    for error in grading.ERROR_LABELS:
        lines.append(f"{error:<22}{tally['by_error'].get(error, 0):>10}"
                     f"{profile['by_error'][error]['findings']:>10}")
    lines.append("")
    lines.append("Her rows and these findings are the same KIND of count, not the same "
                 "count: she logged one row per thing she decided had happened, and this "
                 "logs one finding per contiguous difference.")
    return lines


def format_grid(profiles):
    """IN: dict arm -> profile   OUT: the one table the whole job exists to print.

    One row per arm, so the cells of the grid can be read against each other. WER moves with
    the typist, the speaker column moves with the name-tagger and the stopwatch, and putting
    them side by side is what makes that visible without reading a transcript.
    """
    lines = [
        "=" * 78,
        "THE GRID SO FAR — one row per cell, all graded on the same reference",
        "=" * 78,
        f"{'cell':<26}{'WER':>8}{'speaker':>9}{'turns wrong':>13}{'words only':>12}",
    ]
    for arm, profile in sorted(profiles.items()):
        lines.append(
            f"{arm:<26}{profile['word_error_rate'] * 100:>7.1f}%"
            f"{profile['speaker_error_rate'] * 100:>8.1f}%"
            f"{profile['findings_changing_turn_structure']:>13}"
            f"{profile['findings_changing_words_only']:>12}"
        )
    return lines


def main(argv=None):
    args = build_parser().parse_args(argv)

    stage1, stage2 = Path(args.stage1), Path(args.stage2)
    stem = args.stem or find_sole_stem(stage1)
    reference_turns = load_reference(stage2, stem)
    if not reference_turns:
        raise SystemExit(f"The corrected reference for '{stem}' holds no turns; "
                         "re-run apply_corrections.")

    if args.transcript:
        if not args.arm:
            raise SystemExit("--transcript names one file and says nothing about which cell "
                             "it is; pass --arm as well.")
        found = [(args.arm, Path(args.transcript))]
    else:
        found = find_arm_transcripts(stage1, stem)
        if args.arm:
            found = [(arm, path) for arm, path in found if arm == args.arm]
        if not found:
            raise SystemExit(
                f"No arm transcripts for stem '{stem}' in {stage1}"
                + (f" matching --arm {args.arm!r}" if args.arm else "")
                + ". Run Stage 1c first."
            )

    tally = annotator_tally(stage2, stem)
    reference_stream = grading.word_stream(reference_turns)

    report(["=" * 78,
            f"STAGE 2 GRADING PASS — {stem}",
            "=" * 78,
            f"Reference      : the corrected transcript, "
            f"{len(reference_stream)} words in {len(reference_turns)} turns",
            f"Cells to grade : {len(found)}  "
            f"({', '.join(arm for arm, _ in sorted(found))})",
            "No timestamp is read: the reference's word times are interpolated and are not "
            "scoreable. This grades WORDS and SPEAKERS only.",
            ""])

    profiles, details = {}, {}
    for arm, path in sorted(found):
        turns = candidate_turns(path)
        findings, profile = grading.grade(reference_turns, turns)
        profile.update({"session": stem, "arm": arm, "source": path.name})
        profiles[arm] = profile
        if args.details:
            details[arm] = grading.finding_records(
                findings, reference_stream, grading.word_stream(turns))
        report(format_profile(arm, profile))

    if tally and tally["arm_annotated"] in profiles:
        report([""] + format_calibration(tally, profiles[tally["arm_annotated"]]))

    if len(profiles) > 1:
        report([""] + format_grid(profiles))

    if args.dry_run:
        report("\n--dry-run: nothing was written.")
        return

    outdir = prepare_output_dir(stage2)
    written = []
    for arm, profile in sorted(profiles.items()):
        path = error_profile_path(outdir, stem, arm)
        with path.open("w") as handle:
            dump(profile, handle, indent=4, sort_keys=True)
        written.append(path)
        if arm in details:
            detail_path = error_detail_path(outdir, stem, arm)
            with detail_path.open("w") as handle:
                dump({"session": stem, "arm": arm, "findings": details[arm]},
                     handle, indent=4, ensure_ascii=False)
            written.append(detail_path)
    combined = error_profiles_path(outdir, stem)
    with combined.open("w") as handle:
        dump({"session": stem,
              "reference_words": len(reference_stream),
              "reference_turns": len(reference_turns),
              "annotator_tally": tally,
              "cells": profiles}, handle, indent=4, sort_keys=True)
    written.append(combined)
    report([""] + [f"Wrote {path}" for path in written])


if __name__ == "__main__":
    main()
