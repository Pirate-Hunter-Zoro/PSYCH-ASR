"""Stage 2: apply the QC error log to the baseline transcript and write the reference.

    python -m psych_asr.cli.apply_corrections
    python -m psych_asr.cli.apply_corrections --dry-run          # report only, writes nothing

CPU only, sub-second, no models -- it runs on the login node and needs no Slurm job.

WHAT IT READS
    data/stage1/<stem>.<arm>.diarized.json      the baseline arm's joined transcript
    data/stage1/<stem>.<arm>.transcript.txt     the SAME thing rendered, and the file the
                                                error log's Line column counts lines in
    data/stage1/*Error Log*.csv                 the annotator's spreadsheet export

WHAT IT WRITES, into data/stage2/
    <stem>.corrected.transcript.txt   the hand-corrected reference, readable (PHI)
    <stem>.corrected.turns.json       the same as a turn table, with provenance (PHI)
    <stem>.correction_report.json     counts and row numbers, no text

THE GUARD THAT MATTERS. The error log's Line numbers are line numbers in a rendered .txt,
and this job applies the corrections to the turn structure UNDERNEATH that render. So it
re-renders the JSON and refuses to continue unless the result is byte-for-byte the .txt on
disk. If the two have drifted -- a changed wrap width, a re-run against a different
diarization, a hand-edited file -- then line 348 no longer means what the annotator meant
by it, and every correction placed by line number would land on the wrong sentence while
looking perfectly successful. Better to stop and say so.
"""

from argparse import ArgumentParser
from json import dump
from pathlib import Path

from .. import config
from ..artifacts.error_log import read_error_log, sessions
from ..artifacts.naming import (
    corrected_transcript_path,
    corrected_turns_path,
    correction_report_path,
    diarized_path,
    find_sole_stem,
    transcript_path,
)
from ..artifacts.transcripts import load_transcript
from ..transcript.corrections import apply_corrections
from ..transcript.render import HEADER_LINE, render_turns, render_with_line_index
from ..transcript.summary import format_summary, summarize
from ..transcript.turns import group_into_turns
from ._common import add_arm, prepare_output_dir, report

# Any CSV in the Stage 1 directory whose name says what it is. Globbed rather than named
# outright because the export's filename is whatever the spreadsheet tool produced, spaces
# and brackets included, and that is not worth hardcoding.
ERROR_LOG_GLOB = "*rror*og*.csv"


def build_parser():
    parser = ArgumentParser(
        description="Apply the Stage 2 QC error log to the baseline transcript.")
    parser.add_argument("--stage1", type=str, default=str(config.STAGE1_DIR),
                        help="directory holding the Stage 1 artifacts (default: %(default)s)")
    parser.add_argument("--outdir", type=str, default=str(config.STAGE2_DIR),
                        help="where the corrected reference goes (default: %(default)s)")
    parser.add_argument("--stem", type=str, default=None,
                        help="session stem; inferred from the sole .aligned.json when absent")
    add_arm(parser, default=config.ARM_BASELINE,
            help_text="which arm's transcript the error log was annotated against "
                      "(default: %(default)s -- the baseline, and the only one anyone read)")
    parser.add_argument("--error-log", type=str, default=None,
                        help=f"the QC export; found by {ERROR_LOG_GLOB!r} in --stage1 when absent")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve and report every correction, write nothing")
    return parser


def find_error_log(stage1_dir):
    """IN: the Stage 1 directory   OUT: the one QC export in it.

    Raises SystemExit naming the count rather than picking one: two exports in a directory
    means somebody is mid-way through replacing one, and guessing which is current would
    silently build the reference from the stale sheet.
    """
    found = sorted(Path(stage1_dir).glob(ERROR_LOG_GLOB))
    if len(found) != 1:
        raise SystemExit(
            f"Found {len(found)} files matching {ERROR_LOG_GLOB!r} in {stage1_dir}; "
            "pass --error-log explicitly."
        )
    return found[0]


def check_render_matches(rendered, index, on_disk):
    """Refuse to place corrections by line number against a render that has drifted.

    IN:  the text this run produced from the JSON, its line index, and the .txt on disk
    OUT: a note for the report, or SystemExit

    THE BODY is what is checked, not the whole file. The summary block at the top is a
    fixed number of lines whatever it says inside them, so a difference confined to it --
    Stage 1c writes the title as "<stem> [<arm>]" while render_transcript writes
    "<stem>.<arm>" -- shifts nothing and is reported rather than refused. A difference in
    the dialogue, or a different total line count, means line 348 no longer names the
    sentence the annotator meant, and that IS fatal.

    A MISSING .txt is a warning too: the JSON is the authority and the .txt is derived
    from it, so a deleted copy costs nothing but the check itself.
    """
    if not on_disk.exists():
        return (f"{on_disk.name} is not on disk, so the line numbers could not be checked "
                "against the file they were counted in.")
    existing = on_disk.read_text()
    if existing == rendered:
        return f"{on_disk.name} reproduces byte for byte; the Line column is trustworthy."

    header = sum(1 for entry in index if entry["kind"] == HEADER_LINE)
    mine, theirs = rendered.splitlines(), existing.splitlines()
    if len(mine) == len(theirs) and mine[header:] == theirs[header:]:
        differing = [position + 1 for position in range(header)
                     if mine[position] != theirs[position]]
        return (f"{on_disk.name} differs from a fresh render only in its summary block "
                f"(line{'s' if len(differing) != 1 else ''} {differing}); the dialogue and "
                "the line count are identical, so the Line column is trustworthy.")
    raise SystemExit(
        f"{on_disk.name} does NOT match a fresh render of the JSON beside it "
        f"({len(theirs)} lines on disk vs {len(mine)} rendered, first differing dialogue "
        "line beyond the summary block). The error log's Line column counts lines in that "
        "file, so applying it now would place corrections on the wrong sentences. Re-render "
        "the transcript (python -m psych_asr.cli.render_transcript) and re-annotate, or "
        "point --arm at the transcript the log was actually annotated against."
    )


def format_report(stem, log_path, source_name, drift_note, corrections, summary, data):
    """IN: the run's inputs and the report dict   OUT: log lines. NO transcript text."""
    lines = [
        "=" * 72,
        f"STAGE 2 CORRECTION PASS — {stem}",
        "=" * 72,
        f"Error log      : {log_path.name}",
        f"Annotated      : {source_name}",
        f"Line check     : {drift_note}",
        f"Corrections    : {data['corrections_read']} rows"
        + (f"  (+{len(data['error_log_incomplete_rows'])} incomplete: rows "
           f"{data['error_log_incomplete_rows']})" if data["error_log_incomplete_rows"] else "")
        + f"  ({data['error_log_padding_rows']} padding rows below the table)",
        f"Of those       : {data['rows_changing_turn_structure']} say the diarizer got the "
        f"TURN STRUCTURE wrong (Add Turn?), "
        f"{data['rows_changing_words_only']} only the words",
        f"Turns          : {data['turns_before']} -> {data['turns_after']}",
        f"Words          : {data['words_before']} -> {data['words_after']}",
        "",
        f"{'Error':<22}{'applied':>9}{'already done':>14}{'superseded':>12}{'unplaced':>10}",
    ]
    for error, counts in sorted(data["by_error"].items()):
        lines.append(f"{error:<22}{counts['applied']:>9}"
                     f"{counts['accounted for by an earlier row']:>14}"
                     f"{counts['superseded']:>12}{counts['unplaced']:>10}")
    lines += ["", "How each row was placed:"]
    for locator, count in sorted(data["by_locator"].items(), key=lambda item: -item[1]):
        lines.append(f"  {locator:<22}{count:>5}")
    lines += ["", "Were the logged words findable in the transcript at all?",
              f"  {'Error':<22}{'annotator text':>15}{'machine text':>14}{'neither':>9}"]
    for error, counts in sorted(data["words_found_on_the_page"].items()):
        lines.append(f"  {error:<22}{counts['actual']:>15}{counts['ai']:>14}"
                     f"{counts['not on the page']:>9}")
    lines += ["", "Edits applied:"]
    for kind, count in sorted(data["edits_by_kind"].items()):
        lines.append(f"  {kind:<22}{count:>5}")
    complete = "every cluster mapped" if data["role_mapping_complete"] else \
               "INCOMPLETE -- a cluster nobody logged an error against keeps its machine label"
    lines += ["", f"Speaker role mapping ({complete}):"]
    for label, tally in sorted(data["role_vote"].items()):
        roles = "  ".join(f"{role}={count}" for role, count in sorted(tally["roles"].items()))
        evidence = ", ".join(f"{count} by {kind}"
                             for kind, count in sorted(tally["evidence"].items()))
        lines.append(f"  {label:<16}-> {data['role_mapping'].get(label, label):<12}"
                     f"{roles:<34}({evidence})")
    lines += ["", "Corrected turns by origin:      "
                  + "  ".join(f"{k}={v}" for k, v in sorted(data["turns_by_origin"].items())),
              "Corrected turns by time source: "
                  + "  ".join(f"{k}={v}" for k, v in sorted(data["turns_by_time_source"].items()))]
    if data["accounted_rows"]:
        lines.append(f"\nRows describing an error an earlier row had already corrected "
                     f"(a misattribution is logged twice): {data['accounted_rows']}")
    if data["superseded_rows"]:
        lines.append(f"\nSuperseded spreadsheet rows (already corrected by an earlier row): "
                     f"{data['superseded_rows']}")
    if data["unplaced_rows"]:
        lines.append(f"Spreadsheet rows that could NOT be placed: {data['unplaced_rows']}")
    if data["row_notes"]:
        lines.append("\nPer-row notes (by spreadsheet row):")
        for row, notes in sorted(data["row_notes"].items(), key=lambda item: int(item[0])):
            for note in notes:
                lines.append(f"  row {row:>4}: {note}")
    lines += ["", *format_summary(f"{stem} (corrected)", summary)]
    return lines


def main(argv=None):
    args = build_parser().parse_args(argv)

    stage1 = Path(args.stage1)
    stem = args.stem or find_sole_stem(stage1)
    log_path = Path(args.error_log) if args.error_log else find_error_log(stage1)

    source_json = diarized_path(stage1, stem, args.arm)
    if not source_json.exists():
        raise SystemExit(f"{source_json} does not exist; run Stage 1c for arm {args.arm!r} first.")
    transcript = load_transcript(source_json)

    # The render is done for its INDEX, not for its text: the index is what turns the
    # annotator's line numbers into turns and character offsets. The label is spelled the
    # way Stage 1c spelled it -- "<stem> [<arm>]", from cli/join_speakers.py -- so the
    # re-render reproduces the annotated file exactly rather than only nearly.
    rendered, _, index = render_with_line_index(transcript, f"{stem} [{args.arm}]")
    drift_note = check_render_matches(rendered, index,
                                      transcript_path(stage1, stem, args.arm))

    log = read_error_log(log_path)
    corrections = log.corrections
    if not corrections:
        raise SystemExit(f"{log_path.name} holds no corrections; nothing to apply.")
    logged = sessions(corrections)
    if len(logged) != 1:
        raise SystemExit(f"{log_path.name} covers {len(logged)} sessions. One sheet per "
                         "session: split it before applying.")
    if logged[0] != stem:
        report(f"WARNING: the error log's Session ID is not the stem being corrected. "
               f"Applying it anyway; check that this is the sheet you meant.")

    turns = group_into_turns(transcript.get("segments", []))
    corrected, data = apply_corrections(turns, corrections, index)

    # No segments underneath a corrected turn any more, so each turn IS its own segment.
    # That makes the header's "Segments" line equal its "Turns" line, which is the truth
    # about this artifact rather than a number carried over from the machine transcript.
    summary = summarize(corrected, corrected)
    data.update({
        "session": stem,
        "arm_annotated": args.arm,
        "error_log": log_path.name,
        "error_log_header_row": log.header_row,
        "error_log_padding_rows": log.padding_rows,
        "error_log_incomplete_rows": log.incomplete_rows,
        "line_check": drift_note,
        "summary": {
            "span": summary["span"],
            "speech_time": summary["speech_time"],
            "num_turns": summary["num_turns"],
            "per_speaker": summary["per_speaker"],
        },
    })

    if not args.dry_run:
        outdir = prepare_output_dir(args.outdir)
        text_path = corrected_transcript_path(outdir, stem)
        text_path.write_text(render_turns(corrected, f"{stem} (corrected)", summary))
        with corrected_turns_path(outdir, stem).open("w") as handle:
            dump({"session": stem, "arm_annotated": args.arm, "turns": corrected},
                 handle, indent=4, ensure_ascii=False)
        with correction_report_path(outdir, stem).open("w") as handle:
            dump(data, handle, indent=4, sort_keys=True)

    lines = format_report(stem, log_path, source_json.name, drift_note,
                          corrections, summary, data)
    if args.dry_run:
        lines.append("\n--dry-run: nothing was written.")
    else:
        lines.append(f"\nWrote {corrected_transcript_path(args.outdir, stem)}")
        lines.append(f"Wrote {corrected_turns_path(args.outdir, stem)}")
        lines.append(f"Wrote {correction_report_path(args.outdir, stem)}")
    report(lines)


if __name__ == "__main__":
    main()
