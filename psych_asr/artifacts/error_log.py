"""Reading the Stage 2 QC error log: the hand-annotated list of what the ASR got wrong.

STDLIB ONLY.

A human listened to the session against the baseline arm's readable transcript and logged
one spreadsheet row per error. The export is a CSV whose real header is NOT the first
physical row (the sheet has a blank spacer row above it and four unnamed columns to the
right of the last real one), so the header is FOUND rather than assumed.

One row is one correction:

    Session ID      which session the row belongs to; the padding rows below the data
                    leave it blank, and that emptiness is what ends the table
    Timestamp       h:mm:ss where the annotator heard it -- the only independent time
                    anchor a correction has, and what dates an inserted turn
    Line            1-based line number IN THE BASELINE ARM'S .transcript.txt
    Speaker         the TRUE speaker role, "Therapist" or "Participant"
    Error           Omission | Insertion | Substitution | Speaker Attribution |
                    Proper Noun | Punctuation
    AI Transcript   what the machine wrote there, or a "None" sentinel when it wrote
                    nothing at all
    Actual Speech   what was actually said, or a "None" sentinel when nothing was

BOTH TEXT COLUMNS CAN CARRY A SPEAKER ROLE AS A TRAILING WORD, and missing that costs the
whole attribution analysis. On a speaker-attribution row the annotator writes the
utterance followed by who said it -- the machine's answer in the AI column, the true
answer in the Actual column -- so the two cells differ in that last word and agree on
every word before it. Matched raw, all 32 of the pilot session's attribution rows fail to
appear anywhere in the transcript, because no line of dialogue ends in the word
"Therapist"; the longest run of each snippet that does appear is every token but the last.
This reader splits that trailing role off into `ai_role` and `actual_role` and leaves the
utterance in `ai_text` and `actual_text`.

The `ai_role` is worth more than a tidier match. It is THE DIARIZER'S OWN ATTRIBUTION at
that point, written down by someone who was looking at it, which makes it the direct
evidence for which anonymous cluster is the therapist -- no lexical guessing and no
inference from where corrections landed.
    Meaning Changed / Severity      the annotator's judgement, carried through untouched
    Add Turn?       TRUE when fixing this row creates a turn boundary that the diarizer
                    never produced -- the single most load-bearing column in the sheet,
                    because it is what separates "the words were wrong" from "the turn
                    structure was wrong"
    Subract Turn?   the sheet's own spelling. Always FALSE in the pilot session; read
                    anyway, under both spellings, so a later sheet that uses it works
    Notes           free text

WHERE THE TABLE ENDS, AND WHY THAT IS NOT "the first blank row". The export carries 116
padding rows of bare commas below the data, whose only non-empty cells are the two
boolean flags a spreadsheet writes as FALSE even on a row nobody touched. Ending the read
at the first row with no Session ID looked right and was wrong: the pilot sheet has ONE
interior row -- physical row 61 -- where the annotator filled in a severity judgement and
nothing else, no Session ID and no Error. Stopping there silently read 59 of 117 rows and
reported a clean, complete-looking run on half the log. The table therefore ends where
every MEANINGFUL cell is blank (the flags do not count), a row with content but no Error
is reported as incomplete rather than skipped in silence, and a blank Session ID is filled
down from the row above it the way a spreadsheet reader would read it.

WHY THIS IS NOT `pandas.read_csv`. The AI Transcript column says the literal word "None"
on the 50 rows where the machine transcribed nothing, and "None" is in pandas' default
NA list. Reading the sheet with pandas therefore turns "the annotator explicitly recorded
that nothing was transcribed" and "the annotator left the cell blank" into the same NaN,
and there is no way afterwards to tell a pure omission from an unfilled row. This reader
keeps the raw strings and decides what counts as blank in one visible place, which is
also what lets it run in every env -- the correction pass is CPU-only and must not need
the ASR env's pin set.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

# The header row is the one that names this column. Found, not assumed: the export has a
# blank spacer row above it today, and a sheet re-exported after an edit may have two.
HEADER_KEY = "sessionid"

# Cell values that mean "nothing here". The first is a genuinely empty cell; the rest are
# what an annotator types into a spreadsheet to say the same thing on purpose. Compared
# case-folded, after stripping.
BLANK_CELLS = {"", "none", "n/a", "n\\a", "na", "null", "nan", "-", "--", "x"}

TRUE_CELLS = {"true", "yes", "y", "1", "t"}
FALSE_CELLS = {"false", "no", "n", "0", "f"}

# The columns whose emptiness decides that a row is spreadsheet padding rather than an
# annotation. The two boolean flags are excluded ON PURPOSE: the export writes FALSE into
# them on all 116 trailing rows, so a row is not "non-empty" for having them.
MEANINGFUL_FIELDS = (
    "session", "timestamp", "line", "speaker", "error",
    "ai_text", "actual_text", "meaning_changed", "severity", "notes",
)

# Canonical field name -> the header spellings seen in the wild. Matched case-folded with
# punctuation and spacing removed, so "AI Transcript:" and "ai_transcript" both land.
FIELD_ALIASES = {
    "session":         ("sessionid", "session"),
    "timestamp":       ("timestamp", "time"),
    "line":            ("line", "lineno", "linenumber"),
    "speaker":         ("speaker", "truespeaker", "actualspeaker"),
    "error":           ("error", "errortype"),
    "ai_text":         ("aitranscript", "ai", "asr", "asrtranscript"),
    "actual_text":     ("actualspeech", "actual", "truth", "groundtruth"),
    "meaning_changed": ("meaningchanged", "meaning"),
    "severity":        ("severity",),
    "add_turn":        ("addturn", "addsturn"),
    "subtract_turn":   ("subractturn", "subtractturn"),   # the sheet's typo, and the fix
    "notes":           ("notes", "note", "comment", "comments"),
}

# Error labels, normalized so a stray case or plural does not become a seventh category.
OMISSION = "Omission"
INSERTION = "Insertion"
SUBSTITUTION = "Substitution"
SPEAKER_ATTRIBUTION = "Speaker Attribution"
PROPER_NOUN = "Proper Noun"
PUNCTUATION = "Punctuation"

ERROR_ALIASES = {
    "omission": OMISSION,
    "insertion": INSERTION,
    "substitution": SUBSTITUTION,
    "speakerattribution": SPEAKER_ATTRIBUTION,
    "propernoun": PROPER_NOUN,
    "punctuation": PUNCTUATION,
    "puncutation": PUNCTUATION,       # the same typo the sheet makes in its header
}

THERAPIST = "THERAPIST"
PARTICIPANT = "PARTICIPANT"
ROLE_ALIASES = {
    "therapist": THERAPIST, "clinician": THERAPIST, "t": THERAPIST,
    "participant": PARTICIPANT, "patient": PARTICIPANT, "client": PARTICIPANT,
    "p": PARTICIPANT,
}


def split_role_suffix(text):
    """IN: a text cell   OUT: (the utterance, the role named at the end, or None).

    Only a TRAILING token counts, and only one: the sheet's convention is "<utterance>
    <Role>". A cell that is nothing but a role name keeps the role and returns no
    utterance, which is what a row saying only who spoke should mean.
    """
    if text is None:
        return None, None
    parts = str(text).split()
    if not parts:
        return None, None
    role = ROLE_ALIASES.get(_squash(parts[-1]))
    if role is None:
        return text, None
    remainder = " ".join(parts[:-1]).strip(" -–—:,;")
    return (remainder or None), role


@dataclass
class Correction:
    """One logged error, parsed. Every text field is None rather than "" when blank.

    `row` is the 1-based PHYSICAL row in the CSV, so a correction that cannot be applied
    is reported by the same number the annotator sees in their spreadsheet. It is the only
    identifier a row has, and it is what makes the report actionable without quoting the
    text back.
    """
    row: int
    session: str
    timestamp: float | None
    line: int | None
    speaker: str | None
    error: str
    ai_text: str | None
    actual_text: str | None
    ai_role: str | None
    actual_role: str | None
    meaning_changed: str | None
    severity: str | None
    add_turn: bool
    subtract_turn: bool
    notes: str | None

    @property
    def is_pure_omission(self):
        """The machine wrote nothing here: there is no snippet to find, only a place."""
        return self.ai_text is None

    @property
    def is_pure_deletion(self):
        """The machine wrote something that was never said: the span comes out."""
        return self.actual_text is None


def _squash(name):
    """Header cell -> comparison key: case-folded, all non-alphanumerics removed."""
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def _blank(cell):
    return str(cell).strip().lower() in BLANK_CELLS


def _text(cell):
    """IN: a raw cell   OUT: the trimmed text, or None if the cell means "nothing".

    Surrounding quote characters are stripped because a spreadsheet round trip sometimes
    leaves a pair of them inside the value, and a literal quote at the edge of a snippet
    stops it matching the transcript.
    """
    if _blank(cell):
        return None
    return str(cell).strip().strip('"').strip("'").strip() or None


def _flag(cell):
    """IN: a raw cell   OUT: True/False. An unrecognized value is False, not a crash:
    an empty Add Turn? cell is the sheet's way of saying no."""
    return str(cell).strip().lower() in TRUE_CELLS


def parse_timestamp(cell):
    """IN: "h:mm:ss", "mm:ss", or "" cell   OUT: seconds as a float, or None.

    Accepts either length because the sheet writes 0:00:19 and the transcript writes
    00:19, and the two have to be comparable.
    """
    if _blank(cell):
        return None
    parts = str(cell).strip().split(":")
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return None
    while len(numbers) < 3:
        numbers.insert(0, 0.0)
    hours, minutes, seconds = numbers[-3:]
    return hours * 3600.0 + minutes * 60.0 + seconds


def parse_line(cell):
    """IN: a Line cell, which a spreadsheet export writes as "48" or "48.0"   OUT: int|None."""
    if _blank(cell):
        return None
    try:
        return int(float(str(cell).strip()))
    except ValueError:
        return None


def _find_header(rows):
    """IN: every physical row   OUT: (index of the header row, canonical field -> column).

    Raises SystemExit naming what it looked for when no row names the session column,
    because a silently mis-detected header produces a table of empty corrections that
    looks like a clean session rather than a failed read.
    """
    for index, row in enumerate(rows):
        if any(_squash(cell) == HEADER_KEY for cell in row):
            columns = {}
            for position, cell in enumerate(row):
                key = _squash(cell)
                for field, aliases in FIELD_ALIASES.items():
                    if key in aliases and field not in columns:
                        columns[field] = position
            return index, columns
    raise SystemExit(
        f"No header row in the error log: no cell reads {HEADER_KEY!r}. "
        "The export's real header sits below a blank spacer row; this looked at every row "
        "and found none."
    )


@dataclass
class ErrorLog:
    """A parsed QC export, plus what the parse had to leave out.

    `incomplete_rows` is not a curiosity. A row with a judgement in it and no Error is a
    row the annotator half-filled, and the only way anyone finds out is a reader that
    counts them and a report that prints the numbers.
    """
    path: Path
    header_row: int
    corrections: list
    incomplete_rows: list
    padding_rows: int


def read_error_log(path):
    """IN: path to the QC export   OUT: an ErrorLog.

    Rows are read to the END OF THE FILE rather than to the first gap, and each is sorted
    into one of three piles: padding (every meaningful cell blank), incomplete (content but
    no Error value), and a Correction. See the module docstring for what the pilot sheet
    does that makes the distinction load-bearing.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    header_index, columns = _find_header(rows)
    missing = {"session", "error"} - set(columns)
    if missing:
        raise SystemExit(
            f"The error log's header is missing {sorted(missing)}. Found: {sorted(columns)}."
        )

    def cell(row, field):
        position = columns.get(field)
        return row[position] if position is not None and position < len(row) else ""

    corrections, incomplete, padding = [], [], 0
    last_session = ""
    for number, row in enumerate(rows[header_index + 1:], start=header_index + 2):
        if all(_blank(cell(row, field)) for field in MEANINGFUL_FIELDS):
            padding += 1
            continue

        raw_error = str(cell(row, "error")).strip()
        if _blank(raw_error):
            incomplete.append(number)
            continue
        error = ERROR_ALIASES.get(_squash(raw_error))
        if error is None:
            raise SystemExit(
                f"Row {number} of the error log has an Error value this pass does not know "
                f"how to apply. Known values: {sorted(set(ERROR_ALIASES.values()))}."
            )

        # A blank Session ID mid-table is a spreadsheet fill-down the export dropped, not a
        # different session.
        session = str(cell(row, "session")).strip() or last_session
        last_session = session

        ai_text, ai_role = split_role_suffix(_text(cell(row, "ai_text")))
        actual_text, actual_role = split_role_suffix(_text(cell(row, "actual_text")))
        corrections.append(Correction(
            row=number,
            session=session,
            timestamp=parse_timestamp(cell(row, "timestamp")),
            line=parse_line(cell(row, "line")),
            speaker=ROLE_ALIASES.get(_squash(cell(row, "speaker"))),
            error=error,
            ai_text=ai_text,
            actual_text=actual_text,
            ai_role=ai_role,
            actual_role=actual_role,
            meaning_changed=_text(cell(row, "meaning_changed")),
            severity=_text(cell(row, "severity")),
            add_turn=_flag(cell(row, "add_turn")),
            subtract_turn=_flag(cell(row, "subtract_turn")),
            notes=_text(cell(row, "notes")),
        ))
    return ErrorLog(path=path, header_row=header_index + 1, corrections=corrections,
                    incomplete_rows=incomplete, padding_rows=padding)


def sessions(corrections):
    """IN: corrections   OUT: sorted distinct Session ID values.

    One sheet per session is the convention; this is how the CLI notices when it is not.
    """
    return sorted({c.session for c in corrections})
