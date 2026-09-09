"""Applying the Stage 2 QC error log to a machine transcript.

STDLIB ONLY. CPU, well under a second, no models -- so the corrected reference can be
rebuilt in any env the moment the annotator edits a row.

WHAT THIS PRODUCES AND WHY IT IS NOT A `str.replace` LOOP
---------------------------------------------------------
The first version of this pass loaded the log, took each row's AI snippet, and called
`text.replace(snippet, actual)` on the whole transcript. Three things are wrong with that,
and all three are silent:

1. It ignores the Line column, so a two-word snippet like the missing "mm hmm" replaces
   EVERY occurrence in fifty minutes of speech instead of the one the annotator heard.
2. It cannot represent 47 of the 117 rows at all. Those say the machine transcribed
   nothing -- there is no snippet to replace, only a place where a turn belongs.
3. It cannot represent the 32 speaker-attribution rows even in principle. Those do not
   change the words; they change WHO SAID THEM, which means splitting one turn into
   three, and a string replacement has no concept of a turn.

So the pass works on the TURN STRUCTURE instead. Each row resolves to a locus -- a turn,
and where possible a character span inside it -- and each locus turns into one of four
edits:

    replace    the words in the span become the annotator's words (Substitution,
               Proper Noun, Punctuation, Insertion, and the Omissions logged with the
               surrounding phrase)
    insert     a new turn appears at a point (the Omissions where the machine wrote
               nothing, and Add Turn? is TRUE)
    extract    the span leaves its host turn and becomes a turn of its own, attributed to
               the other speaker (Speaker Attribution with Add Turn? TRUE) -- which splits
               the host in two around it
    relabel    the whole host turn was attributed to the wrong person (Speaker
               Attribution with Add Turn? FALSE)

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not invent word timings. A corrected turn's start and end are the host turn's own,
split PROPORTIONALLY TO CHARACTER OFFSET where a turn is cut in two, and an inserted turn
takes the timestamp the annotator wrote down. Every turn therefore carries a
"time_source" saying which of the three it is, because a linear-in-characters guess is
good enough to read the transcript against the audio and is NOT good enough to score a
diarizer against. Turning this into a reference RTTM needs the corrected words re-aligned
to the waveform; that is a separate step and must not be faked here.

It also refuses to apply two overlapping edits to the same words. A misattribution is
logged as a PAIR of rows -- a Speaker Attribution row that moves the words to the right
speaker, and an Insertion row that takes them out of where they were -- and applying both
would delete the words twice. The earlier row in sheet order wins; the later is reported
as superseded rather than quietly dropped.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from ..artifacts.error_log import (
    INSERTION,
    OMISSION,
    SPEAKER_ATTRIBUTION,
)
from ..evaluate.labels import best_label_mapping
from .render import DIALOGUE_LINE, HEADING_LINE
from .turns import UNKNOWN_SPEAKER

# How far from the line the annotator wrote down a snippet may be found and still be
# believed. Turns are ~34 s long in the pilot session, so three either side is roughly a
# three-minute window: wide enough to absorb an off-by-a-few line number typed by hand,
# narrow enough that a two-word backchannel cannot match the wrong half of the session.
NEIGHBOURHOOD = 3

# Shorter than this, normalized, a snippet is not evidence of anything -- "a", "I", "mm"
# occur hundreds of times -- so such a row is placed by its line number alone.
MIN_SNIPPET = 3

# How a locus was found, most trustworthy first. Reported as counts, so a run that leaned
# on the weak locators says so without anyone reading the transcript.
BY_LINE_AND_SNIPPET = "line+snippet"
BY_SNIPPET_NEAR_LINE = "snippet-near-line"
BY_SNIPPET_ANYWHERE = "snippet-anywhere"
BY_LINE = "line-only"
BY_TIMESTAMP = "timestamp-only"
ALREADY_CLAIMED = "already-claimed"
UNRESOLVED = "unresolved"

# What happened to a row.
APPLIED = "applied"
SUPERSEDED = "superseded"
UNPLACED = "unplaced"
# The words this row is about were already dealt with by an earlier row: a misattribution
# is logged twice, once as the attribution that moves the words and once as the insertion
# that takes them out of where they were, and moving them IS taking them out. Not a
# failure, and not an application either.
ACCOUNTED = "accounted for by an earlier row"

_PUNCT = re.compile(r"[^\w\s']+")
_SPACE = re.compile(r"\s+")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                         "–": "-", "—": "-"})


def normalize(text):
    """IN: any text   OUT: the form snippets are matched in.

    Case-folded, curly punctuation straightened, everything but word characters,
    apostrophes and single spaces removed. The annotator typed the snippets by hand from
    the audio while reading the .txt, so they agree with the transcript on the WORDS and
    routinely disagree on capitalization, commas and which kind of apostrophe -- matching
    on the raw characters finds nothing about a third of the time.
    """
    folded = unicodedata.normalize("NFKC", str(text)).translate(_QUOTES).lower()
    return _SPACE.sub(" ", _PUNCT.sub(" ", folded)).strip()


def _normalize_with_map(text):
    """IN: a turn's text   OUT: (normalized text, source index for each normalized char).

    The parallel index is the whole point: a match in normalized space has to come back as
    a span in the ORIGINAL characters, or the replacement lands on the wrong bytes.
    """
    folded = unicodedata.normalize("NFKC", str(text)).translate(_QUOTES).lower()
    out, source = [], []
    at_space = True
    for position, character in enumerate(folded):
        if character.isspace() or _PUNCT.fullmatch(character):
            if at_space:
                continue
            out.append(" ")
            source.append(position)
            at_space = True
            continue
        out.append(character)
        source.append(position)
        at_space = False
    while out and out[-1] == " ":
        out.pop()
        source.pop()
    return "".join(out), source


class DocumentIndex:
    """Every turn's text as one normalized string, with each character's owner recorded.

    IN:  the turn list
    OUT: an object whose `find` returns matches as (turn, start, end) in ORIGINAL offsets

    One flat string rather than one per turn, because a snippet the annotator typed can
    run across a turn boundary -- that is precisely what a speaker-attribution error looks
    like from the reader's side, the machine having welded two people's words into one
    block. A per-turn search cannot see those at all.
    """

    def __init__(self, turns):
        pieces, owners = [], []
        for position, turn in enumerate(turns):
            normalized, source = _normalize_with_map(turn.get("text", ""))
            if not normalized:
                continue
            if pieces:
                pieces.append(" ")
                owners.append((position, None))
            pieces.append(normalized)
            owners.extend((position, offset) for offset in source)
        self.text = "".join(pieces)
        self.owners = owners

    def find(self, needle, near=None):
        """IN: a normalized snippet + an optional turn index to prefer
        OUT: list of (turn, start, end, crosses) matches, nearest `near` first

        `crosses` is True when the match began in one turn and ended in another; the caller
        decides what to do with that rather than having it hidden here.
        """
        if not needle or len(needle) < MIN_SNIPPET or not self.text:
            return []
        matches = []
        for found in re.finditer(re.escape(needle), self.text):
            first, last = found.start(), found.end() - 1
            start_turn, start_offset = self.owners[first]
            end_turn, end_offset = self.owners[last]
            if start_offset is None or end_offset is None:
                continue
            crosses = start_turn != end_turn
            end = end_offset + 1 if not crosses else None
            matches.append((start_turn, start_offset, end, crosses))
        if near is not None:
            matches.sort(key=lambda match: abs(match[0] - near))
        return matches


def line_lookup(index):
    """IN: the render index from render_with_line_index   OUT: line number -> entry.

    Only heading and dialogue lines are kept. A line number pointing at the summary block
    or at a blank separator carries no turn, and letting it resolve to "turn None" would
    make an unplaceable row look placed.
    """
    return {
        entry["line"]: entry
        for entry in index
        if entry["kind"] in (HEADING_LINE, DIALOGUE_LINE) and entry["turn"] is not None
    }


@dataclass
class Locus:
    """Where a correction lands. `end` is None for an insertion point.

    `matched` names WHICH column was found verbatim on the page -- "actual", "ai", or None
    for a row placed by its line number alone. It is the difference between "these words
    are in the transcript under the wrong name" and "these words are not in the transcript
    at all", and the two call for opposite edits, so it is recorded rather than inferred.
    """
    turn: int
    start: int | None
    end: int | None
    how: str
    crosses_turns: bool = False
    matched: str | None = None


@dataclass
class Edit:
    """One resolved edit against one host turn, in that turn's own character offsets."""
    row: int
    kind: str                     # replace | insert | extract | relabel
    start: int
    end: int
    text: str = ""
    speaker: str | None = None
    timestamp: float | None = None
    notes: list = field(default_factory=list)


REPLACE, INSERT, EXTRACT, RELABEL = "replace", "insert", "extract", "relabel"


def search_texts(correction):
    """IN: a Correction   OUT: the snippets to look for, most-likely-verbatim first.

    WHAT IS ON THE PAGE is the only thing worth searching for, and for five of the six
    error types that is the MACHINE's text: the annotator's words are, by definition of an
    omission or a substitution, what the machine did not write.

    A speaker-attribution row is the exception and is searched the other way round. There
    the machine heard the words correctly and only filed them under the wrong person, so
    the annotator's cell is the same utterance and finding it locates the characters to
    move. (In the pilot sheet the two cells differ only in the role name appended to each,
    which the reader has already split off, so either would do -- but preferring the
    annotator's is what makes the extracted words the ones the annotator vouched for.)

    Searching the annotator's column on the other five types is not merely useless, it is
    ACTIVELY WRONG. A missing "mm hmm" occurs a hundred times elsewhere in fifty minutes of
    speech; matching it makes 34 rows report that the omitted words were found on the page,
    which is a coincidence dressed up as evidence.
    """
    if correction.error == SPEAKER_ATTRIBUTION:
        candidates = [("actual", correction.actual_text), ("ai", correction.ai_text)]
    else:
        candidates = [("ai", correction.ai_text)]
    return [(which, normalize(text)) for which, text in candidates if text]


def _time_locus(turns, timestamp):
    """IN: turns + a time in seconds   OUT: the turn covering it, else the nearest one."""
    if timestamp is None or not turns:
        return None
    for position, turn in enumerate(turns):
        if turn["start"] <= timestamp <= turn["end"]:
            return position
    return min(range(len(turns)),
               key=lambda position: min(abs(turns[position]["start"] - timestamp),
                                        abs(turns[position]["end"] - timestamp)))


def locate(correction, turns, lines, document, claimed=None):
    """IN: a Correction + the turn list + the line lookup + the DocumentIndex
           + the character spans earlier rows have already claimed, per turn
    OUT: a Locus, or None when the row cannot be placed at all

    The order is the order of decreasing evidence, and the chosen locator is recorded on
    the Locus so the run can report how much of its work rested on which.

    `claimed` is what makes two rows about the same two-word backchannel land on two
    DIFFERENT occurrences of it. Without it the second row matched the same characters as
    the first, collided with them, and was reported as superseded -- five of the pilot
    session's attribution rows were lost that way, in a session where attribution is the
    thing being measured.
    """
    hint = lines.get(correction.line) if correction.line else None
    hint_turn = hint["turn"] if hint else None
    claimed = claimed or {}

    blocked = set()

    def free(match):
        turn, start, end, _ = match
        if end is None:
            return True
        if any(start < taken_end and taken_start < end
               for taken_start, taken_end in claimed.get(turn, [])):
            blocked.add(turn)
            return False
        return True

    for which, needle in search_texts(correction):
        raw = document.find(needle, near=hint_turn)
        matches = [match for match in raw if free(match)]
        # THE HINT TURN IS DECISIVE WHEN IT SPOKE. If the words are at the logged line and
        # every occurrence there is already spoken for, this row is not free to go looking
        # for the same words in a NEIGHBOURING turn -- that is how the second row of a
        # misattribution pair came to delete an unrelated "okay" two turns away.
        #
        # What it does instead depends on whether the row has a turn to create. A row that
        # only removes or rewrites the machine's words has nothing left to do once those
        # words are gone, and is reported as already handled. A row that ADDS a turn always
        # has work: two attribution rows about two separate "yeah"s in one turn are two
        # utterances, and the machine only ever transcribed one of them -- so the second
        # falls through to its line number and is INSERTED rather than moved.
        if hint_turn is not None and hint_turn in blocked:
            if not correction.add_turn:
                return Locus(hint_turn, None, None, ALREADY_CLAIMED)
            break
        if not matches:
            continue
        if hint_turn is not None:
            exact = [match for match in matches if match[0] == hint_turn]
            if exact:
                turn, start, end, crosses = exact[0]
                return Locus(turn, start, end, BY_LINE_AND_SNIPPET, crosses, which)
            near = [match for match in matches if abs(match[0] - hint_turn) <= NEIGHBOURHOOD]
            if near:
                turn, start, end, crosses = near[0]
                return Locus(turn, start, end, BY_SNIPPET_NEAR_LINE, crosses, which)
        # No line to anchor to, or the line points somewhere the words are not. A single
        # unambiguous occurrence in the whole session is still good evidence; several are
        # not, and fall through to the line or the timestamp.
        if len(matches) == 1:
            turn, start, end, crosses = matches[0]
            return Locus(turn, start, end, BY_SNIPPET_ANYWHERE, crosses, which)

    # Every occurrence of these words was spoken for by an earlier row. Say that, rather
    # than reporting the row as one whose words could not be found -- the difference is
    # between a log that disagrees with the audio and a log that describes one error twice.
    # Again only for a row with no turn to create: one that adds a turn still has work to
    # do, and continues on to its line number.
    if blocked and not correction.add_turn and (hint_turn is None or hint_turn in blocked):
        return Locus(sorted(blocked)[0], None, None, ALREADY_CLAIMED)

    if hint is not None:
        # A heading line means "this turn", with no offset inside it; a dialogue line gives
        # the span of that line's own words, and an insertion goes at its end.
        if hint["kind"] == DIALOGUE_LINE:
            return Locus(hint["turn"], hint["end"], None, BY_LINE)
        return Locus(hint["turn"], None, None, BY_LINE)

    by_time = _time_locus(turns, correction.timestamp)
    if by_time is not None:
        return Locus(by_time, None, None, BY_TIMESTAMP)
    return None


def to_edit(correction, locus, turns):
    """IN: a Correction + its Locus + the turn list   OUT: an Edit.

    This is where the six error labels and the Add Turn? flag become the four things that
    can actually be done to a turn.
    """
    turn = turns[locus.turn]
    span = (locus.start, locus.end) if locus.start is not None and locus.end is not None else None
    notes = []
    if locus.crosses_turns:
        notes.append("snippet ran across a turn boundary; clipped to the turn it began in")

    if correction.error == SPEAKER_ATTRIBUTION:
        if not correction.add_turn:
            return Edit(correction.row, RELABEL, 0, 0,
                        speaker=correction.speaker, notes=notes)
        if span:
            # Matched on the annotator's own words: those characters ARE the utterance, so
            # they move across intact. Matched on the machine's rendering instead: the
            # machine's characters go away and the annotator's take their place.
            moved = (turn["text"][span[0]:span[1]] if locus.matched == "actual"
                     else (correction.actual_text or ""))
            return Edit(correction.row, EXTRACT, span[0], span[1], text=moved,
                        speaker=correction.speaker, timestamp=correction.timestamp, notes=notes)
        point = locus.start if locus.start is not None else len(turn["text"])
        notes.append("no span found; the reattributed words were inserted rather than moved")
        return Edit(correction.row, INSERT, point, point, text=correction.actual_text or "",
                    speaker=correction.speaker, timestamp=correction.timestamp, notes=notes)

    if correction.error == OMISSION and correction.is_pure_omission:
        point = locus.start if locus.start is not None else len(turn["text"])
        if correction.add_turn:
            return Edit(correction.row, INSERT, point, point, text=correction.actual_text or "",
                        speaker=correction.speaker, timestamp=correction.timestamp, notes=notes)
        # Words missing from inside a turn the diarizer got right: they go back into it.
        return Edit(correction.row, REPLACE, point, point,
                    text=" " + (correction.actual_text or ""), notes=notes)

    if span is None:
        notes.append("the machine's words could not be located; nothing was replaced")
        return Edit(correction.row, REPLACE, 0, 0, text="", notes=notes + ["unplaced"])

    if correction.add_turn and correction.error == INSERTION:
        notes.append("Add Turn? was TRUE but an insertion gives no boundary to split on; "
                     "the words were removed and no turn was created")
    return Edit(correction.row, REPLACE, span[0], span[1],
                text=correction.actual_text or "", notes=notes)


def _interpolate(turn, offset, length):
    """IN: a turn + a character offset into it + its text length   OUT: a time inside it.

    Linear in characters. Speech is not, which is exactly why the result is labelled
    "interpolated" wherever it is used rather than being presented as a measurement.
    """
    if length <= 0:
        return turn["start"]
    share = max(0.0, min(1.0, offset / length))
    return turn["start"] + (turn["end"] - turn["start"]) * share


def _rebuild_turn(turn, edits):
    """IN: one host turn + the Edits that land on it   OUT: the turns it becomes.

    Walks the host's text once, left to right, flushing the host's own accumulated words
    into a turn whenever an edit interrupts them with somebody else's.

    EVERY piece is timed by interpolating its CHARACTER RANGE in the host turn, including
    the inserted ones -- so the pieces of a split turn tile its span exactly, in order,
    without overlapping. The annotator's own observed time is kept as `logged_at` beside
    the interpolated span rather than being written into `start`, which is what an earlier
    version did: a logged time three seconds later than the split point it belongs to
    produced turns that overlapped their neighbours and a talk-time table summing to 107%
    of the session. An inserted turn therefore has ZERO duration, which is the honest
    answer -- how long a backchannel lasted is not in the spreadsheet.
    """
    text = turn.get("text", "")
    length = len(text)
    speaker = turn.get("speaker", UNKNOWN_SPEAKER)
    rows = list(turn.get("corrections", []))

    for edit in edits:
        if edit.kind == RELABEL:
            speaker = edit.speaker or speaker
            rows.append(edit.row)

    ordered = sorted((edit for edit in edits if edit.kind != RELABEL),
                     key=lambda edit: (edit.start, edit.end))
    produced, buffer, buffer_start, cursor = [], [], 0, 0

    def flush(end_offset):
        if not buffer:
            return
        body = "".join(buffer).strip()
        buffer.clear()
        if not body:
            return
        whole = buffer_start == 0 and end_offset >= length
        produced.append({
            "speaker": speaker,
            "start": _interpolate(turn, buffer_start, length),
            "end": _interpolate(turn, end_offset, length),
            "text": _SPACE.sub(" ", body),
            "time_source": turn.get("time_source", "asr") if whole else "interpolated",
            "logged_at": None,
            "origin": turn.get("origin", "asr"),
            "corrections": rows,
        })

    for edit in ordered:
        buffer.append(text[cursor:edit.start])
        rows.append(edit.row)
        if edit.kind == REPLACE:
            buffer.append(edit.text)
            cursor = edit.end
            continue
        # insert / extract: somebody else speaks here, so the host's run ends.
        flush(edit.start)
        buffer_start = edit.end
        produced.append({
            "speaker": edit.speaker or UNKNOWN_SPEAKER,
            "start": _interpolate(turn, edit.start, length),
            "end": _interpolate(turn, edit.end, length),
            "text": _SPACE.sub(" ", edit.text.strip()),
            "time_source": "interpolated",
            "logged_at": edit.timestamp,
            "origin": "inserted" if edit.kind == INSERT else "reattributed",
            "corrections": [edit.row],
        })
        cursor = edit.end

    buffer.append(text[cursor:])
    flush(length)

    if not produced:
        # Every word in the turn was deleted. The turn goes with them rather than being
        # left as an empty block with a timestamp on it.
        return []
    if len(produced) == 1 and not ordered:
        produced[0]["speaker"] = speaker
    return produced


def merge_adjacent(turns):
    """IN: the rebuilt turn list   OUT: the same list with same-speaker neighbours joined.

    Cutting a turn open to insert a backchannel leaves the host's own words either side of
    it; when the backchannel turns out to belong to the host as well, the three pieces are
    one turn again and should read as one.
    """
    merged = []
    for turn in turns:
        if merged and merged[-1]["speaker"] == turn["speaker"]:
            previous = merged[-1]
            previous["text"] = (previous["text"] + " " + turn["text"]).strip()
            previous["end"] = max(previous["end"], turn["end"])
            previous["corrections"] = sorted(set(previous["corrections"]) | set(turn["corrections"]))
            if turn["origin"] != "asr":
                previous["origin"] = turn["origin"]
            if previous["logged_at"] is None:
                previous["logged_at"] = turn["logged_at"]
            continue
        merged.append(dict(turn))
    return merged


def role_mapping(turns, corrections, placements):
    """IN: turns + corrections + row -> Locus   OUT: (machine label -> role, the vote table).

    Which anonymous cluster is the therapist is not guessed from lexical cues here, and it
    is not inferred either. The error log states it outright on every attribution row: the
    AI Transcript cell ends in the role the DIARIZER assigned those words to, so the host
    turn's machine label and that role are the same person by construction.

    Rows with no such role still vote, from AGREEMENT: a row that corrects the words inside
    an existing turn without creating a boundary is the annotator accepting the diarizer's
    speaker at that point.

    WHAT MUST NOT VOTE, and this is the subtle one: a row with Add Turn? TRUE and no role
    of its own. Those are utterances the machine missed ENTIRELY, and the turn they are
    inserted into is usually the other person's -- so the host label and the logged role
    are deliberately different, and counting them as agreement poisons the tally. On the
    pilot sheet that alone took the SPEAKER_00 vote from decisive to 20-17.

    The vote table is returned alongside the mapping because a near-tie means the mapping is
    not safe to use, and that has to be visible to someone who cannot read the transcript.
    """
    roles = sorted({role for c in corrections
                    for role in (c.speaker, c.actual_role, c.ai_role) if role})
    votes = {}

    def vote(label, role, kind):
        bucket = votes.setdefault(label, {"roles": {}, "evidence": {}})
        bucket["roles"][role] = bucket["roles"].get(role, 0) + 1
        bucket["evidence"][kind] = bucket["evidence"].get(kind, 0) + 1

    for correction in corrections:
        locus = placements.get(correction.row)
        if locus is None:
            continue
        label = turns[locus.turn].get("speaker") or UNKNOWN_SPEAKER
        if label == UNKNOWN_SPEAKER:
            continue
        if correction.ai_role:
            # The strongest evidence there is: the annotator wrote down which role the
            # machine had put these words under, and the machine's label for that turn is
            # right here. Nothing is inferred.
            vote(label, correction.ai_role, "the machine's own label")
            continue
        if correction.error == SPEAKER_ATTRIBUTION or correction.add_turn:
            continue
        vote(label, correction.speaker, "agreement")

    # best_label_mapping takes two positionally-aligned label columns, so the tally is
    # expanded back into columns rather than a second argmax being written here: the
    # brute-force pairing it does is already the one the word-level comparison uses.
    reference, candidate = [], []
    for label, tally in votes.items():
        for role in roles:
            reference.extend([role] * tally["roles"].get(role, 0))
            candidate.extend([label] * tally["roles"].get(role, 0))
    mapping = best_label_mapping(reference, candidate)

    # A cluster nobody logged an error against gets its role by ELIMINATION, and only when
    # the elimination is unique: one unmapped label, one unclaimed role. Two of either is
    # a guess about which cluster is the therapist, and a guess is what this function
    # exists not to make -- so the labels stay anonymous and the report says the mapping
    # is incomplete.
    present = {turn.get("speaker") for turn in turns} - {None, UNKNOWN_SPEAKER}
    unmapped = sorted(present - set(mapping))
    spare = [role for role in roles if role not in mapping.values()]
    if len(unmapped) == 1 and len(spare) == 1:
        mapping[unmapped[0]] = spare[0]
    return mapping, votes


def apply_corrections(transcript_turns, corrections, index):
    """IN: the baseline turn list + the parsed log + the render index
    OUT: (corrected turns, report dict)

    The report carries counts and row numbers only, never text, so it can be read out of a
    job log by somebody who is not cleared to read the transcript itself.
    """
    turns = [dict(turn) for turn in transcript_turns]
    for turn in turns:
        turn.setdefault("origin", "asr")
        turn.setdefault("time_source", "asr")
        turn.setdefault("corrections", [])

    lines = line_lookup(index)
    document = DocumentIndex(turns)

    placements, edits_by_turn = {}, {}
    outcomes, locators, notes_by_row = {}, {}, {}
    # turn index -> the character spans already spoken for. Read by the locator on the way
    # in and written by the overlap check on the way out.
    claimed = {}

    for correction in corrections:
        locus = locate(correction, turns, lines, document, claimed)
        if locus is None:
            outcomes[correction.row] = UNPLACED
            locators[correction.row] = UNRESOLVED
            continue
        placements[correction.row] = locus
        locators[correction.row] = locus.how
        if locus.how == ALREADY_CLAIMED:
            outcomes[correction.row] = ACCOUNTED
            continue
        edit = to_edit(correction, locus, turns)
        if "unplaced" in edit.notes:
            outcomes[correction.row] = UNPLACED
            notes_by_row[correction.row] = [n for n in edit.notes if n != "unplaced"]
            continue

        # Overlap check, in sheet order: the misattribution pairs describe the same words
        # twice and must not both fire.
        taken = claimed.setdefault(locus.turn, [])
        if edit.kind in (REPLACE, EXTRACT) and edit.end > edit.start and any(
            edit.start < end and start < edit.end for start, end in taken
        ):
            outcomes[correction.row] = SUPERSEDED
            continue
        if edit.end > edit.start:
            taken.append((edit.start, edit.end))

        edits_by_turn.setdefault(locus.turn, []).append(edit)
        outcomes[correction.row] = APPLIED
        if edit.notes:
            notes_by_row[correction.row] = edit.notes

    mapping, votes = role_mapping(turns, corrections, placements)

    rebuilt = []
    for position, turn in enumerate(turns):
        rebuilt.extend(_rebuild_turn(turn, edits_by_turn.get(position, [])))
    for turn in rebuilt:
        turn["speaker"] = mapping.get(turn["speaker"], turn["speaker"])
    corrected = merge_adjacent(rebuilt)

    kinds = {}
    for edits in edits_by_turn.values():
        for edit in edits:
            kinds[edit.kind] = kinds.get(edit.kind, 0) + 1

    by_error = {}
    for correction in corrections:
        bucket = by_error.setdefault(
            correction.error, {APPLIED: 0, ACCOUNTED: 0, SUPERSEDED: 0, UNPLACED: 0})
        bucket[outcomes.get(correction.row, UNPLACED)] += 1

    found = {}
    for correction in corrections:
        locus = placements.get(correction.row)
        bucket = found.setdefault(correction.error, {"actual": 0, "ai": 0, "not on the page": 0})
        bucket[locus.matched if locus and locus.matched else "not on the page"] += 1

    # THE VERDICT ON THE DIARIZER, and the single number the correction pass exists to
    # produce. Every other count here describes how well the pass ran; this one describes
    # how well the DIARIZER ran, and it comes straight off the annotator's own Add Turn?
    # column. A row flagged TRUE is one the annotator is saying the diarizer did not merely
    # mistype but structurally missed: a whole utterance it never heard, or one it welded
    # into the wrong speaker's turn.
    structural = sum(1 for correction in corrections if correction.add_turn)

    report = {
        "corrections_read": len(corrections),
        "rows_changing_turn_structure": structural,
        "rows_changing_words_only": len(corrections) - structural,
        "words_found_on_the_page": found,
        "by_error": by_error,
        "by_locator": {
            locator: sum(1 for value in locators.values() if value == locator)
            for locator in sorted(set(locators.values()))
        },
        "edits_by_kind": kinds,
        "turns_before": len(turns),
        "turns_after": len(corrected),
        "words_before": sum(len(turn["text"].split()) for turn in turns),
        "words_after": sum(len(turn["text"].split()) for turn in corrected),
        "role_mapping": mapping,
        "role_mapping_complete": all(
            (turn.get("speaker") in mapping or turn.get("speaker") == UNKNOWN_SPEAKER)
            for turn in turns),
        "role_vote": votes,
        "unplaced_rows": sorted(row for row, state in outcomes.items() if state == UNPLACED),
        "accounted_rows": sorted(row for row, state in outcomes.items() if state == ACCOUNTED),
        "superseded_rows": sorted(row for row, state in outcomes.items() if state == SUPERSEDED),
        "row_notes": {str(row): note for row, note in sorted(notes_by_row.items())},
        "turns_by_origin": {
            origin: sum(1 for turn in corrected if turn["origin"] == origin)
            for origin in sorted({turn["origin"] for turn in corrected})
        },
        "turns_by_time_source": {
            source: sum(1 for turn in corrected if turn["time_source"] == source)
            for source in sorted({turn["time_source"] for turn in corrected})
        },
    }
    return corrected, report
