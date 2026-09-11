"""Grading a machine transcript the way the annotator graded the first one.

STDLIB ONLY. CPU, a couple of seconds on a fifty-minute session, no models -- so every
cell of the model grid is gradeable on the login node the moment its transcript lands.

WHAT THIS IS FOR
----------------
The Stage 2 correction pass turned a human's 117-row error log into a corrected
reference. That reference is the first object in this project that can tell a transcript
it is wrong. This module is the other half: it takes ANY arm's transcript and produces
the same breakdown the human produced by hand -- the six error labels, and the split
between "the turns were wrong" and "the words were wrong" -- without anybody listening
to anything.

One human listening pass cost fifty minutes and produced one column of the grid. This
produces every other column from the same reference, in seconds, and the numbers are
directly comparable to hers because they are counted on the same definitions.

WHY NOT JUST A WORD ERROR RATE
------------------------------
WER answers one question -- were the words right -- and the pilot log says two thirds of
what went wrong was WHO SAID THEM. A single WER cannot distinguish an arm that mistyped
forty words from an arm that filed forty correct words under the wrong person, and those
two failures break different Stage 3 lanes. So every word is asked two questions, and the
answers are counted separately:

    what was said     did this reference word appear at all, and as itself
    who said it       of the words that DID appear, is each under the right person

The second question is the one the diarization bake-off is about, and it is measured over
the words both sides agree on -- so a bad typist cannot flatter or punish a name-tagger.

THE ORDER OF OPERATIONS, AND WHY IT IS NOT THE OBVIOUS ONE
----------------------------------------------------------
A diarizer has no idea who anyone is: SPEAKER_00 is not THERAPIST until something says so.
But working out which cluster is which person needs the two word streams already lined up,
and lining them up must not be influenced by the labels or the alignment starts assuming
its own answer. So it runs in three passes, in this order:

    1. line up the two word streams on the WORDS ALONE, labels ignored
    2. read the label mapping off the words that matched
    3. classify every difference, labels now meaningful

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It never looks at a timestamp. The corrected reference has interpolated word times and
zero-duration inserted turns, and is explicitly not scoreable for timing (see
transcript/corrections.py, _rebuild_turn) -- so a grader that used them would be reporting
the interpolation. Timing is DER's job, against a reference RTTM that does not exist yet.
Everything here is words and speaker labels, which is exactly the part of the reference
that is trustworthy today.

It also cannot produce three things the human's sheet carries, and no amount of work here
would change that:

    Severity / Meaning Changed  a judgement about the session, not a property of the text
    one row per error           the sheet has one row where a person decided one thing
                                happened; this has one FINDING per contiguous difference,
                                and a person may well have logged two adjacent slips as
                                one row or one long slip as two
    where inside a line         the reference itself cannot say (see the deck's slide 22)

So the row counts here are the same KIND of number as hers and not the identical number.
Compare them as profiles, not as a checksum.

THE CALIBRATION THAT IS AVAILABLE FOR FREE
------------------------------------------
Grade the arm the log was annotated against. The reference IS that arm plus her 117
corrections, so the differences this finds are those corrections seen from the other side,
and the profile should land near the sheet's own tally. That is the cheapest available
check that the classifier's definitions agree with the annotator's, and the CLI prints
both side by side when the correction report is on disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

from ..artifacts.error_log import (
    INSERTION,
    OMISSION,
    PROPER_NOUN,
    PUNCTUATION,
    SPEAKER_ATTRIBUTION,
    SUBSTITUTION,
)
from ..transcript.corrections import normalize
from ..transcript.turns import UNKNOWN_SPEAKER
from .labels import best_label_mapping

# The four things the machine can do to the turn structure, which are the four rows of the
# annotator's own Add Turn? table. The first three are what she ticked.
MISSED_TURN = "missed a whole turn"
WELDED_TURN = "welded into the other speaker's turn"
INVENTED_TURN = "invented a whole turn"
WORDS_ONLY = "words only"

STRUCTURAL = (MISSED_TURN, WELDED_TURN, INVENTED_TURN)

# Every label the grader can emit, in the order the deck's frequency chart uses.
ERROR_LABELS = (OMISSION, SPEAKER_ATTRIBUTION, INSERTION, SUBSTITUTION,
                PROPER_NOUN, PUNCTUATION)

# A capitalized word mid-sentence is the only evidence available that a word is a name,
# and these are the words that are capitalized for other reasons. "I" and its contractions
# are the whole list in English; the sentence-initial case is handled by looking at what
# precedes the word rather than by listing anything.
NEVER_A_NAME = {"i"}
SENTENCE_ENDS = (".", "?", "!", ":", ";")


@dataclass
class Token:
    """One word of one side, carrying everything the classifier has to ask about it.

    `text` is the raw word -- punctuation and capitals intact, because those are what the
    Punctuation and Proper Noun labels are about. `key` is the normalized form, and it is
    the ONLY thing the alignment compares, so a comma or a capital can never make two
    identical words fail to line up.
    """
    text: str
    key: str
    speaker: str
    turn: int
    first_in_turn: bool


def word_stream(turns, unknown=UNKNOWN_SPEAKER):
    """IN: a turn list   OUT: one Token per word, in order.

    Tokens whose normalized form is empty -- a lone dash, a stray bracket -- are dropped.
    They are not words, and letting them into the alignment would make one side's typography
    look like the other side's missing speech.
    """
    stream = []
    for position, turn in enumerate(turns):
        speaker = turn.get("speaker") or unknown
        first = True
        for raw in str(turn.get("text", "")).split():
            key = normalize(raw)
            if not key:
                continue
            stream.append(Token(raw, key, speaker, position, first))
            first = False
    return stream


def align(reference, candidate):
    """IN: two Token streams   OUT: difflib opcodes over their normalized words.

    AUTOJUNK IS OFF, AND THAT IS NOT A TUNING CHOICE. SequenceMatcher's default treats any
    element appearing in more than 1% of a sequence longer than 200 as junk not worth
    matching on. In fifty minutes of speech "the", "you" and "i" all clear 1% easily, so the
    default silently refuses to align on the most common words in the language and the
    result is confetti.

    The alignment is longest-common-subsequence rather than minimum-edit-distance, which is
    what makes it run in seconds on 7300 words with no numpy. It can put a difference
    boundary a word to the left or right of where a Levenshtein alignment would; it never
    changes what matched. Every arm is aligned by the same rule, which is what a comparison
    needs.
    """
    matcher = SequenceMatcher(None, [token.key for token in reference],
                              [token.key for token in candidate], autojunk=False)
    return matcher.get_opcodes()


def label_mapping(reference, candidate, opcodes):
    """IN: both streams + the opcodes   OUT: dict candidate label -> reference role.

    Read off the words that MATCHED, and only those: a word both sides wrote is a word
    whose speaker both sides had an opinion about, which is the only place the two label
    sets can be compared at all.

    An unlabeled candidate word votes for nothing rather than for a role. The join stamps
    a speaker only where a transcript span overlaps a diarized turn, so an UNKNOWN word is
    the diarizer declining to answer -- and a decline that voted would drag whichever role
    it happened to sit next to.
    """
    reference_roles, candidate_labels = [], []
    for tag, i1, i2, j1, _ in opcodes:
        if tag != "equal":
            continue
        for offset in range(i2 - i1):
            label = candidate[j1 + offset].speaker
            reference_roles.append(reference[i1 + offset].speaker)
            candidate_labels.append(None if label == UNKNOWN_SPEAKER else label)
    return best_label_mapping(reference_roles, candidate_labels)


@dataclass
class Finding:
    """One contiguous thing the machine got wrong, classified the annotator's way.

    `structure` is this module's answer to her Add Turn? column: which of the four things
    the machine did to the turn structure, three of which she ticked.
    """
    error: str
    structure: str
    reference_span: tuple
    candidate_span: tuple
    reference_words: int
    candidate_words: int
    reference_turns: tuple = ()
    candidate_turns: tuple = ()
    notes: list = field(default_factory=list)

    @property
    def add_turn(self):
        """TRUE on exactly the rows the annotator would have ticked."""
        return self.structure in STRUCTURAL


def _turn_extents(stream):
    """IN: a Token stream   OUT: dict turn index -> (first token index, last + 1).

    A turn's words are contiguous in the stream by construction, so its extent is all that
    is needed to ask whether a span covers the whole of it.
    """
    extents = {}
    for index, token in enumerate(stream):
        start, end = extents.get(token.turn, (index, index))
        extents[token.turn] = (min(start, index), max(end, index) + 1)
    return extents


def _turns_in(stream, start, end):
    """IN: a stream + a half-open token span   OUT: the turn indices it touches, in order."""
    seen = []
    for token in stream[start:end]:
        if not seen or seen[-1] != token.turn:
            seen.append(token.turn)
    return tuple(seen)


def _wholly_inside(extents, turns, spans):
    """IN: turn extents + turn indices + the spans of one opcode kind
    OUT: the turns whose EVERY word falls inside those spans

    This is the whole basis of the Add Turn? decision. Words missing from the middle of a
    turn the machine did produce are a typing error; a turn whose every word is missing is
    an utterance the machine never heard, and fixing it puts a turn on the page that was
    not there. The same test, run on the candidate's side, finds a turn the machine invented.
    """
    covered = []
    for turn in turns:
        first, last = extents[turn]
        if all(any(start <= index < end for start, end in spans) for index in range(first, last)):
            covered.append(turn)
    return covered


def _is_proper_noun(stream, index):
    """IN: a stream + a token index   OUT: does this word look like a name?

    A capital mid-sentence is the only evidence there is, so this rules out every other
    reason a word carries one: the first word of a turn, the first word after a full stop,
    and "I". It is a heuristic and is reported as its own count, so a run where it fires
    forty times says so rather than quietly inflating one label at another's expense.
    """
    token = stream[index]
    if not token.text[:1].isupper() or token.key in NEVER_A_NAME:
        return False
    if token.first_in_turn or index == 0:
        return False
    previous = stream[index - 1]
    if previous.turn != token.turn:
        return False
    return not previous.text.rstrip().endswith(SENTENCE_ENDS)


def _classify_replace(reference, i1, i2):
    """IN: the reference stream + the span the machine wrote something else over
    OUT: Proper Noun when a name was in it, else Substitution.

    Checked in that order because a mangled name IS a substitution, and the annotator's
    sheet gives it its own label -- the fix is different. A wrong word is retyped; a wrong
    name means the model has never seen it and every occurrence is wrong.
    """
    if any(_is_proper_noun(reference, index) for index in range(i1, i2)):
        return PROPER_NOUN
    return SUBSTITUTION


def _punctuation_runs(reference, candidate, i1, i2, j1):
    """IN: both streams + one equal opcode   OUT: runs where only the punctuation differs.

    A difference that survives here has already survived normalization, so it is a comma, a
    full stop or an apostrophe -- never a different word. Case alone is NOT a difference:
    moving a turn boundary re-capitalizes whatever now starts the turn, and counting that
    would report hundreds of punctuation errors for one relocated turn.
    """
    runs, start = [], None
    for offset in range(i2 - i1):
        left, right = reference[i1 + offset], candidate[j1 + offset]
        differs = left.text.lower() != right.text.lower()
        if differs and start is None:
            start = offset
        elif not differs and start is not None:
            runs.append((start, offset))
            start = None
    if start is not None:
        runs.append((start, i2 - i1))
    return runs


def _attribution_runs(reference, candidate, i1, i2, j1, mapping):
    """IN: both streams + one equal opcode + the label mapping
    OUT: runs of matched words filed under the wrong person

    A run is broken whenever the host turn changes on EITHER side, so one finding is one
    utterance inside one of the machine's turns -- which is the unit the annotator logged.
    An unlabeled candidate word breaks the run too and starts its own, because "the wrong
    person" and "no person at all" are different failures and the report counts them apart.
    """
    runs, start, previous = [], None, None
    for offset in range(i2 - i1):
        left, right = reference[i1 + offset], candidate[j1 + offset]
        mapped = mapping.get(right.speaker)
        wrong = mapped != left.speaker
        here = (left.turn, right.turn, right.speaker == UNKNOWN_SPEAKER)
        if wrong and (start is None or here != previous):
            if start is not None:
                runs.append((start, offset, previous[2]))
            start, previous = offset, here
        elif not wrong and start is not None:
            runs.append((start, offset, previous[2]))
            start, previous = None, None
    if start is not None:
        runs.append((start, i2 - i1, previous[2]))
    return runs


def find_differences(reference, candidate, opcodes, mapping):
    """IN: both streams, the opcodes, and the label mapping   OUT: list of Finding.

    The six labels fall out of the opcodes almost mechanically, and the mapping from one to
    the other is the definition of each label rather than a choice:

        delete   the reference has words the machine does not          Omission
        insert   the machine has words the reference does not          Insertion
        replace  both wrote something and they differ                  Substitution
                 ... and a capitalized word was among them            Proper Noun
        equal    same words, and the raw spelling differs             Punctuation
                 same words, filed under the wrong person             Speaker Attribution

    The last two are why `equal` is not simply skipped. An arm can match the reference word
    for word and still be wrong about every single speaker, which is exactly the failure
    this project is measuring.
    """
    reference_extents, candidate_extents = _turn_extents(reference), _turn_extents(candidate)
    deletions = [(i1, i2) for tag, i1, i2, _, _ in opcodes if tag == "delete"]
    insertions = [(j1, j2) for tag, _, _, j1, j2 in opcodes if tag == "insert"]

    findings = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "delete":
            turns = _turns_in(reference, i1, i2)
            missed = _wholly_inside(reference_extents, turns, deletions)
            findings.append(Finding(
                error=OMISSION,
                structure=MISSED_TURN if missed else WORDS_ONLY,
                reference_span=(i1, i2), candidate_span=(j1, j1),
                reference_words=i2 - i1, candidate_words=0,
                reference_turns=turns, candidate_turns=(),
            ))
        elif tag == "insert":
            turns = _turns_in(candidate, j1, j2)
            invented = _wholly_inside(candidate_extents, turns, insertions)
            findings.append(Finding(
                error=INSERTION,
                structure=INVENTED_TURN if invented else WORDS_ONLY,
                reference_span=(i1, i1), candidate_span=(j1, j2),
                reference_words=0, candidate_words=j2 - j1,
                reference_turns=(), candidate_turns=turns,
            ))
        elif tag == "replace":
            findings.append(Finding(
                error=_classify_replace(reference, i1, i2),
                structure=WORDS_ONLY,
                reference_span=(i1, i2), candidate_span=(j1, j2),
                reference_words=i2 - i1, candidate_words=j2 - j1,
                reference_turns=_turns_in(reference, i1, i2),
                candidate_turns=_turns_in(candidate, j1, j2),
            ))
        else:
            for start, end in _punctuation_runs(reference, candidate, i1, i2, j1):
                findings.append(Finding(
                    error=PUNCTUATION, structure=WORDS_ONLY,
                    reference_span=(i1 + start, i1 + end),
                    candidate_span=(j1 + start, j1 + end),
                    reference_words=end - start, candidate_words=end - start,
                    reference_turns=_turns_in(reference, i1 + start, i1 + end),
                    candidate_turns=_turns_in(candidate, j1 + start, j1 + end),
                ))
            for start, end, unlabeled in _attribution_runs(
                    reference, candidate, i1, i2, j1, mapping):
                host = _turns_in(candidate, j1 + start, j1 + end)
                whole = _wholly_inside(candidate_extents, host, [(j1 + start, j1 + end)])
                findings.append(Finding(
                    error=SPEAKER_ATTRIBUTION,
                    # The whole of the machine's turn is under the wrong name: nothing moves,
                    # the name changes. Part of it is, and fixing it cuts that turn open.
                    structure=WORDS_ONLY if len(whole) == len(host) else WELDED_TURN,
                    reference_span=(i1 + start, i1 + end),
                    candidate_span=(j1 + start, j1 + end),
                    reference_words=end - start, candidate_words=end - start,
                    reference_turns=_turns_in(reference, i1 + start, i1 + end),
                    candidate_turns=host,
                    notes=["the machine gave these words no speaker at all"] if unlabeled else [],
                ))
    return findings


def _edit_counts(opcodes):
    """IN: the opcodes   OUT: (matched, substituted, deleted, inserted) word counts.

    A replace of unequal length is split the way every WER implementation splits it: the
    overlapping part is substitution and the remainder is whichever of deletion or insertion
    the imbalance points at.
    """
    matched = substituted = deleted = inserted = 0
    for tag, i1, i2, j1, j2 in opcodes:
        left, right = i2 - i1, j2 - j1
        if tag == "equal":
            matched += left
        elif tag == "delete":
            deleted += left
        elif tag == "insert":
            inserted += right
        else:
            substituted += min(left, right)
            deleted += max(0, left - right)
            inserted += max(0, right - left)
    return matched, substituted, deleted, inserted


def grade(reference_turns, candidate_turns):
    """IN: the corrected reference's turns + one arm's turns   OUT: (findings, report).

    The report carries COUNTS AND LABEL NAMES ONLY, never a word of either transcript, so
    it can be read out of a job log, committed to a figure, or handed to somebody who is not
    cleared to read the session. Everything that would identify WHICH words went wrong stays
    in the findings, which the caller may write out separately and which are PHI.
    """
    reference = word_stream(reference_turns)
    candidate = word_stream(candidate_turns)
    opcodes = align(reference, candidate)
    mapping = label_mapping(reference, candidate, opcodes)
    findings = find_differences(reference, candidate, opcodes, mapping)

    matched, substituted, deleted, inserted = _edit_counts(opcodes)
    total = len(reference)
    roles = {token.speaker for token in reference}

    attribution = [f for f in findings if f.error == SPEAKER_ATTRIBUTION]
    misattributed = sum(f.reference_words for f in attribution)
    unlabeled = sum(f.reference_words for f in attribution if f.notes)

    by_error = {
        label: {
            "findings": sum(1 for f in findings if f.error == label),
            "reference_words": sum(f.reference_words for f in findings if f.error == label),
        }
        for label in ERROR_LABELS
    }
    by_structure = {
        structure: sum(1 for f in findings if f.structure == structure)
        for structure in (MISSED_TURN, WELDED_TURN, INVENTED_TURN, WORDS_ONLY)
    }
    structural = sum(1 for f in findings if f.add_turn)

    report = {
        "reference_words": total,
        "candidate_words": len(candidate),
        "reference_turns": len(reference_turns),
        "candidate_turns": len(candidate_turns),
        "role_mapping": mapping,
        # INCOMPLETE means at least one of the machine's clusters got no ROLE, either
        # because it agreed with the reference nowhere or because there was no role left to
        # give it -- best_label_mapping pads with EXTRA_n so a spurious third speaker is
        # never silently folded onto one of the two real people. Both cases make every word
        # under that cluster a misattribution, which is the right answer and a very loud
        # one, so it is flagged rather than left to be inferred from the numbers.
        "role_mapping_complete": all(
            mapping.get(token.speaker) in roles or token.speaker == UNKNOWN_SPEAKER
            for token in candidate),
        # THE WHAT-WAS-SAID CHANNEL. Depends on the typist and on nothing else, because the
        # name-tagger never reads a word and the stopwatch never changes one.
        "words": {
            "matched": matched,
            "substituted": substituted,
            "omitted": deleted,
            "inserted": inserted,
        },
        "word_error_rate": (substituted + deleted + inserted) / total if total else float("nan"),
        # THE WHO-SAID-IT CHANNEL, over the words both sides agree on -- so a bad typist can
        # neither flatter nor punish a name-tagger. This is the number the bake-off is about.
        "speaker_error_rate": misattributed / matched if matched else float("nan"),
        "misattributed_words": misattributed,
        "unlabeled_matched_words": unlabeled,
        "findings": len(findings),
        "findings_changing_turn_structure": structural,
        "findings_changing_words_only": len(findings) - structural,
        "by_error": by_error,
        "by_structure": by_structure,
    }
    return findings, report


def finding_records(findings, reference, candidate):
    """IN: findings + both Token streams   OUT: one dict per finding, WITH THE WORDS IN IT.

    THIS IS THE PHI-BEARING HALF and the only thing here that carries transcript text. It
    exists because a classifier nobody can inspect is a classifier nobody should believe:
    somebody inside the fence reads a sample of these once, confirms that what the grader
    calls a welded turn is a welded turn, and after that the counts stand on their own.
    """
    records = []
    for finding in findings:
        i1, i2 = finding.reference_span
        j1, j2 = finding.candidate_span
        records.append({
            "error": finding.error,
            "structure": finding.structure,
            "add_turn": finding.add_turn,
            "reference_text": " ".join(token.text for token in reference[i1:i2]),
            "candidate_text": " ".join(token.text for token in candidate[j1:j2]),
            "reference_speakers": sorted({token.speaker for token in reference[i1:i2]}),
            "candidate_speakers": sorted({token.speaker for token in candidate[j1:j2]}),
            "reference_turns": list(finding.reference_turns),
            "candidate_turns": list(finding.candidate_turns),
            "notes": finding.notes,
        })
    return records
