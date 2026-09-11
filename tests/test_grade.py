"""Grading an arm against the corrected reference in the annotator's own categories.

Synthetic throughout. Every fixture is a short invented exchange written in the test that
uses it, so the suite still contains no session content and still runs anywhere.

The tests are grouped by the decision each one pins down, because the six labels are
almost mechanical and all the judgement is in the other column: did this difference change
the TURN STRUCTURE, or only the words inside a turn the machine already had?

The candidate side always goes through group_into_turns, the same call Stage 1c uses, so a
fixture cannot accidentally assert on a turn arrangement a real transcript could not
produce -- two consecutive segments with the same label are one turn, and a test that
forgot it would be testing nothing.
"""

from psych_asr.artifacts.error_log import (
    INSERTION,
    OMISSION,
    PARTICIPANT,
    PROPER_NOUN,
    PUNCTUATION,
    SPEAKER_ATTRIBUTION,
    SUBSTITUTION,
    THERAPIST,
)
from psych_asr.evaluate.grade import (
    ERROR_LABELS,
    INVENTED_TURN,
    MISSED_TURN,
    WELDED_TURN,
    WORDS_ONLY,
    align,
    grade,
    label_mapping,
    word_stream,
)
from psych_asr.transcript.turns import UNKNOWN_SPEAKER, group_into_turns


def reference(*pairs):
    """IN: (role, text) pairs   OUT: a reference turn list, one second per turn.

    The roles are spelled THERAPIST / PARTICIPANT because that is what the correction pass
    writes: the clusters are already named on the reference side and anonymous on the
    machine's, which is the asymmetry the grader has to resolve.
    """
    return [{"speaker": role, "start": float(index), "end": float(index) + 1.0, "text": text}
            for index, (role, text) in enumerate(pairs)]


def machine(*pairs):
    """IN: (cluster label, text) segment pairs   OUT: the arm's turn list.

    Grouped rather than used raw, so consecutive same-label segments collapse exactly as
    they do in the pipeline.
    """
    segments = [{"speaker": label, "start": float(index), "end": float(index) + 1.0,
                 "text": text}
                for index, (label, text) in enumerate(pairs)]
    return group_into_turns(segments)


def run(reference_turns, candidate_turns):
    return grade(reference_turns, candidate_turns)


def only(findings, error):
    return [finding for finding in findings if finding.error == error]


# ------------------------------------------------------------ the label mapping


def test_the_clusters_are_named_from_the_words_that_matched():
    """SPEAKER_00 is nobody until the words under it agree with the reference's."""
    findings, profile = run(
        reference((THERAPIST, "so how was your week"),
                  (PARTICIPANT, "it was fine i think")),
        machine(("SPEAKER_00", "so how was your week"),
                ("SPEAKER_01", "it was fine i think")))
    assert profile["role_mapping"] == {"SPEAKER_00": THERAPIST, "SPEAKER_01": PARTICIPANT}
    assert findings == []
    assert profile["word_error_rate"] == 0.0
    assert profile["speaker_error_rate"] == 0.0


def test_a_perfect_transcript_with_the_cluster_numbers_swapped_scores_zero():
    """The numbering is arbitrary. A swap is not an error and must never be scored as one."""
    findings, profile = run(
        reference((THERAPIST, "so how was your week"),
                  (PARTICIPANT, "it was fine i think")),
        machine(("SPEAKER_01", "so how was your week"),
                ("SPEAKER_00", "it was fine i think")))
    assert profile["role_mapping"] == {"SPEAKER_01": THERAPIST, "SPEAKER_00": PARTICIPANT}
    assert findings == []


def test_an_unlabeled_word_votes_for_no_role():
    """The join stamps a speaker only where a span overlaps a diarized turn. A word it
    declined to label must not drag whichever role it sits beside into the mapping."""
    reference_turns = reference((THERAPIST, "so how was your week"),
                                (PARTICIPANT, "it was fine i think"))
    candidate = machine(("SPEAKER_00", "so how was your week"),
                        (UNKNOWN_SPEAKER, "it was fine i think"))
    left, right = word_stream(reference_turns), word_stream(candidate)
    mapping = label_mapping(left, right, align(left, right))
    assert mapping == {"SPEAKER_00": THERAPIST}


def test_words_the_machine_gave_no_speaker_are_counted_apart():
    """"The wrong person" and "no person at all" are different failures with different
    fixes, so the second is a subcount rather than being folded into the first."""
    _, profile = run(
        reference((THERAPIST, "so how was your week"),
                  (PARTICIPANT, "it was fine i think")),
        machine(("SPEAKER_00", "so how was your week"),
                (UNKNOWN_SPEAKER, "it was fine i think")))
    assert profile["misattributed_words"] == 5
    assert profile["unlabeled_matched_words"] == 5
    # UNKNOWN is not a cluster that failed to get a role -- it is the diarizer declining to
    # answer -- so the mapping is complete and the unlabeled subcount is what says so.
    assert profile["role_mapping_complete"] is True


# ----------------------------------------------- what was said: the word labels


def test_a_misheard_word_is_a_substitution_and_changes_no_turn():
    findings, profile = run(
        reference((PARTICIPANT, "it was on thursday i think")),
        machine(("SPEAKER_00", "it was on tuesday i think")))
    assert len(only(findings, SUBSTITUTION)) == 1
    assert only(findings, SUBSTITUTION)[0].structure == WORDS_ONLY
    assert profile["findings_changing_turn_structure"] == 0


def test_a_mangled_name_is_a_proper_noun_not_a_substitution():
    """A wrong word is retyped. A name the model has never seen is wrong every time it
    occurs, which is why the sheet gives it a label of its own."""
    findings, _ = run(
        reference((PARTICIPANT, "i talked to Madison about it")),
        machine(("SPEAKER_00", "i talked to Madeline about it")))
    assert len(only(findings, PROPER_NOUN)) == 1
    assert not only(findings, SUBSTITUTION)


def test_the_capital_that_opens_a_sentence_is_not_a_name():
    """Every turn starts with a capital and so does every sentence. Counting those as names
    would file most substitutions under the wrong label."""
    findings, _ = run(
        reference((PARTICIPANT, "I went out. Then it rained.")),
        machine(("SPEAKER_00", "I went out. Than it rained.")))
    assert len(only(findings, SUBSTITUTION)) == 1
    assert not only(findings, PROPER_NOUN)


def test_a_comma_is_a_punctuation_error_and_a_capital_is_not():
    """Moving a turn boundary re-capitalizes whatever now starts the turn. If case counted,
    one relocated turn would report a dozen punctuation errors."""
    findings, _ = run(
        reference((PARTICIPANT, "it was fine, i think")),
        machine(("SPEAKER_00", "It was fine. i think")))
    punctuation = only(findings, PUNCTUATION)
    assert len(punctuation) == 1
    assert punctuation[0].reference_words == 1
    assert punctuation[0].structure == WORDS_ONLY


def test_the_word_error_rate_counts_each_channel_once():
    """One substitution, one word dropped, one word invented, over ten reference words."""
    _, profile = run(
        reference((PARTICIPANT, "one two three four five six seven eight nine ten")),
        machine(("SPEAKER_00", "one two tree four six seven eight nine ten eleven")))
    assert profile["reference_words"] == 10
    assert profile["words"] == {"matched": 8, "substituted": 1, "omitted": 1, "inserted": 1}
    assert profile["word_error_rate"] == 0.3


# ----------------------------------- the Add Turn? column, computed rather than ticked


def test_a_whole_utterance_the_machine_never_heard_is_a_missed_turn():
    """The annotator's largest bucket: 47 rows where the machine wrote nothing at all."""
    findings, profile = run(
        reference((THERAPIST, "and thats the part we want to build on"),
                  (PARTICIPANT, "mm hmm"),
                  (THERAPIST, "next week")),
        machine(("SPEAKER_00", "and thats the part we want to build on"),
                ("SPEAKER_00", "next week")))
    omissions = only(findings, OMISSION)
    assert len(omissions) == 1
    assert omissions[0].structure == MISSED_TURN
    assert omissions[0].add_turn is True
    assert profile["findings_changing_turn_structure"] == 1


def test_words_dropped_from_inside_a_turn_leave_the_turns_alone():
    """Same label, opposite verdict on the structure: the machine found this turn, it just
    did not write all of it down."""
    findings, profile = run(
        reference((PARTICIPANT, "i went for a walk with my dog and felt better")),
        machine(("SPEAKER_00", "i went for a walk and felt better")))
    omissions = only(findings, OMISSION)
    assert len(omissions) == 1
    assert omissions[0].reference_words == 3
    assert omissions[0].structure == WORDS_ONLY
    assert profile["findings_changing_turn_structure"] == 0


def test_two_people_welded_into_one_turn_is_a_moved_turn():
    """Every word is right and the transcript is still wrong: fixing it cuts the machine's
    turn in two. 27 of the annotator's rows are this."""
    findings, profile = run(
        reference((THERAPIST, "did it help"),
                  (PARTICIPANT, "not really")),
        machine(("SPEAKER_00", "did it help not really")))
    attribution = only(findings, SPEAKER_ATTRIBUTION)
    assert len(attribution) == 1
    assert attribution[0].reference_words == 2
    assert attribution[0].structure == WELDED_TURN
    assert profile["word_error_rate"] == 0.0
    assert profile["speaker_error_rate"] == 2 / 5


def test_a_whole_turn_under_the_wrong_name_moves_nothing():
    """The boundary is right and only the name is wrong, so nothing is added or moved. Five
    of the annotator's 32 attribution rows are these, and they sit in her words-only 42."""
    findings, profile = run(
        reference((THERAPIST, "so how was your week"),
                  (PARTICIPANT, "it was fine i think"),
                  (THERAPIST, "okay")),
        machine(("SPEAKER_00", "so how was your week"),
                ("SPEAKER_01", "it was fine i think"),
                ("SPEAKER_02", "okay")))
    attribution = only(findings, SPEAKER_ATTRIBUTION)
    assert len(attribution) == 1
    assert attribution[0].structure == WORDS_ONLY
    assert attribution[0].add_turn is False
    assert profile["findings_changing_turn_structure"] == 0
    assert profile["role_mapping_complete"] is False


def test_a_turn_the_machine_invented_is_ticked():
    """The annotator's rarest structural row -- one in 117 -- and the only one where the fix
    removes a turn rather than adding one."""
    findings, profile = run(
        reference((THERAPIST, "so how was your week"),
                  (PARTICIPANT, "it was fine i think")),
        machine(("SPEAKER_00", "so how was your week"),
                ("SPEAKER_01", "it was fine i think"),
                ("SPEAKER_00", "thanks for watching")))
    insertions = only(findings, INSERTION)
    assert len(insertions) == 1
    assert insertions[0].structure == INVENTED_TURN
    assert profile["findings_changing_turn_structure"] == 1


def test_words_invented_inside_a_real_turn_are_not_ticked():
    findings, profile = run(
        reference((PARTICIPANT, "it was fine i think")),
        machine(("SPEAKER_00", "it was fine i think you know")))
    insertions = only(findings, INSERTION)
    assert len(insertions) == 1
    assert insertions[0].candidate_words == 2
    assert insertions[0].structure == WORDS_ONLY
    assert profile["findings_changing_turn_structure"] == 0


# --------------------------------------------------------------- the report shape


def test_the_reference_graded_against_itself_finds_nothing():
    """The identity case. It proves only that, which is why every other test above exists --
    but a grader that fails it is reporting its own normalization as error."""
    reference_turns = reference((THERAPIST, "so how was your week"),
                                (PARTICIPANT, "it was fine, i think"),
                                (THERAPIST, "tell me about Tuesday"))
    findings, profile = run(reference_turns, reference_turns)
    assert findings == []
    assert profile["word_error_rate"] == 0.0
    assert profile["speaker_error_rate"] == 0.0
    assert profile["findings"] == 0


def test_every_label_is_present_in_the_report_even_at_zero():
    """A figure drawn from these profiles reads them by key. A label that vanished when it
    happened to be zero would make one arm's chart a different shape from another's."""
    _, profile = run(
        reference((PARTICIPANT, "it was fine i think")),
        machine(("SPEAKER_00", "it was fine i think")))
    assert set(profile["by_error"]) == set(ERROR_LABELS)
    assert all(counts["findings"] == 0 for counts in profile["by_error"].values())


def test_the_two_splits_of_the_findings_add_up():
    """The ticked and unticked counts are the whole population cut once, the way the sheet's
    75/42 is -- not two overlapping tallies."""
    _, profile = run(
        reference((THERAPIST, "did it help"),
                  (PARTICIPANT, "not really"),
                  (THERAPIST, "okay and what about thursday"),
                  (PARTICIPANT, "mm hmm")),
        machine(("SPEAKER_00", "did it help not really"),
                ("SPEAKER_00", "okay and what about tuesday")))
    total = profile["findings_changing_turn_structure"] + profile["findings_changing_words_only"]
    assert total == profile["findings"]
    assert sum(counts["findings"] for counts in profile["by_error"].values()) == profile["findings"]
    assert sum(profile["by_structure"].values()) == profile["findings"]
