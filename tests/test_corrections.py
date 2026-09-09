"""The Stage 2 error-log reader and the correction pass.

Synthetic throughout: every fixture here is a three-turn conversation written in the test
that uses it, so the suite still runs anywhere and still contains no session content.
"""

import pytest

from psych_asr.artifacts.error_log import (
    OMISSION,
    PARTICIPANT,
    SPEAKER_ATTRIBUTION,
    SUBSTITUTION,
    THERAPIST,
    Correction,
    parse_line,
    parse_timestamp,
    read_error_log,
    split_role_suffix,
)
from psych_asr.transcript.corrections import (
    ACCOUNTED,
    DocumentIndex,
    apply_corrections,
    normalize,
)
from psych_asr.transcript.render import (
    DIALOGUE_LINE,
    render,
    render_with_line_index,
)
from psych_asr.transcript.summary import summarize
from psych_asr.transcript.turns import group_into_turns

HEADER = ",Session ID,Timestamp,Line,Speaker,Error,AI Transcript:,Actual Speech:,Meaning Changed,Severity,Add Turn?,Subract Turn?,Notes,,,,"


def write_log(tmp_path, rows, spacer=True):
    """Build a CSV shaped like the QC export: a blank spacer row, the header, then rows."""
    lines = [",,,,,,,,,,,,,,,,"] if spacer else []
    lines.append(HEADER)
    lines.extend(rows)
    path = tmp_path / "log.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


def row(session="S1", timestamp="0:00:10", line="3", speaker="Therapist", error="Omission",
        ai="None", actual="mm hmm", meaning="No", severity="Minor", add="FALSE",
        subtract="FALSE", notes=""):
    return (f",{session},{timestamp},{line},{speaker},{error},{ai},{actual},"
            f"{meaning},{severity},{add},{subtract},{notes},,,,")


def make_correction(**overrides):
    """A Correction with every field defaulted, so a test names only what it is about."""
    fields = dict(row=2, session="S1", timestamp=10.0, line=None, speaker=THERAPIST,
                  error=SUBSTITUTION, ai_text=None, actual_text=None, ai_role=None,
                  actual_role=None, meaning_changed="No", severity="Minor",
                  add_turn=False, subtract_turn=False, notes=None)
    fields.update(overrides)
    return Correction(**fields)


def conversation():
    """Three turns, and the word "okay" deliberately in two of them."""
    segments = [
        {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00",
         "text": "So how has the week been since we last talked, okay?"},
        {"start": 10.0, "end": 14.0, "speaker": "SPEAKER_01",
         "text": "It was fine I think."},
        {"start": 14.0, "end": 30.0, "speaker": "SPEAKER_00",
         "text": "Okay. Tell me about the activity schedule we made."},
    ]
    return segments, group_into_turns(segments)


def run(turns, corrections):
    segments = [dict(turn) for turn in turns]
    _, _, index = render_with_line_index({"segments": segments}, "fixture")
    return apply_corrections(turns, corrections, index)


# ---------------------------------------------------------------- the reader


def test_the_header_is_found_below_the_spacer_row(tmp_path):
    """The export puts a row of bare commas above the real header."""
    log = read_error_log(write_log(tmp_path, [row()]))
    assert len(log.corrections) == 1
    assert log.header_row == 2


def test_the_table_does_not_end_at_the_first_blank_session_id(tmp_path):
    """The pilot sheet has one interior row with a severity and nothing else. Stopping
    there read 59 of 117 rows and reported a clean run on half the log."""
    rows = [row(line="3"), ",,,,,,,,No,Minor,FALSE,FALSE,,,,,", row(line="5")]
    log = read_error_log(write_log(tmp_path, rows))
    assert len(log.corrections) == 2
    assert log.incomplete_rows == [4]


def test_trailing_padding_rows_are_not_corrections(tmp_path):
    """116 rows of bare commas sit below the data, with FALSE in both flag columns."""
    rows = [row()] + [",,,,,,,,,,FALSE,FALSE,,,,," for _ in range(5)]
    log = read_error_log(write_log(tmp_path, rows))
    assert len(log.corrections) == 1
    assert log.padding_rows == 5


def test_the_word_none_in_a_text_column_means_nothing_was_transcribed(tmp_path):
    """pandas' default NA list eats exactly this value, which is why the reader is csv."""
    log = read_error_log(write_log(tmp_path, [row(ai="None", actual="yeah")]))
    assert log.corrections[0].ai_text is None
    assert log.corrections[0].is_pure_omission
    assert log.corrections[0].actual_text == "yeah"


def test_a_blank_session_id_mid_table_is_a_dropped_fill_down(tmp_path):
    rows = [row(session="S1"), row(session="")]
    log = read_error_log(write_log(tmp_path, rows))
    assert [c.session for c in log.corrections] == ["S1", "S1"]


def test_a_trailing_role_word_is_split_off_both_text_columns(tmp_path):
    """An attribution row is "<utterance> <Role>" in both cells. Matched raw, none of the
    pilot session's 32 attribution rows appear anywhere in the transcript."""
    log = read_error_log(write_log(tmp_path, [
        row(error="Speaker Attribution", ai="mm hmm Therapist",
            actual="mm hmm Participant", add="TRUE"),
    ]))
    correction = log.corrections[0]
    assert correction.ai_text == "mm hmm"
    assert correction.actual_text == "mm hmm"
    assert correction.ai_role == THERAPIST
    assert correction.actual_role == PARTICIPANT


def test_a_cell_that_is_only_a_role_leaves_no_utterance():
    assert split_role_suffix("Participant") == (None, PARTICIPANT)


def test_a_cell_ending_in_an_ordinary_word_keeps_all_of_it():
    assert split_role_suffix("the activity schedule") == ("the activity schedule", None)


def test_an_unknown_error_value_stops_the_run(tmp_path):
    """A seventh category is work this pass has not been taught, not a row to skip."""
    with pytest.raises(SystemExit):
        read_error_log(write_log(tmp_path, [row(error="Hallucination")]))


def test_timestamps_are_read_at_either_length():
    assert parse_timestamp("0:01:19") == 79.0
    assert parse_timestamp("01:19") == 79.0
    assert parse_timestamp("") is None


def test_a_spreadsheet_writes_the_line_number_as_a_float():
    assert parse_line("48.0") == 48
    assert parse_line("48") == 48
    assert parse_line("") is None


# ------------------------------------------------------- the render line index


def test_the_line_index_does_not_change_the_rendered_text():
    segments, _ = conversation()
    text, _ = render({"segments": segments}, "fixture")
    indexed, _, _ = render_with_line_index({"segments": segments}, "fixture")
    assert text == indexed


def test_a_dialogue_line_span_points_at_that_line_s_own_words():
    segments, turns = conversation()
    _, _, index = render_with_line_index({"segments": segments}, "fixture")
    for entry in index:
        if entry["kind"] != DIALOGUE_LINE:
            continue
        span = turns[entry["turn"]]["text"][entry["start"]:entry["end"]]
        assert span and not span.startswith(" ") and not span.endswith(" ")


def test_normalization_ignores_case_and_punctuation():
    assert normalize("It was FINE, I think.") == "it was fine i think"


def test_the_document_index_reports_a_match_that_crosses_a_turn_boundary():
    _, turns = conversation()
    document = DocumentIndex(turns)
    crossing = document.find(normalize("i think okay"))
    assert crossing and crossing[0][3] is True


# ------------------------------------------------------------- applying edits


def test_a_substitution_changes_only_the_occurrence_at_the_logged_line():
    """The whole reason this is not a str.replace loop: "okay" is in two turns."""
    segments, turns = conversation()
    _, _, index = render_with_line_index({"segments": segments}, "fixture")
    line = next(entry["line"] for entry in index
                if entry["kind"] == DIALOGUE_LINE and entry["turn"] == 2)
    corrected, report = apply_corrections(turns, [make_correction(
        row=2, line=line, error=SUBSTITUTION, ai_text="Okay", actual_text="All right",
    )], index)
    assert corrected[2]["text"].startswith("All right")
    assert "okay" in corrected[0]["text"].lower()
    assert report["by_error"][SUBSTITUTION]["applied"] == 1


def test_a_missed_utterance_becomes_a_new_turn_that_splits_its_host():
    segments, turns = conversation()
    _, _, index = render_with_line_index({"segments": segments}, "fixture")
    line = next(entry["line"] for entry in index
                if entry["kind"] == DIALOGUE_LINE and entry["turn"] == 0)
    corrected, report = apply_corrections(turns, [make_correction(
        row=2, line=line, error=OMISSION, ai_text=None, actual_text="mm hmm",
        speaker=PARTICIPANT, add_turn=True,
    )], index)
    assert len(corrected) > len(turns)
    inserted = [turn for turn in corrected if turn["origin"] == "inserted"]
    assert len(inserted) == 1
    assert inserted[0]["text"] == "mm hmm"
    assert report["edits_by_kind"]["insert"] == 1


def test_a_missed_utterance_by_the_same_speaker_goes_back_into_the_turn():
    """Add Turn? FALSE means the diarizer had the speaker right; the words belong inside."""
    segments, turns = conversation()
    _, _, index = render_with_line_index({"segments": segments}, "fixture")
    line = next(entry["line"] for entry in index
                if entry["kind"] == DIALOGUE_LINE and entry["turn"] == 1)
    corrected, _ = apply_corrections(turns, [make_correction(
        row=2, line=line, error=OMISSION, ai_text=None, actual_text="really",
        speaker=PARTICIPANT, add_turn=False,
    )], index)
    assert len(corrected) == len(turns)
    assert "really" in corrected[1]["text"]


def test_misattributed_words_leave_their_host_rather_than_being_copied():
    """Insert-without-remove is how a naive pass doubles the words it reattributes."""
    segments, turns = conversation()
    _, _, index = render_with_line_index({"segments": segments}, "fixture")
    line = next(entry["line"] for entry in index
                if entry["kind"] == DIALOGUE_LINE and entry["turn"] == 2)
    corrected, report = apply_corrections(turns, [make_correction(
        row=2, line=line, error=SPEAKER_ATTRIBUTION, ai_text="Okay",
        actual_text="Okay", ai_role=THERAPIST, actual_role=PARTICIPANT,
        speaker=PARTICIPANT, add_turn=True,
    )], index)
    assert report["edits_by_kind"]["extract"] == 1
    words = normalize(" ".join(turn["text"] for turn in corrected)).split()
    assert words.count("okay") == 2      # the one in turn 0 and the one that moved
    moved = next(turn for turn in corrected if turn["origin"] == "reattributed")
    assert moved["speaker"] == PARTICIPANT


def test_a_whole_turn_filed_under_the_wrong_person_is_relabelled():
    segments, turns = conversation()
    _, _, index = render_with_line_index({"segments": segments}, "fixture")
    line = next(entry["line"] for entry in index
                if entry["kind"] == DIALOGUE_LINE and entry["turn"] == 1)
    corrected, report = apply_corrections(turns, [make_correction(
        row=2, line=line, error=SPEAKER_ATTRIBUTION, ai_text="It was fine",
        actual_text="It was fine", ai_role=PARTICIPANT, actual_role=THERAPIST,
        speaker=THERAPIST, add_turn=False,
    )], index)
    assert report["edits_by_kind"]["relabel"] == 1
    # SPEAKER_01 is voted PARTICIPANT by the machine's own label, SPEAKER_00 gets
    # THERAPIST by elimination, and the relabelled middle turn welds the three into one.
    assert report["role_mapping"] == {"SPEAKER_01": PARTICIPANT, "SPEAKER_00": THERAPIST}
    assert len(corrected) == 1
    assert corrected[0]["speaker"] == THERAPIST


def test_two_rows_about_the_same_words_do_not_both_fire():
    """A misattribution is logged twice -- as the attribution and as the insertion that
    removes the words from where they were. Both firing deletes them twice."""
    segments, turns = conversation()
    _, _, index = render_with_line_index({"segments": segments}, "fixture")
    line = next(entry["line"] for entry in index
                if entry["kind"] == DIALOGUE_LINE and entry["turn"] == 2)
    corrections = [
        make_correction(row=2, line=line, error=SPEAKER_ATTRIBUTION, ai_text="Okay",
                        actual_text="Okay", speaker=PARTICIPANT, add_turn=True),
        make_correction(row=3, line=line, error="Insertion", ai_text="Okay",
                        actual_text=None, speaker=THERAPIST),
    ]
    _, report = apply_corrections(turns, corrections, index)
    assert report["accounted_rows"] == [3]
    assert report["by_error"]["Insertion"][ACCOUNTED] == 1


def test_the_machine_s_own_recorded_label_maps_the_clusters_to_roles():
    """The AI Transcript cell names the role the DIARIZER gave those words to."""
    segments, turns = conversation()
    _, _, index = render_with_line_index({"segments": segments}, "fixture")
    lines = {turn: next(entry["line"] for entry in index
                        if entry["kind"] == DIALOGUE_LINE and entry["turn"] == turn)
             for turn in (0, 1)}
    corrections = [
        make_correction(row=2, line=lines[0], error=SPEAKER_ATTRIBUTION, ai_text="okay",
                        actual_text="okay", ai_role=THERAPIST, actual_role=PARTICIPANT,
                        speaker=PARTICIPANT, add_turn=True),
        make_correction(row=3, line=lines[1], error=SPEAKER_ATTRIBUTION,
                        ai_text="It was fine", actual_text="It was fine",
                        ai_role=PARTICIPANT, actual_role=THERAPIST,
                        speaker=THERAPIST, add_turn=True),
    ]
    _, report = apply_corrections(turns, corrections, index)
    assert report["role_mapping"] == {"SPEAKER_00": THERAPIST, "SPEAKER_01": PARTICIPANT}


def test_a_row_with_add_turn_and_no_role_of_its_own_never_votes():
    """Those are utterances the machine missed entirely, so the turn they land in is
    usually the OTHER person's. Counting them as agreement took the pilot session's vote
    from 49-3 to 20-17."""
    segments, turns = conversation()
    _, _, index = render_with_line_index({"segments": segments}, "fixture")
    line = next(entry["line"] for entry in index
                if entry["kind"] == DIALOGUE_LINE and entry["turn"] == 0)
    _, report = apply_corrections(turns, [make_correction(
        row=2, line=line, error=OMISSION, ai_text=None, actual_text="mm hmm",
        speaker=PARTICIPANT, add_turn=True,
    )], index)
    assert report["role_vote"] == {}


def test_corrected_turns_tile_their_host_without_overlapping():
    """Writing the annotator's observed time into an inserted turn's start put turns out
    of order and made the talk-time table sum to 107% of the session."""
    segments, turns = conversation()
    _, _, index = render_with_line_index({"segments": segments}, "fixture")
    line = next(entry["line"] for entry in index
                if entry["kind"] == DIALOGUE_LINE and entry["turn"] == 0)
    corrected, _ = apply_corrections(turns, [make_correction(
        row=2, line=line, error=OMISSION, ai_text=None, actual_text="mm hmm",
        speaker=PARTICIPANT, add_turn=True, timestamp=9999.0,
    )], index)
    for earlier, later in zip(corrected, corrected[1:]):
        assert earlier["end"] <= later["start"] + 1e-9
    summary = summarize(corrected, corrected)
    assert summary["speech_time"] <= summary["span"] + 1e-9


def test_the_annotator_s_observed_time_is_kept_beside_the_interpolated_span():
    segments, turns = conversation()
    _, _, index = render_with_line_index({"segments": segments}, "fixture")
    line = next(entry["line"] for entry in index
                if entry["kind"] == DIALOGUE_LINE and entry["turn"] == 0)
    corrected, _ = apply_corrections(turns, [make_correction(
        row=2, line=line, error=OMISSION, ai_text=None, actual_text="mm hmm",
        speaker=PARTICIPANT, add_turn=True, timestamp=7.5,
    )], index)
    inserted = next(turn for turn in corrected if turn["origin"] == "inserted")
    assert inserted["logged_at"] == 7.5
    assert inserted["time_source"] == "interpolated"


def test_a_row_whose_words_are_nowhere_is_reported_not_guessed():
    segments, turns = conversation()
    _, _, index = render_with_line_index({"segments": segments}, "fixture")
    _, report = apply_corrections(turns, [make_correction(
        row=2, line=None, timestamp=None, error=SUBSTITUTION,
        ai_text="words that are not in this conversation at all",
        actual_text="something else",
    )], index)
    assert report["unplaced_rows"] == [2]
    assert report["words_before"] == report["words_after"]
