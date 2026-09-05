"""The interchange format between Stage 1b and Stage 1c."""

import pytest

from psych_asr.artifacts.rttm_io import (
    MIN_TURN_DURATION, format_turn_summary, read_rttm, summarize_turns, write_rttm,
)


def test_round_trip_preserves_turns(tmp_path, turn_table):
    path = tmp_path / "s.rttm"
    written = write_rttm(turn_table, "S1", path)
    read_back = read_rttm(path)

    assert written == len(read_back)
    for original, restored in zip(sorted(turn_table), read_back):
        assert restored["start"] == pytest.approx(original[0], abs=1e-3)
        assert restored["end"] == pytest.approx(original[1], abs=1e-3)
        assert restored["speaker"] == original[2]


def test_output_is_order_independent(tmp_path, rng, turn_table):
    """Two arms emitting the same turns in different orders must produce identical bytes,
    or a byte comparison between runs measures iteration order."""
    shuffled = list(turn_table)
    rng.shuffle(shuffled)
    a, b = tmp_path / "a.rttm", tmp_path / "b.rttm"
    write_rttm(turn_table, "S1", a)
    write_rttm(shuffled, "S1", b)
    assert a.read_bytes() == b.read_bytes()


def test_sub_millisecond_turns_are_dropped(tmp_path):
    """A zero-duration turn makes assign_word_speakers' strict `intersection > 0` test
    unsatisfiable, so it is speech to nobody."""
    turns = [(0.0, 1.0, "A"), (2.0, 2.0, "B"), (3.0, 3.0 + MIN_TURN_DURATION / 2, "C")]
    assert write_rttm(turns, "S1", tmp_path / "s.rttm") == 1


def test_whitespace_in_a_label_cannot_shift_the_columns(tmp_path):
    write_rttm([(0.0, 1.0, "spk 2")], "S1", tmp_path / "s.rttm")
    line = (tmp_path / "s.rttm").read_text().split()
    assert len(line) == 10
    assert line[7] == "spk_2"


def test_non_speaker_records_and_comments_are_skipped(tmp_path):
    path = tmp_path / "s.rttm"
    path.write_text(
        "; a comment\n"
        "SPKR-INFO S1 1 <NA> <NA> <NA> unknown A <NA> <NA>\n"
        "SPEAKER S1 1 0.000 2.000 <NA> <NA> A <NA> <NA>\n"
        "\n"
    )
    assert [t["speaker"] for t in read_rttm(path)] == ["A"]


def test_empty_input_writes_an_empty_file(tmp_path):
    path = tmp_path / "s.rttm"
    assert write_rttm([], "S1", path) == 0
    assert path.read_text() == ""
    assert read_rttm(path) == []


def test_overlap_is_double_coverage_of_the_timeline():
    """Overlap survives in the turn table and NOWHERE else -- the joined transcript
    structurally cannot represent two speakers at once."""
    turns = [{"start": 0.0, "end": 10.0, "speaker": "A"},
             {"start": 8.0, "end": 12.0, "speaker": "B"}]
    summary = summarize_turns(turns)
    assert summary["turn_time"] == pytest.approx(14.0)
    assert summary["covered_time"] == pytest.approx(12.0)
    assert summary["overlap_time"] == pytest.approx(2.0)
    assert summary["num_speakers"] == 2
    assert summary["span"] == pytest.approx(12.0)


def test_summary_of_nothing_does_not_divide_by_zero():
    assert summarize_turns([])["overlap_time"] == 0.0
    assert "0.0%" in "\n".join(format_turn_summary("arm", "S1", summarize_turns([])))
