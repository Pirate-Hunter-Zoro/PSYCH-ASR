"""The transcript JSON shapes, and the one repair a JSON round trip makes necessary."""

import json

import pytest

from psych_asr.artifacts.transcripts import (
    load_transcript, relink_word_segments, save_transcript, speaker_labels,
    unlabeled_counts, word_label_columns, word_start_time,
)


def test_a_json_round_trip_breaks_the_aliasing(tmp_path, aligned_transcript):
    """THIS IS THE BUG relink_word_segments EXISTS FOR, and its absence is invisible: the
    file looks complete, the segment-level talk-time table is correct, and every word-level
    feature downstream is silently empty."""
    path = save_transcript(aligned_transcript, tmp_path / "s.aligned.json")
    reloaded = load_transcript(path)

    reloaded["segments"][0]["words"][0]["speaker"] = "SPEAKER_00"
    assert "speaker" not in reloaded["word_segments"][0], "fixture no longer reproduces the bug"

    relink_word_segments(reloaded)
    reloaded["segments"][0]["words"][0]["speaker"] = "SPEAKER_01"
    assert reloaded["word_segments"][0]["speaker"] == "SPEAKER_01"


def test_relink_refuses_a_transcript_that_is_not_the_shape_it_assumes(aligned_transcript):
    """A silent overwrite here would produce a file whose word-level speakers are WRONG
    rather than absent, which is far harder to notice."""
    broken = json.loads(json.dumps(aligned_transcript))
    broken["word_segments"].append({"word": "extra"})
    with pytest.raises(SystemExit):
        relink_word_segments(broken)


def test_relink_is_a_no_op_on_an_already_consistent_transcript(aligned_transcript):
    before = [w["word"] for w in aligned_transcript["word_segments"]]
    relink_word_segments(aligned_transcript)
    assert [w["word"] for w in aligned_transcript["word_segments"]] == before


def test_readers_tolerate_a_missing_speaker_key():
    """assign_word_speakers sets it only where a span overlaps a turn, and fill_nearest is
    off, so there is no fallback."""
    transcript = {"segments": [{"speaker": "A"}, {}, {"speaker": "A"}],
                  "word_segments": [{"word": "x", "speaker": "A"}, {"word": "y"}]}
    assert speaker_labels(transcript) == ["A"]
    assert unlabeled_counts(transcript) == (1, 1)


def test_the_word_column_keeps_declined_labels_as_none():
    """WHICH words an arm declined to label is itself one of the differences worth seeing,
    so they must not be skipped."""
    transcript = {"word_segments": [{"word": "a", "speaker": "A"}, {"word": "b"}]}
    assert word_label_columns(transcript) == (["a", "b"], ["A", None])


def test_an_untimed_word_falls_back_to_the_nearest_timed_one():
    transcript = {"word_segments": [{"word": "a", "start": 1.0}, {"word": "b"}, {"word": "c"}]}
    assert word_start_time(transcript, 2) == 1.0


def test_a_transcript_with_no_timings_at_all_returns_none():
    transcript = {"word_segments": [{"word": "a"}, {"word": "b"}]}
    assert word_start_time(transcript, 1) is None


def test_saved_json_keeps_the_layout_the_fixture_was_written_with(tmp_path):
    """indent=4 and ensure_ascii=False, so a re-run compares to the regression fixture byte
    for byte rather than diffing on layout."""
    path = save_transcript({"segments": [], "language": "en", "t": "café"}, tmp_path / "s.json")
    text = path.read_text()
    assert "café" in text
    assert '\n    "' in text
