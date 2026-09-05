"""Turn grouping, the talk-time table, and the readable render."""

import pytest

from psych_asr.transcript.render import render
from psych_asr.transcript.summary import format_summary, format_timestamp, summarize
from psych_asr.transcript.turns import UNKNOWN_SPEAKER, group_into_turns


def test_minutes_are_never_rolled_into_hours():
    """One mm:ss scale matches what a media player's position readout shows."""
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(59.6) == "01:00"
    assert format_timestamp(3012.4) == "50:12"
    assert format_timestamp(3723) == "62:03"


def test_consecutive_same_speaker_segments_become_one_turn():
    """Alignment re-splits at sentence boundaries, so one uninterrupted minute arrives as a
    dozen segments. Ungrouped, the file reads as a list of sentences."""
    segments = [
        {"start": 0, "end": 2, "text": "So how has the week been?", "speaker": "A"},
        {"start": 2, "end": 4, "text": "Since we last talked.", "speaker": "A"},
        {"start": 4, "end": 5, "text": "Fine.", "speaker": "B"},
    ]
    turns = group_into_turns(segments)
    assert [t["speaker"] for t in turns] == ["A", "B"]
    assert turns[0]["text"] == "So how has the week been? Since we last talked."
    assert turns[0]["end"] == 4


def test_a_segment_with_no_speaker_is_labelled_not_dropped():
    """A cluster of UNKNOWN blocks is itself the diagnostic that diarization under-covered
    the audio. Hiding them would hide that."""
    segments = [{"start": 0, "end": 1, "text": "Hello.", "speaker": "A"},
                {"start": 1, "end": 2, "text": "Mm."}]
    assert [t["speaker"] for t in group_into_turns(segments)] == ["A", UNKNOWN_SPEAKER]


def test_the_backchannel_scan_can_keep_absent_as_absent():
    """group_into_turns(unknown=None) must not weld two unlabeled stretches together on the
    strength of a placeholder they never carried."""
    segments = [{"start": 0, "end": 1, "text": "Mm."},
                {"start": 1, "end": 2, "text": "Okay."}]
    assert [t["speaker"] for t in group_into_turns(segments, unknown=None)] == [None]
    assert len(group_into_turns(segments)) == 1


def test_empty_text_never_breaks_a_turn_in_half():
    segments = [{"start": 0, "end": 1, "text": "One.", "speaker": "A"},
                {"start": 1, "end": 2, "text": "   ", "speaker": "B"},
                {"start": 2, "end": 3, "text": "Two.", "speaker": "A"}]
    assert len(group_into_turns(segments)) == 1


def test_talk_time_share_denominator_is_speech_not_wall_clock():
    """Silence must not dilute the split between the two people."""
    segments = [{"start": 0, "end": 30, "text": "x", "speaker": "A"},
                {"start": 100, "end": 110, "text": "y", "speaker": "B"}]
    summary = summarize(segments, group_into_turns(segments))
    assert summary["span"] == 110
    assert summary["speech_time"] == 40
    assert summary["per_speaker"]["A"]["share"] == pytest.approx(75.0)
    assert summary["per_speaker"]["B"]["share"] == pytest.approx(25.0)


def test_unknown_sorts_last_however_loud_it_is():
    """It is a diagnostic, not a person."""
    segments = [{"start": 0, "end": 100, "text": "x"},
                {"start": 100, "end": 110, "text": "y", "speaker": "A"}]
    lines = format_summary("S1", summarize(segments, group_into_turns(segments)))
    # The table starts after the column header; everything before it is the run summary.
    table = lines[lines.index(next(l for l in lines if l.startswith("Speaker "))) + 1:-1]
    assert [line.split()[0] for line in table] == ["A", UNKNOWN_SPEAKER]


def test_a_collapsed_clustering_is_visible_in_the_header():
    """Two people in a room do not split 97/3. That number falsifies the run from the log
    alone, before anyone opens the audio."""
    segments = [{"start": 0, "end": 97, "text": "x", "speaker": "A"},
                {"start": 97, "end": 100, "text": "y", "speaker": "B"}]
    header = "\n".join(format_summary("S1", summarize(segments, group_into_turns(segments))))
    assert "97.0%" in header and "3.0%" in header


def test_render_emits_one_timestamped_block_per_turn(aligned_transcript):
    for segment in aligned_transcript["segments"]:
        segment["speaker"] = "SPEAKER_00"
    text, summary = render(aligned_transcript, "S1")
    assert text.startswith("=" * 72)
    assert text.endswith("\n")
    assert text.count("] SPEAKER_00") == summary["num_turns"] == 1
