"""Stage 1c end to end, on a synthetic transcript. Needs whisperx; skipped where absent."""

import pytest

pytest.importorskip("whisperx", reason="the join runs in asr_env")

from psych_asr.artifacts.rttm_io import write_rttm  # noqa: E402
from psych_asr.artifacts.transcripts import load_transcript, save_transcript  # noqa: E402
from psych_asr.cli import join_speakers  # noqa: E402


def _run(tmp_path, aligned_transcript, turn_table, arm="community-1"):
    aligned = save_transcript(aligned_transcript, tmp_path / "S1.aligned.json")
    rttm = tmp_path / f"S1.{arm}.rttm"
    write_rttm(turn_table, "S1", rttm)
    join_speakers.main([str(aligned), str(rttm)])
    return tmp_path / f"S1.{arm}.diarized.json", tmp_path / f"S1.{arm}.transcript.txt"


def test_the_join_writes_both_artifacts_under_the_arm_name(tmp_path, aligned_transcript, turn_table):
    diarized, readable = _run(tmp_path, aligned_transcript, turn_table)
    assert diarized.exists() and readable.exists()


def test_word_level_speakers_actually_land(tmp_path, aligned_transcript, turn_table, capsys):
    """The regression this guards is silent: without relink_word_segments the file looks
    complete and word_segments comes back with ZERO speakers."""
    diarized, _ = _run(tmp_path, aligned_transcript, turn_table)
    joined = load_transcript(diarized)
    labelled = sum(1 for word in joined["word_segments"] if "speaker" in word)
    assert labelled > 0.5 * len(joined["word_segments"])


def test_the_arm_is_taken_from_the_rttm_filename(tmp_path, aligned_transcript, turn_table):
    """Which model produced which transcript is a property of the file. Adding a fifth arm
    needs no change to the join."""
    diarized, readable = _run(tmp_path, aligned_transcript, turn_table, arm="sortformer-streaming")
    assert diarized.name == "S1.sortformer-streaming.diarized.json"
    assert "[sortformer-streaming]" in readable.read_text()


def test_an_empty_turn_table_is_refused_rather_than_joined(tmp_path, aligned_transcript):
    with pytest.raises(SystemExit, match="produced nothing to join"):
        _run(tmp_path, aligned_transcript, [])
