"""The gate that decides whether the 1a/1b/1c split changed behaviour."""

import copy

from psych_asr.evaluate.regression import compare_fields, compare_structures, structure_of


def _fixture():
    words = [{"word": "a", "speaker": "SPEAKER_00"}, {"word": "b", "speaker": "SPEAKER_00"},
             {"word": "c", "speaker": "SPEAKER_01"}, {"word": "d"}]
    return {
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "a b", "speaker": "SPEAKER_00", "words": words[:2]},
            {"start": 2.0, "end": 3.0, "text": "c", "speaker": "SPEAKER_01", "words": words[2:3]},
            {"start": 3.0, "end": 4.0, "text": "d", "words": words[3:]},
        ],
        "word_segments": words,
        "language": "en",
    }


def test_structure_records_what_the_fixture_is_recorded_as():
    structure = structure_of(_fixture())
    assert structure["segments"] == 3
    assert structure["words"] == 4
    assert structure["turns"] == 3
    assert structure["speaker_labels"] == ["SPEAKER_00", "SPEAKER_01"]
    assert structure["unknown_segments"] == 1
    assert structure["unlabeled_words"] == 1
    assert structure["talk_time_shares"] == {"SPEAKER_00": 50.0, "SPEAKER_01": 25.0, "UNKNOWN": 25.0}


def test_an_identical_run_matches_on_both_levels():
    fixture = _fixture()
    matched, rows = compare_structures(structure_of(fixture), structure_of(copy.deepcopy(fixture)))
    assert matched and all(row[3] for row in rows)
    assert compare_fields(fixture, copy.deepcopy(fixture)) == []


def test_a_swapped_speaker_fails_the_structure_check():
    """A mismatch here means the refactor changed behaviour. It is a bug, not a finding."""
    mutant = _fixture()
    mutant["segments"][1]["speaker"] = "SPEAKER_02"
    matched, rows = compare_structures(structure_of(_fixture()), structure_of(mutant))
    assert not matched
    assert dict((row[0], row[3]) for row in rows)["speaker_labels"] is False


def test_decode_noise_shows_up_only_at_the_field_level():
    """Whisper decodes in float16 on a GPU and is not bit-reproducible, so a scatter of
    differing characters with MATCHING structure is hardware, not the split."""
    mutant = _fixture()
    mutant["segments"][0]["text"] = "a b "
    matched, _ = compare_structures(structure_of(_fixture()), structure_of(mutant))
    assert matched
    differences = compare_fields(_fixture(), mutant)
    assert len(differences) == 1 and differences[0].startswith("segment[0].text")


def test_the_difference_list_is_capped_so_a_log_is_not_flooded_with_phi():
    fixture = _fixture()
    mutant = copy.deepcopy(fixture)
    for segment in mutant["segments"]:
        segment["text"] = "different"
        segment["start"] = -1
    assert len(compare_fields(fixture, mutant, limit=2)) == 2
