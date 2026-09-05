"""The cross-arm word-level diff, and the label alignment it rests on."""

import pytest

from psych_asr.evaluate import compare
from psych_asr.evaluate.labels import best_label_mapping, dominant_speaker, map_labels_to_reference


class _Turn:
    def __init__(self, start, end):
        self.start, self.end = start, end


class _Annotation:
    """The smallest object with pyannote's itertracks/labels/label_duration surface, so the
    label logic is testable without a diarization stack."""

    def __init__(self, turns):
        self._turns = turns

    def itertracks(self, yield_label=True):
        for start, end, label in self._turns:
            yield _Turn(start, end), None, label

    def labels(self):
        return sorted({label for _, _, label in self._turns})

    def label_duration(self, label):
        return sum(end - start for start, end, name in self._turns if name == label)


def test_two_arms_that_agree_perfectly_score_perfectly_despite_different_names():
    """Without the remap, SPEAKER_00 here and speaker_1 there would score 0%."""
    reference = ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01", "SPEAKER_01"]
    candidate = ["b", "b", "a", "a"]
    assert best_label_mapping(reference, candidate) == {"b": "SPEAKER_00", "a": "SPEAKER_01"}


def test_a_third_speaker_is_not_folded_onto_one_of_the_baseline_two():
    """An arm that found an extra cluster must show that, not have it absorbed."""
    reference = ["A"] * 4 + ["B"] * 4
    candidate = ["x"] * 4 + ["y"] * 3 + ["z"]
    mapping = best_label_mapping(reference, candidate)
    assert mapping["x"] == "A" and mapping["y"] == "B"
    assert mapping["z"].startswith("EXTRA_")


def test_an_arm_that_labelled_nothing_maps_to_nothing():
    assert best_label_mapping(["A", "B"], [None, None]) == {}


def test_differing_word_sequences_are_refused():
    """Every arm must be joined onto the SAME Stage 1a output, or a label difference cannot
    be attributed to the diarizer at all."""
    with pytest.raises(SystemExit, match="different word sequence"):
        compare.require_identical_words({"a": ["x", "y"], "b": ["x", "z"]}, "a")


def test_the_baseline_leads_and_a_missing_baseline_does_not_stop_the_run():
    assert compare.order_arms(["diarizen", "community-1"])[0] == ["community-1", "diarizen"]
    ordered, reference = compare.order_arms(["sortformer", "diarizen"])
    assert reference == "diarizen" and ordered == ["diarizen", "sortformer"]


def test_words_no_arm_labelled_are_not_a_disagreement():
    """Every arm said the same nothing. Counting it would conflate "who spoke" with "was
    this stretch covered at all," and those call for different fixes."""
    labels = {"a": ["A", None, "A"], "b": ["A", None, "B"]}
    assert compare.disagreement_runs(labels, ["a", "b"], 3) == [(2, 3)]


def test_a_run_that_reaches_the_end_of_the_transcript_is_still_closed():
    labels = {"a": ["A", "A"], "b": ["A", "B"]}
    assert compare.disagreement_runs(labels, ["a", "b"], 2) == [(1, 2)]


def test_agreement_is_computed_only_over_words_both_arms_labelled():
    labels = {"a": ["A", "A", None], "b": ["A", "B", "B"]}
    matrix = compare.pairwise_agreement(labels, ["a", "b"])
    assert matrix["a"]["a"] == 100.0
    assert matrix["a"]["b"] == pytest.approx(50.0)


def test_per_arm_shape_reports_shares_over_labelled_words_only():
    shape = compare.per_arm_shape(["A", "A", "B", None])
    assert shape["distinct_labels"] == 2
    assert shape["unlabeled_words"] == 1
    assert shape["word_share"]["A"] == pytest.approx(2 / 3)


def test_region_records_carry_the_surrounding_exchange():
    """A speaker label is uncodeable without it -- that is the unit a human or an LLM triage
    pass gets handed."""
    words = [f"w{i}" for i in range(20)]
    labels = {"a": ["A"] * 20, "b": ["A"] * 8 + ["B"] * 3 + ["A"] * 9}
    records = compare.build_region_records([(8, 11)], words, labels, ["a", "b"],
                                           start_time_of=lambda i: 12.5, context_words=2)
    record = records[0]
    assert record["disputed_text"] == "w8 w9 w10"
    assert record["context_range"] == [6, 13]
    assert record["start_time"] == 12.5
    assert compare.summarize_region(record, ["a", "b"]) == {"a": {"A": 3}, "b": {"B": 3}}


def test_greedy_annotation_mapping_is_one_to_one():
    reference = _Annotation([(0, 10, "A"), (10, 20, "B")])
    hypothesis = _Annotation([(0, 9, "s1"), (10, 19, "s0"), (19, 20, "s2")])
    mapping = map_labels_to_reference(reference, hypothesis)
    assert mapping["s1"] == "A" and mapping["s0"] == "B"
    assert len(set(mapping.values())) == len(mapping)


def test_dominant_speaker_is_the_same_argmax_the_join_uses():
    annotation = _Annotation([(0, 5, "A"), (4, 10, "B")])
    assert dominant_speaker(annotation, 0, 4.5) == "A"
    assert dominant_speaker(annotation, 5, 10) == "B"
    assert dominant_speaker(annotation, 100, 110) is None
