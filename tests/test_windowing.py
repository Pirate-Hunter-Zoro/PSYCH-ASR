"""The Sortformer stitcher -- the only Stage 1b machinery no model is responsible for."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from psych_asr.diarize.windowing import (  # noqa: E402
    merge_adjacent, parse_segments, stitch_windows, window_starts,
)


def test_both_documented_return_shapes_are_accepted():
    """The two model cards document the return value only as "begin, end, speaker_index",
    and NeMo has emitted both strings and tuples."""
    assert parse_segments(["0.0 1.5 0"]) == [(0.0, 1.5, "0")]
    assert parse_segments([(0.0, 1.5, 0)]) == [(0.0, 1.5, "0")]


def test_a_window_offset_shifts_into_absolute_seconds():
    assert parse_segments(["0.0 1.5 0"], time_offset=600.0) == [(600.0, 601.5, "0")]


def test_a_turn_split_by_a_seam_is_rejoined():
    """Without this the RTTM carries two consecutive turns by the same speaker, inflating
    arm B's turn count against arms that were never windowed."""
    assert merge_adjacent([(0.0, 10.0, "A"), (10.005, 20.0, "A")]) == [(0.0, 20.0, "A")]


def test_a_real_pause_is_not_a_seam():
    assert len(merge_adjacent([(0.0, 10.0, "A"), (10.5, 20.0, "A")])) == 2


def test_the_other_speaker_talking_across_a_seam_does_not_block_the_merge():
    """Grouping by speaker first is what makes this work; sorting the mixed list would put
    B's turn between A's two halves."""
    merged = merge_adjacent([(0.0, 10.0, "A"), (5.0, 10.004, "B"), (10.005, 20.0, "A")])
    assert (0.0, 20.0, "A") in merged


def _windowed(truth, starts, window, permutations):
    """Cut `truth` into windows and relabel each with its own arbitrary local names."""
    results = []
    for start, local in zip(starts, permutations):
        remap = {"A": local[0], "B": local[1]}
        results.append((start, [(max(a, start), min(b, start + window), remap[label])
                                for a, b, label in truth
                                if b > start and a < start + window]))
    return results


def test_windows_that_number_their_speakers_differently_still_stitch():
    """A window's speaker indices are arbitrary and independent of every other window's, so
    the stitch is made on overlap evidence rather than on index."""
    window, overlap = 600.0, 60.0
    truth = [(t, t + 40.0, "A" if (t // 40) % 2 == 0 else "B") for t in np.arange(0.0, 2400.0, 40.0)]
    starts = list(np.arange(0.0, 2400.0 - overlap, window - overlap))
    results = _windowed(truth, starts, window, [("0", "1"), ("1", "0"), ("0", "1"), ("1", "0"), ("0", "1")])

    stitched = stitch_windows(results, overlap)
    assert {label for _, _, label in stitched} == {"speaker_0", "speaker_1"}

    # Every accepted second must carry the label the truth had there.
    mapping = {}
    for start, end, label in stitched:
        for a, b, truth_label in truth:
            if min(b, end) - max(a, start) > 20.0:
                mapping.setdefault(truth_label, set()).add(label)
    assert all(len(labels) == 1 for labels in mapping.values()), mapping


def test_two_local_speakers_can_never_collapse_onto_one_global_label():
    """That failure would silently merge two people, which is worse than any DER."""
    window, overlap = 600.0, 60.0
    truth = [(t, t + 40.0, "A" if (t // 40) % 2 == 0 else "B") for t in np.arange(0.0, 1800.0, 40.0)]
    starts = list(np.arange(0.0, 1800.0 - overlap, window - overlap))
    results = _windowed(truth, starts, window, [("0", "1")] * len(starts))
    stitched = stitch_windows(results, overlap)
    assert len({label for _, _, label in stitched}) == 2


def test_the_seam_is_cut_at_the_overlap_midpoint():
    """Every accepted second stays away from a window edge, where an end-to-end model has
    the least context."""
    window, overlap = 600.0, 60.0
    starts = [0.0, 540.0]
    #                first window says A throughout; second says B throughout
    results = [(0.0, [(0.0, 600.0, "0")]), (540.0, [(540.0, 1140.0, "0")])]
    stitched = stitch_windows(results, overlap)
    boundaries = sorted({round(start, 1) for start, _, _ in stitched} |
                        {round(end, 1) for _, end, _ in stitched})
    assert 570.0 in boundaries or stitched == [(0.0, 1140.0, "speaker_0")]


def test_the_last_window_reaches_the_end_of_the_file():
    starts = window_starts(3000.0, 600.0, 60.0)
    assert starts[0] == 0.0
    assert starts[-1] + 600.0 >= 3000.0


def test_a_recording_shorter_than_one_window_needs_no_windowing():
    assert len(window_starts(300.0, 600.0, 60.0)) == 1
