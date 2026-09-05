"""Splice a long recording's per-window diarizations into one consistent timeline.

numpy at module scope, scipy inside the one function that solves an assignment. No torch,
no NeMo -- so this file is unit-testable from any env, which matters more here than
anywhere else in the package: it is the only machinery in Stage 1b that can be WRONG in a
way no model is responsible for.

THIS IS MACHINERY ONLY ARM B NEEDS. NVIDIA's model card puts the offline Sortformer's
ceiling near 12 minutes on a 48 GB RTX A6000; the node's A40 has 46 GB and a pilot session
is ~50 minutes. Arm C handles arbitrary length by construction and never comes through
here. When the arms are scored, that asymmetry belongs in the writeup: SOME OF ARM B'S
ERROR WILL BE SEAM ERROR RATHER THAN MODEL ERROR.
"""

import numpy as np

# The model emits one activity decision per 0.08 s of audio; matching windows on that same
# grid means the stitcher never invents a resolution the model does not have.
FRAME_SECONDS = 0.08

# Hard four-speaker ceiling, and the number of global label slots the stitcher carries.
MAX_SPEAKERS = 4


def parse_segments(raw_segments, time_offset=0.0):
    """Normalize NeMo's diarize() output into (start, end, local_speaker) tuples.

    IN:  raw_segments -- one audio file's entry from diarize(), whose elements are either
         "start end speaker" strings or (start, end, speaker) sequences depending on NeMo
         version; time_offset -- seconds to add, for a window cut out of a longer file
    OUT: list of (float, float, str)

    Both shapes are accepted rather than pinned to one, because the two model cards
    document the return value only as "begin_seconds, end_seconds, speaker_index".
    """
    turns = []
    for item in raw_segments:
        fields = item.split() if isinstance(item, str) else list(item)
        start, end, speaker = float(fields[0]), float(fields[1]), str(fields[2])
        turns.append((start + time_offset, end + time_offset, speaker))
    return turns


def activity_grid(turns, labels, grid_start, grid_end):
    """Rasterize turns onto the model's own 80 ms frame grid.

    IN:  turns  -- (start, end, label) tuples in absolute seconds
         labels -- the label ordering that becomes the rows
         grid_start / grid_end -- the window to rasterize, in seconds
    OUT: bool array of shape (len(labels), num_frames)

    Used only to compare two windows on the stretch they share.
    """
    num_frames = max(int(round((grid_end - grid_start) / FRAME_SECONDS)), 0)
    grid = np.zeros((len(labels), num_frames), dtype=bool)
    index_of = {label: i for i, label in enumerate(labels)}
    for start, end, label in turns:
        if label not in index_of:
            continue
        first = int(np.floor((max(start, grid_start) - grid_start) / FRAME_SECONDS))
        last = int(np.ceil((min(end, grid_end) - grid_start) / FRAME_SECONDS))
        if last > first:
            grid[index_of[label], max(first, 0):min(last, num_frames)] = True
    return grid


def match_labels(previous_turns, window_turns, overlap_start, overlap_end):
    """Decide which of the new window's local speakers is which of the running speakers.

    IN:  previous_turns -- turns accepted so far, carrying GLOBAL labels
         window_turns   -- the new window's turns, carrying LOCAL labels
         overlap_start / overlap_end -- the stretch both windows saw, in seconds
    OUT: dict mapping local label -> global label

    A window's speaker indices are arbitrary and independent of every other window's, so
    the stitch has to be made on evidence rather than on index. The evidence is the overlap
    region: whichever local speaker's activity best coincides with a global speaker's
    activity there is the same person. The pairing is solved as a LINEAR ASSIGNMENT over
    frame agreement, so it is one-to-one -- two local speakers can never collapse onto the
    same global label, which is the failure that would silently merge two people.

    A local speaker with no overlap evidence at all gets a free global slot instead of a
    guess; that is what keeps a speaker who only appears late in the session from being
    grafted onto an existing label.
    """
    from scipy.optimize import linear_sum_assignment

    global_labels = [f"speaker_{i}" for i in range(MAX_SPEAKERS)]
    local_labels = sorted({label for _, _, label in window_turns})
    if not local_labels:
        return {}

    previous_grid = activity_grid(previous_turns, global_labels, overlap_start, overlap_end)
    window_grid = activity_grid(window_turns, local_labels, overlap_start, overlap_end)

    # Agreement counted as frames where both are active; negated because the solver minimizes.
    agreement = window_grid.astype(np.int32) @ previous_grid.astype(np.int32).T
    rows, columns = linear_sum_assignment(-agreement)

    mapping = {}
    taken = set()
    for row, column in zip(rows, columns):
        if agreement[row, column] > 0:
            mapping[local_labels[row]] = global_labels[column]
            taken.add(global_labels[column])
    for label in local_labels:
        if label not in mapping:
            free = [g for g in global_labels if g not in taken]
            if not free:
                break
            mapping[label] = free[0]
            taken.add(free[0])
    return mapping


def merge_adjacent(turns, tolerance=0.01):
    """Rejoin same-speaker turns that the seam cut in half.

    IN:  turns -- (start, end, label) tuples; tolerance -- the largest gap still treated
         as "touching", in seconds
    OUT: the same turns with touching or overlapping same-speaker pairs merged, sorted

    A turn spanning a window boundary is truncated by one window and re-emitted by the
    next, so without this it lands in the RTTM as two consecutive turns by the same
    speaker. That is a pure artifact of the stitcher, and it would inflate arm B's turn
    count and its between-turn latency against arms that never got windowed. The tolerance
    is deliberately far below any real pause in speech, so nothing but a seam is joined.
    """
    by_label = {}
    for start, end, label in turns:
        by_label.setdefault(label, []).append((start, end))

    merged = []
    # Grouped by speaker first, so a seam split is still rejoined when the OTHER speaker
    # happens to be talking across it. Sorting the mixed list instead would put that
    # speaker's turn between the two halves and block the merge.
    for label, spans in by_label.items():
        current = None
        for start, end in sorted(spans):
            if current is not None and start - current[1] <= tolerance:
                current[1] = max(current[1], end)
            else:
                if current is not None:
                    merged.append((current[0], current[1], label))
                current = [start, end]
        if current is not None:
            merged.append((current[0], current[1], label))
    return sorted(merged)


def stitch_windows(window_results, overlap):
    """Splice per-window diarizations into one timeline with consistent speaker labels.

    IN:  window_results -- list of (window_start_seconds, turns) in window order, each
         turns list carrying that window's own local labels in ABSOLUTE seconds
         overlap -- how much consecutive windows share, in seconds. The stride is not
         needed: every cut point is derived from the NEXT window's own start time, so a
         window list with an irregular stride still splices correctly.
    OUT: list of (start, end, global_label) tuples covering the whole recording

    Each window contributes the timeline up to the MIDPOINT of its overlap with the next
    one. Cutting at the midpoint rather than at a window edge keeps every accepted second
    away from the region where a model has the least context -- the first and last moments
    of its input, where an end-to-end diarizer is at its worst.
    """
    accepted = []
    # The PREVIOUS window's full mapped turns, not `accepted`. Accepted turns have already
    # been truncated at the overlap midpoint, so matching against them would throw away
    # half the evidence the overlap exists to provide.
    previous_mapped = []
    for index, (window_start, turns) in enumerate(window_results):
        if index == 0:
            # Normalize the first window's local labels onto the global slot names too, so
            # every label in the output comes from one namespace regardless of what the
            # model happened to call its speakers.
            first_labels = sorted({label for _, _, label in turns})
            first_mapping = {label: f"speaker_{i}" for i, label in enumerate(first_labels)}
            mapped = [(s, e, first_mapping[label]) for s, e, label in turns]
        else:
            mapping = match_labels(previous_mapped, turns, window_start, window_start + overlap)
            mapped = [(s, e, mapping.get(label, label)) for s, e, label in turns]
        previous_mapped = mapped

        if index + 1 < len(window_results):
            cut = window_results[index + 1][0] + overlap / 2.0
        else:
            cut = float("inf")
        keep_from = (window_start + overlap / 2.0) if index > 0 else 0.0

        accepted = [(s, min(e, keep_from), label) for s, e, label in accepted if s < keep_from]
        accepted.extend(
            (max(s, keep_from), min(e, cut), label)
            for s, e, label in mapped
            if e > keep_from and s < cut
        )
    return merge_adjacent(t for t in accepted if t[1] > t[0])


def window_starts(duration, window, overlap):
    """IN: recording duration, window length, overlap -- all seconds   OUT: list of starts.

    Stops once a window would begin inside the tail already covered, so the last window
    reaches the end of the file rather than a stub being cut for the final few seconds.
    """
    stride = window - overlap
    return list(np.arange(0.0, max(duration - overlap, 0.0), stride))
