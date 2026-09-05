"""Regression gate for the 1a/1b/1c split: does composing the three steps reproduce the
single-job result exactly?

STDLIB ONLY.

The split must be provably behavior-preserving BEFORE any challenger diarizer is added.
Otherwise a difference between arms cannot be attributed -- it might be the model, or it
might be the refactor, and there is no way to tell afterwards.

The fixture is job 2032471 (2026-08-12, full ~50 min session), whose four-pass single-job
Stage 1 produced 489 segments, 7298 words, exactly 2 speaker labels, 89 turns, one UNKNOWN
segment, 79/21 talk time. Composing 1a -> 1b(community-1) -> 1c must reproduce those.

TWO LEVELS OF COMPARISON, reported separately because they fail for different reasons:

  STRUCTURE -- segment count, word count, turn count, speaker labels, unlabeled counts,
  talk-time shares. These are what the fixture is recorded as, and a mismatch here means
  the refactor changed behavior. This one is a hard failure.

  EXACT FIELDS -- per-segment start/end/text/speaker and per-word speaker. A mismatch here
  WITH matching structure is worth reading before it is worth worrying about: Whisper
  decodes in float16 on a GPU and is not bit-reproducible across runs, so a handful of
  differing characters is a property of the hardware, not of the split. A systematic
  divergence is not.
"""

from ..transcript.summary import summarize
from ..transcript.turns import group_into_turns

# How many individual field differences to collect before stopping. Enough to see whether a
# divergence is a scatter of decode noise or a systematic shift; not enough to dump PHI
# across a whole log.
MAX_REPORTED_DIFFERENCES = 10


def structure_of(transcript):
    """IN: a .diarized.json dict   OUT: dict of the numbers the fixture is recorded as."""
    segments = transcript.get("segments", [])
    words = transcript.get("word_segments", [])
    summary = summarize(segments, group_into_turns(segments))
    return {
        "segments": len(segments),
        "words": len(words),
        "turns": summary["num_turns"],
        "speaker_labels": sorted({s.get("speaker") for s in segments} - {None}),
        "unknown_segments": sum(1 for s in segments if "speaker" not in s),
        "unlabeled_words": sum(1 for w in words if "speaker" not in w),
        "talk_time_shares": {
            speaker: round(entry["share"], 1)
            for speaker, entry in sorted(summary["per_speaker"].items())
        },
    }


def compare_fields(reference, candidate, limit=MAX_REPORTED_DIFFERENCES):
    """IN: two .diarized.json dicts   OUT: list of human-readable difference strings.

    Compared POSITIONALLY, which is only meaningful once the counts already match -- so
    call this after structure_of has agreed, never instead of it.
    """
    differences = []

    for index, (left, right) in enumerate(zip(reference.get("segments", []), candidate.get("segments", []))):
        for key in ("start", "end", "text", "speaker"):
            if left.get(key) != right.get(key):
                differences.append(f"segment[{index}].{key}: {left.get(key)!r} != {right.get(key)!r}")
                if len(differences) >= limit:
                    return differences

    for index, (left, right) in enumerate(zip(reference.get("word_segments", []), candidate.get("word_segments", []))):
        if left.get("speaker") != right.get("speaker") or left.get("word") != right.get("word"):
            differences.append(
                f"word[{index}]: {left.get('word')!r}/{left.get('speaker')} != "
                f"{right.get('word')!r}/{right.get('speaker')}"
            )
            if len(differences) >= limit:
                return differences

    return differences


def compare_structures(reference_structure, candidate_structure):
    """IN: two structure_of dicts   OUT: (all matched, list of (key, left, right, matched)).

    The rows are returned rather than printed so the gate's exit status and its table come
    from the same pass over the data.
    """
    rows = []
    matched_all = True
    for key in reference_structure:
        left, right = reference_structure[key], candidate_structure[key]
        matched = left == right
        matched_all = matched_all and matched
        rows.append((key, left, right, matched))
    return matched_all, rows
