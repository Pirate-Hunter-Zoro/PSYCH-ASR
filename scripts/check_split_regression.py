"""Regression check for the 1a/1b/1c split: does composing the three steps reproduce the
single-job result exactly?

The split must be provably behavior-preserving BEFORE any challenger diarizer is added.
Otherwise a difference between arms cannot be attributed -- it might be the model, or it
might be the refactor, and there is no way to tell afterwards.

The fixture is job 2032471 (2026-08-12, full ~50 min session), whose four-pass
run_whisperx.py produced data/stage1/<stem>.diarized.json: 489 segments, 7298 words,
exactly 2 speaker labels, 89 turns, one UNKNOWN segment, 79/21 talk time. Composing
1a -> 1b(community-1) -> 1c must reproduce those numbers.

Two levels of comparison, reported separately because they fail for different reasons:

  STRUCTURE -- segment count, word count, turn count, speaker labels, unlabeled counts,
  talk-time shares. These are what the fixture is recorded as, and a mismatch here means
  the refactor changed behavior.

  EXACT FIELDS -- per-segment start/end/text/speaker and per-word speaker. A mismatch here
  with matching structure is worth reading before it is worth worrying about: Whisper
  decodes in float16 on a GPU and is not bit-reproducible across runs, so a handful of
  differing characters is a property of the hardware, not of the split. A systematic
  divergence is not.

CPU only. Runs in asr_env for the renderer import; loads no models.
"""

from argparse import ArgumentParser
from pathlib import Path
import json

# Sibling module in scripts/; `python scripts/check_split_regression.py` puts that dir on sys.path[0].
from render_transcript import group_into_turns, summarize, UNKNOWN_SPEAKER

# How many individual field differences to print before stopping. Enough to see whether a
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


def compare_fields(reference, candidate):
    """IN: two .diarized.json dicts   OUT: list of human-readable difference strings.

    Compared positionally, which is only meaningful once the counts already match.
    """
    differences = []

    for index, (left, right) in enumerate(zip(reference.get("segments", []), candidate.get("segments", []))):
        for key in ("start", "end", "text", "speaker"):
            if left.get(key) != right.get(key):
                differences.append(f"segment[{index}].{key}: {left.get(key)!r} != {right.get(key)!r}")
                if len(differences) >= MAX_REPORTED_DIFFERENCES:
                    return differences

    for index, (left, right) in enumerate(zip(reference.get("word_segments", []), candidate.get("word_segments", []))):
        if left.get("speaker") != right.get("speaker") or left.get("word") != right.get("word"):
            differences.append(
                f"word[{index}]: {left.get('word')!r}/{left.get('speaker')} != "
                f"{right.get('word')!r}/{right.get('speaker')}"
            )
            if len(differences) >= MAX_REPORTED_DIFFERENCES:
                return differences

    return differences


def main():
    parser = ArgumentParser(description="Verify the 1a/1b/1c split reproduces the single-job fixture.")
    parser.add_argument("reference", type=str, help="the fixture <stem>.diarized.json from job 2032471")
    parser.add_argument("candidate", type=str, help="the split's <stem>.community-1.diarized.json")
    args = parser.parse_args()

    with open(args.reference) as f:
        reference = json.load(f)
    with open(args.candidate) as f:
        candidate = json.load(f)

    reference_structure = structure_of(reference)
    candidate_structure = structure_of(candidate)

    rule = "=" * 72
    print(rule)
    print("1a/1b/1c SPLIT REGRESSION CHECK")
    print(rule)
    print(f"reference: {Path(args.reference).name}")
    print(f"candidate: {Path(args.candidate).name}")
    print("")
    print(f"{'':<20}{'reference':>26}{'candidate':>26}")

    structure_ok = True
    for key in reference_structure:
        left, right = reference_structure[key], candidate_structure[key]
        matched = left == right
        structure_ok = structure_ok and matched
        print(f"{key:<20}{str(left):>26}{str(right):>26}  {'ok' if matched else 'MISMATCH'}")

    print(rule)
    if not structure_ok:
        print("STRUCTURE DIFFERS. The split changed behavior -- this is a bug, not a finding.")
        print("Do not add a challenger arm until it is fixed; a difference between arms")
        print("could not be attributed to the model afterwards.")
        raise SystemExit(1)

    print("Structure matches the fixture.")
    differences = compare_fields(reference, candidate)
    if not differences:
        print("Exact field comparison also matches: the split is byte-identical to the fixture.")
    else:
        print(f"Structure matches but {len(differences)}+ individual fields differ. Read these before")
        print("treating it as a failure -- Whisper decodes in float16 on the GPU and is not")
        print("bit-reproducible across runs, so scattered decode noise is expected. A systematic")
        print("shift (every timestamp offset, every speaker swapped) is not.")
        for difference in differences:
            print(f"  {difference}")
    print(rule)


if __name__ == "__main__":
    main()
