"""Stage 1c: join one arm's RTTM onto the shared aligned transcript, then render.

IN:  <stem>.aligned.json from Stage 1a + <stem>.<arm>.rttm from Stage 1b
OUT: <stem>.<arm>.diarized.json + <stem>.<arm>.transcript.txt

CPU only -- no model loads, no GPU. This is the shared tail of every bake-off arm, which
is the point: one join implementation and one renderer across all arms, so the comparison
measures the diarizers rather than the glue.

The arm name is carried in the filename of every artifact from 1b onward, so which
transcript came from which model is a property of the file, not of a note somewhere.

Runs in asr_env, for whisperx.assign_word_speakers. That function reads exactly three
columns off the diarization DataFrame -- start, end, speaker -- and ignores everything
else, which is the whole reason an RTTM from any diarizer substitutes in unchanged.
"""

from argparse import ArgumentParser
from pathlib import Path
import json

import whisperx

# Sibling modules in scripts/; `python scripts/join_speakers.py` puts that dir on sys.path[0].
from rttm_io import read_rttm_as_dataframe
from render_transcript import format_summary, write_readable_transcript


def relink_word_segments(transcript):
    """Restore the aliasing that json.dump / json.load silently broke.

    IN:  a transcript dict freshly loaded from <stem>.aligned.json  OUT: nothing; mutated in place

    THIS IS LOAD-BEARING, AND ITS ABSENCE IS INVISIBLE. In memory, whisperx.align builds
    "word_segments" by concatenating THE SAME word dicts that hang off each segment's
    "words" list -- one object, two references. assign_word_speakers walks the segments and
    stamps "speaker" onto those dicts, and the words in "word_segments" acquire the key
    because they ARE those dicts.

    Serializing to JSON and reading it back produces two INDEPENDENT copies. Without this
    call the join stamps only the per-segment copies, "word_segments" comes back with zero
    speakers out of 7298, and nothing raises: the file looks complete, the segment-level
    talk-time table is correct, and every word-level feature downstream is silently empty.
    The 1a/1b/1c regression gate is what caught it.

    Rebuilding the list from the segments' own dicts restores one-object-two-references, so
    the join behaves exactly as it did in the single-job script.
    """
    rebuilt = [word for segment in transcript.get("segments", []) for word in segment.get("words", [])]
    existing = transcript.get("word_segments", [])

    # A guard rather than a silent overwrite: if the concatenation does not reproduce the
    # serialized list, the assumption above no longer holds and the join must not paper
    # over it with a plausible-looking substitute.
    if len(rebuilt) != len(existing) or any(
        new.get("word") != old.get("word") for new, old in zip(rebuilt, existing)
    ):
        raise SystemExit(
            f"word_segments ({len(existing)} words) is not the concatenation of the segments' "
            f"words ({len(rebuilt)}). Stage 1a's output does not have the shape Stage 1c assumes; "
            f"joining would produce a file whose word-level speakers are wrong rather than absent."
        )

    transcript["word_segments"] = rebuilt


def main():
    parser = ArgumentParser(description="Stage 1c: join a diarization RTTM onto the aligned transcript.")
    parser.add_argument("aligned", type=str, help="path to <stem>.aligned.json from Stage 1a")
    parser.add_argument("rttm", type=str, help="path to <stem>.<arm>.rttm from Stage 1b")
    parser.add_argument("--arm", type=str, default=None,
                        help="arm name for the output filenames (default: inferred from the RTTM filename)")
    parser.add_argument("--outdir", type=str, default=None,
                        help="where to write (default: beside the aligned JSON)")
    args = parser.parse_args()

    aligned_path = Path(args.aligned)
    rttm_path = Path(args.rttm)

    # "<stem>.aligned.json" -> "<stem>"
    stem = aligned_path.stem
    if stem.endswith(".aligned"):
        stem = stem[: -len(".aligned")]

    # "<stem>.<arm>.rttm" -> "<arm>"
    arm = args.arm
    if arm is None:
        rttm_stem = rttm_path.stem
        arm = rttm_stem[len(stem) + 1:] if rttm_stem.startswith(stem + ".") else rttm_stem

    output_dir = Path(args.outdir) if args.outdir else aligned_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(aligned_path) as f:
        aligned = json.load(f)

    relink_word_segments(aligned)

    # IN: RTTM path                   OUT: DataFrame with start, end, speaker
    speaker_df = read_rttm_as_dataframe(rttm_path)
    if speaker_df.empty:
        raise SystemExit(f"{rttm_path} holds no speaker turns -- the 1b job for arm '{arm}' produced nothing to join.")

    # IN: turns DataFrame + transcript  OUT: same transcript dict, "speaker" keys added in place
    # "speaker" is set only where a transcript span overlaps a turn, and fill_nearest is off,
    # so the key can simply be absent. Anything downstream must tolerate that.
    augmented = whisperx.assign_word_speakers(speaker_df, aligned)

    output_path = output_dir / f"{stem}.{arm}.diarized.json"
    with open(output_path, "w") as f:
        json.dump(augmented, f, indent=4, ensure_ascii=False)

    speakers = {s.get("speaker") for s in augmented["segments"]} - {None}
    unlabeled_segments = sum(1 for s in augmented["segments"] if "speaker" not in s)
    unlabeled_words = sum(1 for w in augmented["word_segments"] if "speaker" not in w)
    print(
        f"Arm                : {arm}\n"
        f"Turns joined       : {len(speaker_df)}\n"
        f"Segments           : {len(augmented['segments'])}\n"
        f"Words              : {len(augmented['word_segments'])}\n"
        f"Speaker labels     : {len(speakers)} {sorted(speakers)}\n"
        f"Unlabeled segments : {unlabeled_segments}\n"
        f"Unlabeled words    : {unlabeled_words}\n"
        f"Wrote {output_path}",
        flush=True,
    )

    # Talk-time split is the cheapest diarization sanity check there is: two people in a
    # room do not split 97/3, so that number falsifies a collapsed clustering from the log
    # alone, before anyone opens the audio.
    readable_path = output_dir / f"{stem}.{arm}.transcript.txt"
    summary = write_readable_transcript(augmented, readable_path, f"{stem} [{arm}]")
    print("\n".join(format_summary(f"{stem} [{arm}]", summary)), flush=True)
    print(f"Wrote {readable_path}", flush=True)


if __name__ == "__main__":
    main()
