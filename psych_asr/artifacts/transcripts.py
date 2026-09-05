"""Reading and writing the Stage 1 transcript JSON, and the one repair it needs.

STDLIB ONLY.

The shape, both before and after the join:

    {"segments":      [{start, end, text, avg_logprob, words: [...], speaker?}, ...],
     "word_segments": [{word, start?, end?, score?, speaker?}, ...],
     "language":      "en"}

"speaker" is NOT guaranteed on any segment or word. assign_word_speakers sets it only
where a transcript span actually overlaps a diarized turn, and fill_nearest is off, so
there is no fallback. Every reader here uses .get and every reader downstream must too.
"""

import json
from pathlib import Path


def load_transcript(path):
    """IN: path to an .aligned.json or a .diarized.json   OUT: the dict."""
    with open(path) as f:
        return json.load(f)


def save_transcript(transcript, path):
    """IN: transcript dict + destination   OUT: the Path written.

    indent=4 and ensure_ascii=False match what Stage 1 has always written, so a re-run
    against the regression fixture compares byte for byte rather than diffing on layout.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(transcript, f, indent=4, ensure_ascii=False)
    return path


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


def speaker_labels(transcript):
    """IN: a transcript dict   OUT: the sorted set of distinct segment speaker labels.

    Segments with no "speaker" key contribute nothing rather than a None entry.
    """
    return sorted({s.get("speaker") for s in transcript.get("segments", [])} - {None})


def unlabeled_counts(transcript):
    """IN: a transcript dict   OUT: (segments with no speaker, words with no speaker).

    A few unlabeled items is normal -- the join only stamps a span that overlaps a turn.
    Many means diarization under-covered the audio, which is the thing worth seeing in a
    job log without opening the transcript.
    """
    segments = sum(1 for s in transcript.get("segments", []) if "speaker" not in s)
    words = sum(1 for w in transcript.get("word_segments", []) if "speaker" not in w)
    return segments, words


def word_label_columns(transcript):
    """IN: a transcript dict   OUT: (list of word strings, list of speaker labels or None).

    The two lists are positionally aligned and are the only thing the arm comparison needs:
    because Stage 1a ran once, every arm sits on the identical word sequence, so the ONLY
    thing that can differ between two arms is this second column.

    A word with no "speaker" key becomes None rather than being skipped -- WHICH words an
    arm declined to label is itself one of the differences worth seeing.
    """
    words, labels = [], []
    for word in transcript.get("word_segments", []):
        words.append(word.get("word", ""))
        labels.append(word.get("speaker"))
    return words, labels


def word_start_time(transcript, index, lookbehind=50):
    """IN: transcript dict + word index   OUT: that word's start in seconds, or None.

    Alignment leaves "start" off a word whose characters never received a timestamp (digits
    and symbols go through the wildcard emission column and can come back NaN), so this
    walks backwards to the nearest timed word rather than indexing and raising.
    """
    words = transcript.get("word_segments", [])
    for probe in range(index, max(index - lookbehind, -1), -1):
        if "start" in words[probe]:
            return words[probe]["start"]
    return None
