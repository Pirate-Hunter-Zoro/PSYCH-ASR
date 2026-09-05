"""Collapsing segments into conversational turns.

STDLIB ONLY, and it is the piece three other modules need: the renderer groups turns to
read as dialogue, the regression gate counts them, and the scorer finds backchannels among
them. One implementation, so those three cannot drift apart.
"""

# assign_word_speakers sets "speaker" only where a transcript span overlaps a diarized
# turn, and fill_nearest is off by default -- so the key can simply be absent. Those
# segments get this label rather than being dropped or merged into a neighbour: a cluster
# of them is itself the diagnostic that diarization under-covered the audio.
UNKNOWN_SPEAKER = "UNKNOWN"


def segment_speaker(segment):
    """Tolerant lookup -- never index "speaker" directly (it is not guaranteed)."""
    return segment.get("speaker") or UNKNOWN_SPEAKER


def group_into_turns(segments, unknown=UNKNOWN_SPEAKER):
    """Collapse consecutive same-speaker segments into conversational turns.

    IN:  list of segment dicts (start, end, text, speaker?), in time order
         unknown -- the label for a segment with no speaker key. Pass None to keep the
         raw absent-speaker value, which is what the backchannel scan wants: it must not
         merge two genuinely unlabeled stretches into one turn on the strength of a
         placeholder they never carried.
    OUT: list of turn dicts: {"speaker": str|None, "start": float, "end": float, "text": str}

    Alignment re-splits segments at sentence boundaries, so one speaker's uninterrupted
    minute arrives as a dozen segments. Grouping is what makes the file read as a
    conversation (one paragraph per turn) instead of one line per sentence.

    Empty-text segments are skipped entirely, so a silent segment never breaks a turn in
    half.
    """
    turns = []
    for segment in segments:
        speaker = segment.get("speaker") or unknown
        text = segment.get("text", "").strip()
        if not text:
            continue
        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["text"] += " " + text
            turns[-1]["end"] = segment["end"]
        else:
            turns.append({
                "speaker": speaker,
                "start": segment["start"],
                "end": segment["end"],
                "text": text,
            })
    return turns
