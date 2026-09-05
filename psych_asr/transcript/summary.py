"""The talk-time table: the cheapest diarization sanity check there is.

STDLIB ONLY.

Two people in a room do not split their speech 97/3, so this number falsifies a collapsed
clustering FROM THE JOB LOG ALONE, before anyone opens the audio. It is also the first
Stage 3a structural feature, so it is worth having early rather than as a by-product.
"""

from .turns import UNKNOWN_SPEAKER, segment_speaker


def format_timestamp(seconds):
    """Seconds (float) -> "mm:ss" with minutes unbounded, e.g. 3012.4 -> "50:12".

    Minutes deliberately are not rolled into hours: a single mm:ss scale is easier to
    scrub against a media player's position readout for a ~50 minute session.
    """
    total_seconds = int(round(seconds))
    return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"


def summarize(segments, turns):
    """IN: segment list + turn list   OUT: dict with span/speech totals and a per-speaker
    breakdown of talk time, share, turn count and segment count.

    The denominator for the share is time ANYONE was speaking, not the wall-clock span --
    silence should not dilute the split between the two people.
    """
    span = max((s["end"] for s in segments), default=0.0)

    per_speaker = {}
    for segment in segments:
        speaker = segment_speaker(segment)
        entry = per_speaker.setdefault(speaker, {"talk_time": 0.0, "turns": 0, "segments": 0})
        entry["talk_time"] += segment["end"] - segment["start"]
        entry["segments"] += 1
    for turn in turns:
        per_speaker[turn["speaker"]]["turns"] += 1

    speech_time = sum(entry["talk_time"] for entry in per_speaker.values())
    for entry in per_speaker.values():
        entry["share"] = (entry["talk_time"] / speech_time * 100.0) if speech_time else 0.0

    return {
        "span": span,
        "speech_time": speech_time,
        "num_segments": len(segments),
        "num_turns": len(turns),
        "per_speaker": per_speaker,
    }


def format_summary(stem, summary):
    """IN: file stem + summary dict   OUT: list of header lines (no trailing newlines).

    Printed into the job log AND written as the transcript file's header, so the log and
    the file cannot disagree about what the run produced.
    """
    rule = "=" * 72
    speech_share = (summary["speech_time"] / summary["span"] * 100.0) if summary["span"] else 0.0

    lines = [
        rule,
        f"TRANSCRIPT — {stem}",
        rule,
        f"Audio span     : {format_timestamp(summary['span'])}",
        f"Speech time    : {format_timestamp(summary['speech_time'])}  ({speech_share:.1f}% of span)",
        f"Segments       : {summary['num_segments']}",
        f"Turns          : {summary['num_turns']}",
        "",
        f"{'Speaker':<16}{'Talk time':>11}{'Share':>10}{'Turns':>8}{'Segments':>10}",
    ]
    # Loudest speaker first, but UNKNOWN always last -- it is diagnostic, not a person.
    ordered = sorted(
        summary["per_speaker"].items(),
        key=lambda item: (item[0] == UNKNOWN_SPEAKER, -item[1]["talk_time"]),
    )
    for speaker, entry in ordered:
        lines.append(
            f"{speaker:<16}{format_timestamp(entry['talk_time']):>11}"
            f"{entry['share']:>9.1f}%{entry['turns']:>8}{entry['segments']:>10}"
        )
    lines.append(rule)
    return lines
