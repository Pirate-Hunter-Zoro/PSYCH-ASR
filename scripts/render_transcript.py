"""Stage 1 post-pass: turn <stem>.diarized.json into a human-readable transcript.

The Stage 1 artifact is indented JSON with a words[] list on every segment. Nobody can
read that against playing audio at speed, and "read 5 minutes against the audio" is the
check the whole project hinges on. This renders the same content as a play script: one
timestamped block per speaker turn, plus a talk-time summary at the top.

Runs on CPU in well under a second -- import it from run_whisperx.py or call it on an
existing .diarized.json from the login node. No GPU, no sbatch, no model loads.

Output is PHI (session content). It is written beside the JSON under data/stage1/, which
is gitignored wholesale. Do not write it anywhere else.
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import textwrap

# assign_word_speakers sets "speaker" only where a transcript span overlaps a diarized
# turn, and fill_nearest is off by default -- so the key can simply be absent. Those
# segments get this label rather than being dropped or merged into a neighbour: a cluster
# of them is itself the diagnostic that diarization under-covered the audio.
UNKNOWN_SPEAKER = "UNKNOWN"

# Width of the wrapped dialogue text, and the indent under each speaker heading.
WRAP_WIDTH = 96
DIALOGUE_INDENT = "    "


def format_timestamp(seconds):
    """Seconds (float) -> "mm:ss" with minutes unbounded, e.g. 3012.4 -> "50:12".

    Minutes deliberately are not rolled into hours: a single mm:ss scale is easier to
    scrub against a media player's position readout for a ~50 minute session.
    """
    total_seconds = int(round(seconds))
    return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"


def segment_speaker(segment):
    """Tolerant lookup -- never index "speaker" directly (it is not guaranteed)."""
    return segment.get("speaker") or UNKNOWN_SPEAKER


def group_into_turns(segments):
    """Collapse consecutive same-speaker segments into conversational turns.

    IN:  list of segment dicts (start, end, text, speaker?), in time order
    OUT: list of turn dicts: {"speaker": str, "start": float, "end": float, "text": str}

    Alignment re-splits segments at sentence boundaries, so one speaker's uninterrupted
    minute arrives as a dozen segments. Grouping is what makes the file read as a
    conversation (one paragraph per turn) instead of one line per sentence.
    """
    turns = []
    for segment in segments:
        speaker = segment_speaker(segment)
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


def summarize(segments, turns):
    """Compute the cheapest possible diarization sanity check.

    IN:  segment list + turn list
    OUT: dict with span/speech totals and a per-speaker breakdown

    Talk-time ratio is the first Stage 3 structural feature anyway, and a 97/3 split
    means diarization collapsed -- visible here long before anyone listens to the audio.
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

    # Denominator for the ratio is time anyone was speaking, not the wall-clock span --
    # silence should not dilute the split between the two people.
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
    """IN: file stem + summary dict   OUT: list of header lines (no trailing newlines)."""
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


def render(transcript, stem):
    """IN: the .diarized.json dict + a file stem   OUT: (full text str, summary dict).

    Body format, one block per turn:

        [03:12] SPEAKER_00
            wrapped dialogue text...
    """
    segments = transcript.get("segments", [])
    turns = group_into_turns(segments)
    summary = summarize(segments, turns)

    lines = format_summary(stem, summary)
    lines.append("")
    for turn in turns:
        lines.append(f"[{format_timestamp(turn['start'])}] {turn['speaker']}")
        lines.extend(textwrap.wrap(
            turn["text"],
            width=WRAP_WIDTH,
            initial_indent=DIALOGUE_INDENT,
            subsequent_indent=DIALOGUE_INDENT,
        ))
        lines.append("")

    return "\n".join(lines) + "\n", summary


def write_readable_transcript(transcript, output_path, stem):
    """Render and write the .txt. Returns the summary dict so callers can print it.

    IN:  transcript dict, output Path, file stem
    OUT: summary dict (also written as the file's header)
    """
    text, summary = render(transcript, stem)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(text)
    return summary


def main():
    parser = ArgumentParser(description="Render a Stage 1 .diarized.json as a readable transcript.")
    parser.add_argument("transcript", type=str, help="path to <stem>.diarized.json")
    parser.add_argument("--outdir", type=str, default=None,
                        help="where to write the .txt (default: beside the input JSON)")
    args = parser.parse_args()

    transcript_path = Path(args.transcript)
    with open(transcript_path) as f:
        transcript = json.load(f)

    # "<stem>.diarized.json" -> "<stem>" -> "<stem>.transcript.txt"
    stem = transcript_path.stem
    if stem.endswith(".diarized"):
        stem = stem[: -len(".diarized")]
    output_dir = Path(args.outdir) if args.outdir else transcript_path.parent
    output_path = output_dir / (stem + ".transcript.txt")

    summary = write_readable_transcript(transcript, output_path, stem)
    print("\n".join(format_summary(stem, summary)), flush=True)
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
