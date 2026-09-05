"""Turn a Stage 1 transcript dict into a play script somebody can read against the audio.

STDLIB ONLY. Runs on CPU in well under a second, which is why it rides along at the end of
every Stage 1 job rather than needing one of its own.

The .diarized.json is the machine artifact: indented JSON with a words[] list on every
segment. Nobody can read that against playing audio at speed, and "read five minutes
against the audio" is the check the whole project hinges on. This renders the same content
as one timestamped block per speaker turn, with the talk-time summary as its header.

OUTPUT IS PHI (session content). It is written beside the JSON under data/stage1/, which
is gitignored wholesale. Do not write it anywhere else.
"""

import textwrap

from .summary import format_summary, format_timestamp, summarize
from .turns import group_into_turns

# Width of the wrapped dialogue text, and the indent under each speaker heading.
WRAP_WIDTH = 96
DIALOGUE_INDENT = "    "


def render(transcript, stem):
    """IN: the transcript dict + a file stem   OUT: (full text str, summary dict).

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
