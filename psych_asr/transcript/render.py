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

import re
import textwrap

from .summary import format_summary, format_timestamp, summarize
from .turns import group_into_turns

# Width of the wrapped dialogue text, and the indent under each speaker heading.
WRAP_WIDTH = 96
DIALOGUE_INDENT = "    "


# What a rendered line is: the summary block, a speaker heading, a wrapped run of dialogue,
# or the blank line between turns. Only DIALOGUE lines carry a character span.
HEADER_LINE = "header"
HEADING_LINE = "heading"
DIALOGUE_LINE = "dialogue"
BLANK_LINE = "blank"


def _dialogue_spans(text, wrapped):
    """IN: a turn's full text + the lines textwrap produced from it
    OUT: one (start, end) character span per line, into that text

    Recovered by SEARCHING rather than by counting, because textwrap is allowed to break a
    hyphenated word across two lines: the pieces are still contiguous in the source, so a
    whitespace-tolerant match of each line's tokens finds them, while a token count would
    silently drift by one for the rest of the turn.

    Raises SystemExit rather than returning a wrong span. The spans are what an annotator's
    line number resolves to, and a line number that points at the wrong words applies a
    correction to the wrong sentence.
    """
    spans, cursor = [], 0
    for line in wrapped:
        parts = line.split()
        if not parts:
            spans.append(None)
            continue
        pattern = r"\s*".join(re.escape(part) for part in parts)
        found = re.compile(pattern).search(text, cursor)
        if found is None:
            raise SystemExit(
                "A wrapped transcript line could not be located in the turn text it came "
                "from; the line-number index would be wrong. This means textwrap altered "
                "the characters rather than only the line breaks."
            )
        spans.append((found.start(), found.end()))
        cursor = found.end()
    return spans


def render_with_line_index(transcript, stem):
    """IN: the transcript dict + a file stem   OUT: (text, summary, index).

    Same text `render` returns, plus the thing the Stage 2 correction pass needs: what
    every 1-based line of that text refers to. `index` is a list with one entry per line,
    each a dict:

        {"line": int, "kind": header|heading|dialogue|blank,
         "turn": int|None, "start": int|None, "end": int|None}

    "turn" indexes the turn list this render was built from; "start"/"end" are character
    offsets into THAT TURN'S text, present on dialogue lines only.

    This exists because the QC error log's Line column is a line number in the RENDERED
    .txt, while every correction has to be applied to the TURN STRUCTURE underneath it.
    Nothing else can translate between the two, and re-deriving the arithmetic at the call
    site would make the translation depend on the wrap width agreeing in two places.
    """
    segments = transcript.get("segments", [])
    turns = group_into_turns(segments)
    summary = summarize(segments, turns)

    lines, index = [], []

    def emit(text, kind, turn=None, span=None):
        lines.append(text)
        index.append({
            "line": len(lines),
            "kind": kind,
            "turn": turn,
            "start": span[0] if span else None,
            "end": span[1] if span else None,
        })

    for line in format_summary(stem, summary):
        emit(line, HEADER_LINE)
    emit("", BLANK_LINE)

    for position, turn in enumerate(turns):
        emit(f"[{format_timestamp(turn['start'])}] {turn['speaker']}", HEADING_LINE, position)
        wrapped = textwrap.wrap(
            turn["text"],
            width=WRAP_WIDTH,
            initial_indent=DIALOGUE_INDENT,
            subsequent_indent=DIALOGUE_INDENT,
        )
        for line, span in zip(wrapped, _dialogue_spans(turn["text"], wrapped)):
            emit(line, DIALOGUE_LINE, position, span)
        emit("", BLANK_LINE)

    return "\n".join(lines) + "\n", summary, index


def render(transcript, stem):
    """IN: the transcript dict + a file stem   OUT: (full text str, summary dict).

    Body format, one block per turn:

        [03:12] SPEAKER_00
            wrapped dialogue text...
    """
    text, summary, _ = render_with_line_index(transcript, stem)
    return text, summary


def render_turns(turns, stem, summary):
    """IN: a turn list + stem + an already-computed summary   OUT: the same text format.

    The Stage 2 corrected transcript is a TURN LIST that no longer has segments behind it
    -- correcting the words rebuilds the turns and there is nothing left to re-derive them
    from -- so it renders through here instead of through `render`. One formatter, so the
    corrected reference and the arm transcripts cannot come to look different.
    """
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
    return "\n".join(lines) + "\n"


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
