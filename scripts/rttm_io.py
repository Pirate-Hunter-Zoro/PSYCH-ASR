"""RTTM read/write -- the interchange format between Stage 1b and Stage 1c.

Stage 1 is split so that ASR runs ONCE and every candidate diarizer fans off the same
aligned transcript (see README, "The seam that makes this cheap"). The seam works because
whisperx.assign_word_speakers reads exactly three columns off the diarization DataFrame --
start, end, speaker -- and ignores everything else. Any diarizer that can emit those three
fields substitutes in with no change to the join.

RTTM is the carrier because it is simultaneously the standard diarization interchange
format AND what DER scorers consume, so the file the bake-off scores and the file the join
reads are the same artifact. No second serialization to keep in sync.

This module is imported from FOUR different conda envs (asr_env, diarizen_env, nemo_env,
diar_eval_env) whose torch and numpy pins are mutually incompatible. It therefore imports
nothing at module scope beyond the standard library; pandas is imported inside the one
function that needs it.

RTTM line format, ten space-separated fields:

    SPEAKER <uri> <channel> <start> <duration> <NA> <NA> <speaker> <NA> <NA>

Start and duration are seconds. Everything marked <NA> is unused by every consumer here.
"""

from pathlib import Path

# NIST's RTTM has ten columns; we emit turns only, so type is always SPEAKER and the
# confidence/lookahead slots stay unfilled.
_RTTM_TEMPLATE = "SPEAKER {uri} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"

# Sub-millisecond turns are a segmentation artifact, not speech, and a zero-duration turn
# makes assign_word_speakers' strict `intersection > 0` test unsatisfiable anyway. Dropping
# them here keeps every arm's RTTM comparable rather than each diarizer's own rounding.
MIN_TURN_DURATION = 0.001


def write_rttm(turns, uri, output_path):
    """Write speaker turns to an RTTM file.

    IN:  turns    -- iterable of (start_seconds, end_seconds, speaker_label)
         uri      -- recording id written into column 2; use the audio file stem
         output_path -- where to write
    OUT: the number of turns actually written (after the zero-duration filter)

    Turns are sorted by start time then end time so two arms that emit the same turns in
    different orders produce byte-identical files.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ordered = sorted(
        (float(start), float(end), str(speaker)) for start, end, speaker in turns
    )
    lines = []
    for start, end, speaker in ordered:
        duration = end - start
        if duration < MIN_TURN_DURATION:
            continue
        # A label containing whitespace would silently shift every later column.
        lines.append(_RTTM_TEMPLATE.format(
            uri=uri, start=start, duration=duration, speaker=speaker.replace(" ", "_")
        ))

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def write_rttm_from_annotation(annotation, uri, output_path):
    """Same as write_rttm, for a pyannote.core Annotation.

    IN:  annotation -- pyannote Annotation (what pyannote and DiariZen both return)
    OUT: number of turns written

    pyannote's own Annotation.to_rttm() exists, but it stamps the uri from the object and
    formats floats its own way. Going through write_rttm keeps every arm's RTTM identical
    in shape, which is the whole point of the interchange format.
    """
    return write_rttm(
        ((turn.start, turn.end, speaker) for turn, _, speaker in annotation.itertracks(yield_label=True)),
        uri,
        output_path,
    )


def read_rttm(rttm_path):
    """Parse an RTTM file into a list of turn dicts.

    IN:  path to an RTTM file
    OUT: list of {"start": float, "end": float, "speaker": str}, in file order

    Blank lines and comment lines are skipped. Any non-SPEAKER record type is skipped too:
    some scorers emit SPKR-INFO lines into the same file and they are not turns.
    """
    turns = []
    with open(rttm_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";") or line.startswith("#"):
                continue
            fields = line.split()
            if fields[0] != "SPEAKER":
                continue
            start = float(fields[3])
            duration = float(fields[4])
            turns.append({"start": start, "end": start + duration, "speaker": fields[7]})
    return turns


def read_rttm_as_dataframe(rttm_path):
    """Parse an RTTM into the DataFrame shape whisperx.assign_word_speakers expects.

    IN:  path to an RTTM file
    OUT: pandas DataFrame with columns start, end, speaker

    assign_word_speakers builds its IntervalTree from exactly these three columns
    (verified in whisperx/diarize.py 3.8.6); the "segment" and "label" columns that
    DiarizationPipeline also produces are never read, so they are not reconstructed here.

    pandas is imported inside the function on purpose -- this module is shared across envs
    and diar_eval_env should stay able to import it without a DataFrame library loading.
    """
    import pandas as pd

    return pd.DataFrame(read_rttm(rttm_path), columns=["start", "end", "speaker"])


def summarize_turns(turns):
    """Cheap per-arm diagnostics, printed by every 1b job into its own log.

    IN:  list of turn dicts from read_rttm (or any dicts with start/end/speaker)
    OUT: dict with turn count, distinct speaker count, covered span, total turn time,
         and overlap time

    Overlap time is the part that only exists here: the joined transcript structurally
    cannot represent two speakers at once, so simultaneous speech survives in the turn
    table and nowhere else. It is computed as (sum of turn durations) minus (duration of
    the union of all turns) -- the amount by which turns double-cover the timeline.
    """
    if not turns:
        return {"num_turns": 0, "num_speakers": 0, "span": 0.0,
                "turn_time": 0.0, "covered_time": 0.0, "overlap_time": 0.0}

    turn_time = sum(t["end"] - t["start"] for t in turns)

    # Union of the intervals, by sweeping in start order and merging what touches.
    merged = []
    for turn in sorted(turns, key=lambda t: (t["start"], t["end"])):
        if merged and turn["start"] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], turn["end"])
        else:
            merged.append([turn["start"], turn["end"]])
    covered_time = sum(end - start for start, end in merged)

    return {
        "num_turns": len(turns),
        "num_speakers": len({t["speaker"] for t in turns}),
        "span": max(t["end"] for t in turns),
        "turn_time": turn_time,
        "covered_time": covered_time,
        "overlap_time": turn_time - covered_time,
    }


def format_turn_summary(arm, uri, summary):
    """IN: arm name, recording id, summarize_turns dict  OUT: list of log lines."""
    overlap_share = (summary["overlap_time"] / summary["covered_time"] * 100.0) if summary["covered_time"] else 0.0
    return [
        "=" * 72,
        f"DIARIZATION — {uri} — arm: {arm}",
        "=" * 72,
        f"Turns          : {summary['num_turns']}",
        f"Speakers       : {summary['num_speakers']}",
        f"Covered span   : {summary['span']:.1f}s",
        f"Speech covered : {summary['covered_time']:.1f}s",
        f"Overlap        : {summary['overlap_time']:.1f}s  ({overlap_share:.1f}% of covered speech)",
        "=" * 72,
    ]
