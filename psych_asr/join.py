"""Stage 1c: stamp one arm's speaker turns onto the shared aligned transcript.

Needs whisperx for assign_word_speakers, so this imports only in asr_env. CPU only -- no
model loads, no GPU.

This is the shared tail of every bake-off arm, which is the point: ONE join implementation
and ONE renderer across all arms, so the comparison measures the diarizers rather than the
glue.

assign_word_speakers reads exactly three columns off the diarization DataFrame -- start,
end, speaker -- and ignores "segment" and "label" entirely (verified in the installed
whisperx/diarize.py 3.8.6). That is the whole reason an RTTM from ANY diarizer substitutes
in unchanged.
"""

import whisperx

from .artifacts.rttm_io import read_rttm_as_dataframe
from .artifacts.transcripts import relink_word_segments, speaker_labels, unlabeled_counts


def join_arm(aligned, rttm_path, arm):
    """IN: the loaded aligned transcript, a path to that arm's RTTM, the arm name
    OUT: (augmented transcript dict, number of turns joined)

    The aligned dict is MUTATED IN PLACE and also returned -- that is assign_word_speakers'
    own contract, kept rather than hidden behind a copy, because a copy would break the
    one-object-two-references aliasing that relink_word_segments exists to restore.

    "speaker" is set only where a transcript span overlaps a turn, and fill_nearest is off,
    so the key can simply be absent. Anything downstream must tolerate that.
    """
    relink_word_segments(aligned)

    # IN: RTTM path   OUT: DataFrame with start, end, speaker
    speaker_df = read_rttm_as_dataframe(rttm_path)
    if speaker_df.empty:
        raise SystemExit(
            f"{rttm_path} holds no speaker turns -- the 1b job for arm '{arm}' produced nothing to join."
        )

    augmented = whisperx.assign_word_speakers(speaker_df, aligned)
    return augmented, len(speaker_df)


def format_join_summary(arm, augmented, num_turns):
    """IN: arm name, the joined transcript, the turn count   OUT: log lines."""
    labels = speaker_labels(augmented)
    unlabeled_segments, unlabeled_words = unlabeled_counts(augmented)
    return [
        f"Arm                : {arm}",
        f"Turns joined       : {num_turns}",
        f"Segments           : {len(augmented['segments'])}",
        f"Words              : {len(augmented['word_segments'])}",
        f"Speaker labels     : {len(labels)} {labels}",
        f"Unlabeled segments : {unlabeled_segments}",
        f"Unlabeled words    : {unlabeled_words}",
    ]


def join_annotation(annotation, aligned):
    """Join an in-memory pyannote Annotation onto the aligned transcript.

    IN:  a pyannote Annotation of speaker turns + the aligned transcript dict
    OUT: (augmented transcript dict, number of turns joined)

    The single-job Stage 1 path uses this; the split's Stage 1c uses join_arm instead,
    which reads an RTTM. NO RELINK IS NEEDED HERE, and one IS needed there: in-process,
    "word_segments" still holds THE SAME word dicts the segments do, so stamping the
    segments stamps the words. Serializing to JSON between the two steps is what breaks
    that aliasing -- see artifacts/transcripts.py:relink_word_segments.

    assign_word_speakers reads exactly start/end/speaker off the frame and ignores the
    "segment" and "label" columns DiarizationPipeline also builds, so they are not
    reconstructed.
    """
    import pandas as pd

    speaker_df = pd.DataFrame(
        [{"start": turn.start, "end": turn.end, "speaker": speaker}
         for turn, _, speaker in annotation.itertracks(yield_label=True)],
        columns=["start", "end", "speaker"],
    )
    return whisperx.assign_word_speakers(speaker_df, aligned), len(speaker_df)
