"""Stage 1b, baseline arm: pyannote speaker-diarization-community-1.

Runs in asr_env, which already holds pyannote.audio 4.0.7.

This calls the pyannote Pipeline DIRECTLY rather than through whisperx's
DiarizationPipeline wrapper, for one reason: pyannote 4.x returns a DiarizeOutput carrying
speaker_diarization, exclusive_speaker_diarization AND speaker_embeddings, and the wrapper
throws away everything but the first before handing back a DataFrame. The exclusive
annotation is the same diarization with the per-frame speaker count clamped to one -- an
overlap-free view of the session -- so holding both gives overlapping speech as a SET
DIFFERENCE, which is cheaper and less error-prone than reconstructing it from turn
intersections.

The call itself is otherwise identical to what DiarizationPipeline.__call__ does (verified
against whisperx/diarize.py 3.8.6): same 16 kHz float32 array from whisperx.load_audio,
same waveform dict, same num_speakers argument, same itertracks read of
speaker_diarization. That identity is what makes the 1a/1b/1c split provably
behavior-preserving against the job-2032471 regression fixture.
"""

import torch
from pyannote.audio import Pipeline

SAMPLE_RATE = 16000


def load_pipeline(model_dir, device="cuda"):
    """IN: the staged community-1 directory   OUT: a pyannote Pipeline on that device.

    from_pretrained accepts a local DIRECTORY and finds config.yaml inside it -- there is
    no need to name the yaml -- and the config's $model/... references resolve against that
    same directory. The older speaker-diarization-3.1 does NOT work here: the 4.x
    constructor loads a PLDA model unconditionally and the 3.1 config predates PLDA, so an
    offline load raises LocalEntryNotFoundError.
    """
    return Pipeline.from_pretrained(str(model_dir)).to(torch.device(device))


def diarize(pipeline, decoded_audio, num_speakers):
    """IN: the pipeline, the (N,) float32 waveform, an EXACT speaker count
    OUT: the DiarizeOutput, carrying both the diarization and the exclusive view

    num_speakers is an exact count, not a bound. Clustering errs in BOTH directions -- it
    splits one person into two when their voice shifts, and merges two similar voices into
    one -- and every pilot session is a known dyad, so pinning the count removes both
    failure modes for free.
    """
    audio_data = {"waveform": torch.from_numpy(decoded_audio[None, :]), "sample_rate": SAMPLE_RATE}
    return pipeline(audio_data, num_speakers=num_speakers)


def exclusive_annotation(output):
    """IN: a DiarizeOutput   OUT: its exclusive_speaker_diarization, or None.

    Fetched with getattr rather than an attribute access, so a pyannote version that stops
    returning it degrades to "the extra file is absent" instead of crashing the arm that
    the regression gate depends on.
    """
    return getattr(output, "exclusive_speaker_diarization", None)
