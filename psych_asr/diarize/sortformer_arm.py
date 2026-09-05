"""Stage 1b, arms B and C: NVIDIA Sortformer.

Runs in nemo_env. NeMo carries its own torch, hydra, lightning and transformers pins,
which is exactly the set that would break asr_env's locked whisperx pin chain.

Both checkpoints load through the same NeMo class, so one module and one env cover both:

  arm B  offline    nvidia/diar_sortformer_4spk-v1              CC BY-NC 4.0
  arm C  streaming  nvidia/diar_streaming_sortformer_4spk-v2.1  NVIDIA Open Model License

Arm C's license is the reason to care which one wins: arms A and B are non-commercial-only,
and the stated deliverable is an R21/R01, so an NC weight license is the kind of thing that
is invisible for two years and then blocks a translation path.

TWO THINGS ABOUT THESE MODELS DIFFER FROM THE PYANNOTE ARMS, AND BOTH ARE FINDINGS RATHER
THAN IMPLEMENTATION DETAILS.

1. THEY CANNOT BE PINNED TO TWO SPEAKERS. Sortformer is end-to-end with a hard four-speaker
   ceiling and no clustering stage to constrain, so there is no num_speakers to pass. The
   pyannote path takes an exact count and pins it to 2. On a known dyad that constraint is
   worth real DER, and losing it is a genuine cost of switching, not a footnote. The CLI
   accepts --num-speakers for flag compatibility and deliberately ignores it, saying so.

2. THE OFFLINE ARM DOES NOT SURVIVE A 50-MINUTE FILE UNAIDED, so it is windowed and
   stitched here. See diarize/windowing.py for the stitcher and for why some of arm B's
   error will be seam error rather than model error.
"""

import shutil
from pathlib import Path

import soundfile as sf
from nemo.collections.asr.models import SortformerEncLabelModel

from .windowing import parse_segments, stitch_windows, window_starts

# NVIDIA's "very high latency" streaming preset, all counts in 80 ms frames. It is the
# accuracy-favouring end of their published table (RTF 0.002); the low-latency preset
# exists to hit a 1 s input-buffer latency, which is irrelevant for batch processing of
# recorded sessions.
STREAMING_PRESET = {
    "chunk_len": 340,
    "chunk_right_context": 40,
    "fifo_len": 40,
    "spkcache_update_period": 300,
    "spkcache_len": 188,
}


def load_model(checkpoint, device="cuda"):
    """IN: absolute path to a .nemo archive   OUT: the loaded model, in eval mode.

    restore_from reads the archive straight off disk and makes NO Hub call at all, so this
    needs no token and works with HF_HUB_OFFLINE=1 set. NVIDIA's cards use from_pretrained
    with a token; restore_from is the offline equivalent and takes strict=False.
    """
    model = SortformerEncLabelModel.restore_from(
        restore_path=str(checkpoint), map_location=device, strict=False,
    )
    model.eval()
    return model


def apply_streaming_preset(model, preset=STREAMING_PRESET):
    """IN: a loaded streaming model   OUT: the preset applied, as a dict for logging."""
    for name, value in preset.items():
        setattr(model.sortformer_modules, name, value)
    return preset


def diarize_whole(model, audio_path, batch_size=1):
    """IN: model + audio path   OUT: (start, end, speaker) tuples over the whole file.

    Used by arm C unconditionally -- streaming handles arbitrary length by construction, so
    no windowing, no stitcher, no seam error -- and by arm B when the recording is short
    enough to fit in one pass.
    """
    segments = model.diarize(audio=[str(audio_path)], batch_size=batch_size)
    return parse_segments(segments[0])


def diarize_windowed(model, audio_path, scratch_dir, window, overlap, batch_size=1, log=print):
    """Run the offline model over overlapping windows and stitch the results.

    IN:  loaded model, path to the 16 kHz mono WAV, a scratch directory for window WAVs,
         window and overlap lengths in seconds, batch size, a line-printer for progress
    OUT: list of (start, end, speaker) tuples over the whole recording

    WINDOW WAVS ARE PHI -- they are cut from session audio. They are written under the
    output directory, which is gitignored, never anywhere else, and the scratch directory
    is removed in a finally block so a crash mid-run does not leave session audio lying
    around under a new name.
    """
    scratch_dir = Path(scratch_dir)
    info = sf.info(str(audio_path))
    duration = info.frames / info.samplerate

    if duration <= window:
        return diarize_whole(model, audio_path, batch_size)

    scratch_dir.mkdir(parents=True, exist_ok=True)
    audio, sample_rate = sf.read(str(audio_path), dtype="float32")
    starts = window_starts(duration, window, overlap)
    log(f"Windowing {duration:.1f}s into {len(starts)} windows of {window:.0f}s "
        f"(stride {window - overlap:.0f}s, overlap {overlap:.0f}s)")

    window_results = []
    try:
        for index, start in enumerate(starts):
            first_sample = int(round(start * sample_rate))
            last_sample = min(int(round((start + window) * sample_rate)), len(audio))
            window_path = scratch_dir / f"window_{index:03d}.wav"
            sf.write(str(window_path), audio[first_sample:last_sample], sample_rate)

            segments = model.diarize(audio=[str(window_path)], batch_size=batch_size)
            turns = parse_segments(segments[0], time_offset=start)
            log(f"  window {index + 1}/{len(starts)} [{start:.0f}s]: {len(turns)} turns, "
                f"{len({t[2] for t in turns})} local speakers")
            window_results.append((start, turns))
            window_path.unlink()
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    return stitch_windows(window_results, overlap)
