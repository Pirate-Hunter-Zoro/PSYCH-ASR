"""Stage 1b, baseline arm: pyannote speaker-diarization-community-1 -> RTTM.

Audio in, RTTM out. Nothing else. All joining and rendering lives in Stage 1c so that
every arm of the bake-off shares one join implementation and one renderer -- otherwise the
comparison would partly measure the glue.

This calls the pyannote Pipeline DIRECTLY rather than through whisperx's
DiarizationPipeline wrapper, for one reason: pyannote 4.x returns a DiarizeOutput carrying
speaker_diarization, exclusive_speaker_diarization AND speaker_embeddings, and the wrapper
throws away everything but the first before handing back a DataFrame. The exclusive
annotation is the same diarization with the per-frame speaker count clamped to one -- an
overlap-free view of the session -- so holding both gives overlapping speech as a set
difference, which is cheaper and less error-prone than reconstructing it from turn
intersections.

The call itself is otherwise identical to what DiarizationPipeline.__call__ does (verified
against whisperx/diarize.py 3.8.6): same 16 kHz float32 array from whisperx.load_audio,
same waveform dict, same num_speakers argument, same itertracks read of
speaker_diarization. That identity is what makes the 1a/1b/1c split provably
behavior-preserving against the job-2032471 regression fixture.

Runs in asr_env, which already holds pyannote.audio 4.0.7.
"""

from argparse import ArgumentParser
from pathlib import Path

import torch
import whisperx
from pyannote.audio import Pipeline

# Sibling module in scripts/; `python scripts/diarize_pyannote.py` puts that dir on sys.path[0].
from rttm_io import write_rttm_from_annotation, read_rttm, summarize_turns, format_turn_summary

ARM = "community-1"


def main():
    parser = ArgumentParser(description="Stage 1b baseline: diarize one WAV with pyannote community-1.")
    parser.add_argument("audio", type=str)
    parser.add_argument("--outdir", type=str, default="data/stage1")
    parser.add_argument("--model-dir", type=str, default="/media/studies/ehr_study/analysis/mferguson/models/pyannote-speaker-diarization-community-1")
    parser.add_argument("--num-speakers", type=int, default=2,
                        help="exact speaker count; every pilot session is a known dyad, and "
                             "pinning it removes both clustering failure modes for free")
    parser.add_argument("--arm", type=str, default=ARM,
                        help="name used in the output filenames; identifies which diarizer produced them")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    uri = audio_path.stem

    # Same decode Stage 1a used, so the diarizer sees exactly the waveform the words were
    # aligned against. IN: audio file  OUT: float32 array, shape (N,)
    decoded_audio = whisperx.load_audio(audio_path)
    audio_data = {"waveform": torch.from_numpy(decoded_audio[None, :]), "sample_rate": 16000}

    # from_pretrained accepts a local DIRECTORY and finds config.yaml inside it; the
    # config's $model/... references resolve against that same directory.
    pipeline = Pipeline.from_pretrained(args.model_dir).to(torch.device("cuda"))

    # IN: waveform dict + exact speaker count   OUT: DiarizeOutput
    output = pipeline(audio_data, num_speakers=args.num_speakers)

    rttm_path = output_dir / f"{uri}.{args.arm}.rttm"
    num_turns = write_rttm_from_annotation(output.speaker_diarization, uri, rttm_path)

    # The overlap-free view, kept alongside. Its own set difference against the main RTTM
    # is where every Stage 3a overlap/interruption feature comes from.
    exclusive = getattr(output, "exclusive_speaker_diarization", None)
    if exclusive is not None:
        exclusive_path = output_dir / f"{uri}.{args.arm}.exclusive.rttm"
        num_exclusive = write_rttm_from_annotation(exclusive, uri, exclusive_path)
        print(f"Wrote {exclusive_path} ({num_exclusive} turns, overlap-free view)", flush=True)

    print(f"Wrote {rttm_path} ({num_turns} turns)", flush=True)
    print("\n".join(format_turn_summary(args.arm, uri, summarize_turns(read_rttm(rttm_path)))), flush=True)


if __name__ == "__main__":
    main()
