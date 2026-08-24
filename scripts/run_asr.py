"""Stage 1a: ASR -> forced alignment. No diarization, no speaker labels.

This is passes 1 and 2 of the original four-pass run_whisperx.py, lifted out unchanged so
they run ONCE per session and every candidate diarizer fans off the identical transcript
with identical word timings.

The reason is methodological, not a compute saving: Stage 1 costs ~3 minutes on one A40,
so re-running ASR per arm would be affordable. It would still be wrong -- Whisper's output
would then differ across arms and a DER comparison would be confounded by transcript
differences rather than measuring the diarizers.

Output: data/stage1/<stem>.aligned.json -- a dict of segments, word_segments, and
language, with NO "speaker" keys anywhere. Stage 1c adds those.

A conceptual walkthrough of these two passes lives at
~/Research-Journey/psych-asr-feasibility/stage1_pipeline_walkthrough.pdf
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import whisperx


def main():
    parser = ArgumentParser(description="Stage 1a: transcribe and force-align one WAV.")
    parser.add_argument("audio", type=str)
    parser.add_argument("--outdir", type=str, default="data/stage1")
    parser.add_argument("--model-dir", type=str, default="/media/studies/ehr_study/analysis/mferguson/models/faster-whisper-large-v3")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    audio_path = Path(args.audio)
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Turn the audio file into normalized pressure readings between -1 and 1 in strict time
    # order. Filters out frequencies above the Nyquist limit of 8000 Hz.
    # IN:  audio file on disk          OUT: float32 array, shape (N,)  [N = seconds * 16000]
    decoded_audio = whisperx.load_audio(audio_path)

    # IN: model dir path              OUT: FasterWhisperPipeline (Whisper + VAD + tokenizer)
    # First arg must be a str, not a Path -- only a str takes faster-whisper's local-dir branch.
    pipeline_model = whisperx.load_model(args.model_dir, device="cuda", compute_type="float16", language="en", local_files_only=True)
    # IN: (N,)   OUT: {"segments": [{text, start, end, avg_logprob}, ...], "language": str}
    # One segment per VAD chunk; start/end are the chunk's bounds. No word-level timing yet.
    processed_audio = pipeline_model.transcribe(decoded_audio, batch_size=args.batch_size, print_progress=True)

    # IN: language code               OUT: (wav2vec2 model, metadata dict)
    # For English this resolves to the torchaudio WAV2VEC2_ASR_BASE_960H bundle out of
    # TORCH_HOME -- not a Hugging Face download, so HF_HUB_OFFLINE does not cover it.
    alignment_model, metadata_dict = whisperx.load_align_model(processed_audio['language'], device="cuda")
    # IN: segments list + (N,)        OUT: {"segments": [...+ "words"], "word_segments": [...]}
    # Segments are re-split per sentence, so the count rises above the ASR segment count.
    align_result = whisperx.align(processed_audio['segments'], alignment_model, metadata_dict, decoded_audio, device="cuda", print_progress=True)
    # align() returns only "segments" and "word_segments" -- it does not carry "language"
    # forward, so re-attach it before dumping. avg_logprob IS preserved through the split.
    align_result['language'] = processed_audio['language']

    output_path = output_dir / (audio_path.stem + ".aligned.json")
    with open(output_path, 'w') as f:
        json.dump(align_result, f, indent=4, ensure_ascii=False)

    print(
        f"Segments (ASR)     : {len(processed_audio['segments'])}\n"
        f"Segments (aligned) : {len(align_result['segments'])}\n"
        f"Words              : {len(align_result['word_segments'])}\n"
        f"Last segment end   : {align_result['segments'][-1]['end']}\n"
        f"Language           : {align_result['language']}\n"
        f"Wrote {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
