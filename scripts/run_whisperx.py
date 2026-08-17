"""Stage 1 entry point: ASR -> forced alignment -> diarization -> speaker join.

Shapes are annotated at each step. N = number of audio samples = seconds * 16000.
A conceptual walkthrough with worked examples lives at
~/Research-Journey/psych-asr-feasibility/stage1_pipeline_walkthrough.pdf
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import whisperx
# Not re-exported at package level; whisperx.DiarizationPipeline raises AttributeError.
from whisperx.diarize import DiarizationPipeline
# Sibling module in scripts/; python scripts/run_whisperx.py puts that dir on sys.path[0].
from render_transcript import format_summary, write_readable_transcript


def main():
    parser = ArgumentParser()
    parser.add_argument("audio", type=str)
    parser.add_argument("--outdir", type=str, default="data/stage1")
    parser.add_argument("--model-dir", type=str, default="/media/studies/ehr_study/analysis/mferguson/models/faster-whisper-large-v3")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--diarize-model-dir", type=str, default="/media/studies/ehr_study/analysis/mferguson/models/pyannote-speaker-diarization-community-1")
    parser.add_argument("--num-speakers", type=int, default=2)
    args = parser.parse_args()

    audio_path = Path(args.audio)
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Turn the audio file into a series of normalized pressure readings between -1 and 1 in strict time order - stored as numpy array
    # Note this filters out sounds with frequencies above the Nyquist N/2 = 8000Hz frequency
    # IN:  audio file on disk          OUT: float32 array, shape (N,)
    decoded_audio = whisperx.load_audio(audio_path)
    # 50 minutes at 16kHz sampling is 16,000 * 60 * 50 which is an array close to length 48,000,000 (shape (48,000,000,))
    # If a word starts at 4.5 seconds, decoded_audio would have its first pressure reading around index 16,000*4.5=72,000

    # IN: model dir path              OUT: FasterWhisperPipeline (Whisper + VAD + tokenizer)
    # First arg must be a str, not a Path -- only a str takes faster-whisper's local-dir branch.
    pipeline_model = whisperx.load_model(args.model_dir, device="cuda", compute_type="float16", language="en", local_files_only=True)
    # IN: (N,)   OUT: {"segments": [{text, start, end, avg_logprob}, ...], "language": str}
    # One segment per VAD chunk; start/end are the chunk's bounds. No word-level timing yet.
    processed_audio = pipeline_model.transcribe(decoded_audio, batch_size=args.batch_size, print_progress=True)

    # IN: language code               OUT: (wav2vec2 model, metadata dict)
    alignment_model, metadata_dict = whisperx.load_align_model(processed_audio['language'], device="cuda")
    # IN: segments list + (N,)        OUT: {"segments": [...+ "words"], "word_segments": [...]}
    # Segments are re-split per sentence, so the count rises. Does not carry "language" forward.
    align_result = whisperx.align(processed_audio['segments'], alignment_model, metadata_dict, decoded_audio, device="cuda", print_progress=True)
    align_result['language'] = processed_audio['language']

    # Perform diarization of speakers with clustering
    # IN: model dir path              OUT: DiarizationPipeline (pyannote community-1)
    pipeline = DiarizationPipeline(args.diarize_model_dir, device="cuda")
    # Like a pytorch model, this is self-callable - returns dataframe of speaker instances, with start, end, and speaker columns
    # IN: (N,) + speaker count        OUT: DataFrame, one row per speaker turn
    # 2nd positional really is num_speakers, so the count is pinned rather than bounded.
    speaker_df = pipeline(decoded_audio, args.num_speakers)
    # IN: turns DataFrame + transcript  OUT: same transcript dict, "speaker" keys added in place
    # "speaker" is only set where a span overlaps a turn -- the key can be absent.
    augmented_transcript = whisperx.assign_word_speakers(speaker_df, align_result) # Walk every segment and every word, find which speaker turns overlap, and stamp dominant one as speaker for that segment

    output_path = output_dir / (audio_path.stem + ".diarized.json")
    with open(output_path, 'w') as f:
        json.dump(augmented_transcript, f, indent=4, ensure_ascii=False)

    speakers = set([s.get('speaker', None) for s in augmented_transcript['segments']]) - set([None])
    num_speakers = len(speakers)
    print(f"Number of segments: {len(augmented_transcript['segments'])}\nLast segment: {augmented_transcript['segments'][-1]['end']}\nDetected language: {augmented_transcript['language']}\nWord count: {len(augmented_transcript['word_segments'])}\nSpeaker count: {num_speakers}", flush=True)
        
    # Readable rendering: consecutive same-speaker segments collapse into one timestamped
    # block, so the file reads as a conversation. CPU-only and instant, so it rides along
    # here rather than needing a second job.
    # IN: transcript dict + out path  OUT: data/stage1/<stem>.transcript.txt (+ summary dict)
    readable_path = output_dir / (audio_path.stem + ".transcript.txt")
    summary = write_readable_transcript(augmented_transcript, readable_path, audio_path.stem)
    # Talk-time split is the cheapest diarization sanity check there is: a 97/3 split means
    # clustering collapsed, and the log says so before anyone opens the audio.
    print("\n".join(format_summary(audio_path.stem, summary)), flush=True)
    print(f"Wrote {readable_path}", flush=True)


if __name__=="__main__":
    main()
