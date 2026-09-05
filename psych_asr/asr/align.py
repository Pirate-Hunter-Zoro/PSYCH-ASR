"""Stage 1a: Whisper transcription, then wav2vec2 forced alignment.

Needs whisperx, so this imports only in asr_env.

These are passes 1 and 2 of the original four-pass Stage 1, kept as ONE function because
they run ONCE per session and every candidate diarizer fans off the identical transcript
with identical word timings.

The reason is methodological, not a compute saving: Stage 1 costs ~3 minutes on one A40,
so re-running ASR per arm would be affordable. It would still be wrong -- Whisper's output
would then differ across arms and a DER comparison would be confounded by transcript
differences rather than measuring the diarizers.

A conceptual walkthrough of these two passes lives at
~/Research-Journey/psych-asr-feasibility/stage1_pipeline_walkthrough.pdf

Shapes are annotated at each step. N = number of audio samples = seconds * 16000.
"""

import whisperx


def load_audio(audio_path):
    """Decode the recording ONCE, for every pass that needs the waveform.

    IN:  audio file on disk   OUT: float32 array, shape (N,), values in [-1, 1]

    Normalized pressure readings in strict time order at 16 kHz, so frequencies above the
    8 kHz Nyquist limit are gone. 50 minutes is an array of about 48,000,000 samples; a
    word starting at 4.5 s begins near index 72,000.

    ASR, alignment and the pyannote diarizer all take this same array, so ffmpeg runs a
    single time per job and the diarizer sees exactly the waveform the words were aligned
    against.
    """
    return whisperx.load_audio(audio_path)


def transcribe_and_align(decoded_audio, model_dir, batch_size=16, device="cuda"):
    """Run the two passes and return the aligned transcript.

    IN:  decoded_audio -- the (N,) array from load_audio
         model_dir     -- absolute path to the staged faster-whisper-large-v3 directory
         batch_size    -- WhisperX's batched decode width
    OUT: (align_result, asr_segment_count)

    align_result is {"segments": [...with "words"], "word_segments": [...], "language"},
    with NO speaker keys anywhere -- Stage 1c adds those.

    PASS 1, ASR. The model is loaded by absolute local path with local_files_only, in
    float16, with language forced to English so per-chunk language detection is skipped.
    Segments come out one per VAD chunk with loose (+/-0.5 s) boundaries and no word-level
    timing. The path must be a STRING, not a Path -- only a str takes faster-whisper's
    local-directory branch.

    PASS 2, FORCED ALIGNMENT. For English, load_align_model resolves to the TORCHAUDIO
    WAV2VEC2_ASR_BASE_960H bundle out of TORCH_HOME -- not a Hugging Face download, so
    HF_HUB_OFFLINE does not cover that path and TORCH_HOME must be exported. align() then
    re-times the existing text against the waveform at phoneme resolution, adding a "words"
    list with per-word start/end/score.

    Alignment RE-SPLITS segments at sentence boundaries (via nltk punkt_tab), so the
    aligned segment count is normally HIGHER than the ASR segment count. avg_logprob is
    carried through the split. align() returns only "segments" and "word_segments" and does
    NOT carry "language" forward, so it is re-attached here.

    WHY ALIGNMENT MUST PRECEDE DIARIZATION: the join is purely temporal, and Whisper's
    native segment edges are loose enough that a boundary landing inside a speaker change
    would stamp words onto the wrong person. Word-level times make the join tight.
    """
    # IN: model dir path   OUT: FasterWhisperPipeline (Whisper + VAD + tokenizer)
    pipeline_model = whisperx.load_model(
        str(model_dir), device=device, compute_type="float16", language="en", local_files_only=True,
    )
    # IN: (N,)   OUT: {"segments": [{text, start, end, avg_logprob}, ...], "language": str}
    processed_audio = pipeline_model.transcribe(
        decoded_audio, batch_size=batch_size, print_progress=True,
    )

    # IN: language code   OUT: (wav2vec2 model, metadata dict)
    alignment_model, metadata_dict = whisperx.load_align_model(processed_audio["language"], device=device)
    # IN: segments list + (N,)   OUT: {"segments": [...+ "words"], "word_segments": [...]}
    align_result = whisperx.align(
        processed_audio["segments"], alignment_model, metadata_dict, decoded_audio,
        device=device, print_progress=True,
    )
    align_result["language"] = processed_audio["language"]

    return align_result, len(processed_audio["segments"])


def format_alignment_summary(align_result, asr_segment_count):
    """IN: the aligned transcript + the pre-alignment segment count   OUT: log lines.

    Both counts are reported because the RISE between them is the sentence re-split, and
    an aligned count that failed to rise means alignment did not run the way it should.
    """
    return [
        f"Segments (ASR)     : {asr_segment_count}",
        f"Segments (aligned) : {len(align_result['segments'])}",
        f"Words              : {len(align_result['word_segments'])}",
        f"Last segment end   : {align_result['segments'][-1]['end']}",
        f"Language           : {align_result['language']}",
    ]
