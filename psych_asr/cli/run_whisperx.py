"""Single-job Stage 1: ASR -> alignment -> diarization -> join -> render, in one process.

    python -m psych_asr.cli.run_whisperx <audio.wav>

THE 1a/1b/1c SPLIT EXISTS ALONGSIDE THIS, and the split is what to use for anything
comparing diarizers. This path remains for one session against the incumbent diarizer with
no dependency chain to submit, and it is the shape that produced the regression fixture.

It is now assembled from the same library functions the split's three steps use -- the same
decode, the same pyannote call, the same join, the same renderer -- so the two paths cannot
drift apart. That is the property the regression gate exists to check, and it is cheaper to
make true by construction than to keep checking.

It writes the UN-ARMED filenames (<stem>.diarized.json, <stem>.transcript.txt), which is
how the gate tells the fixture from the split's own output.
"""

from argparse import ArgumentParser
from pathlib import Path

from .. import config
from ..artifacts.naming import diarized_path, transcript_path
from ..artifacts.transcripts import save_transcript, speaker_labels
from ..asr.align import load_audio, transcribe_and_align
from ..diarize import pyannote_arm
from ..join import join_annotation
from ..transcript.render import write_readable_transcript
from ..transcript.summary import format_summary
from ._common import add_num_speakers, add_output_dir, prepare_output_dir, report


def build_parser():
    parser = ArgumentParser(description="Single-job Stage 1 against the incumbent diarizer.")
    parser.add_argument("audio", type=str)
    add_output_dir(parser)
    parser.add_argument("--model-dir", type=str, default=str(config.WHISPER_MODEL_DIR),
                        help="staged faster-whisper-large-v3 directory (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--diarize-model-dir", type=str, default=str(config.PYANNOTE_MODEL_DIR),
                        help="staged community-1 directory (default: %(default)s)")
    add_num_speakers(parser)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    audio_path = Path(args.audio)
    stem = audio_path.stem
    output_dir = prepare_output_dir(args.outdir)

    # Decoded ONCE and reused by ASR, alignment and the diarizer, so ffmpeg runs a single
    # time and the diarizer sees exactly the waveform the words were aligned against.
    decoded_audio = load_audio(audio_path)
    align_result, _ = transcribe_and_align(decoded_audio, args.model_dir, batch_size=args.batch_size)

    pipeline = pyannote_arm.load_pipeline(args.diarize_model_dir)
    output = pyannote_arm.diarize(pipeline, decoded_audio, args.num_speakers)
    augmented, _ = join_annotation(output.speaker_diarization, align_result)

    output_path = save_transcript(augmented, diarized_path(output_dir, stem))
    report([
        f"Number of segments: {len(augmented['segments'])}",
        f"Last segment: {augmented['segments'][-1]['end']}",
        f"Detected language: {augmented['language']}",
        f"Word count: {len(augmented['word_segments'])}",
        f"Speaker count: {len(speaker_labels(augmented))}",
        f"Wrote {output_path}",
    ])

    # Talk-time split is the cheapest diarization sanity check there is: a 97/3 split means
    # clustering collapsed, and the log says so before anyone opens the audio.
    readable_path = transcript_path(output_dir, stem)
    summary = write_readable_transcript(augmented, readable_path, stem)
    report(format_summary(stem, summary) + [f"Wrote {readable_path}"])


if __name__ == "__main__":
    main()
