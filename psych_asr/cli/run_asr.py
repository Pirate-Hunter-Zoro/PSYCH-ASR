"""Stage 1a: ASR -> forced alignment. No diarization, no speaker labels.

    python -m psych_asr.cli.run_asr <audio.wav>

Runs ONCE per session, in asr_env on a GPU. Every candidate diarizer fans off the aligned
transcript this writes, so that a DER difference between arms is a difference between
diarizers and not between transcripts.

Output: data/stage1/<stem>.aligned.json -- segments, word_segments and language, with NO
"speaker" keys anywhere. Stage 1c adds those.
"""

from argparse import ArgumentParser
from pathlib import Path

from .. import config
from ..artifacts.naming import aligned_path
from ..artifacts.transcripts import save_transcript
from ..asr.align import format_alignment_summary, load_audio, transcribe_and_align
from ._common import add_output_dir, prepare_output_dir, report


def build_parser():
    parser = ArgumentParser(description="Stage 1a: transcribe and force-align one WAV.")
    parser.add_argument("audio", type=str)
    add_output_dir(parser)
    parser.add_argument("--model-dir", type=str, default=str(config.WHISPER_MODEL_DIR),
                        help="staged faster-whisper-large-v3 directory (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    audio_path = Path(args.audio)
    output_dir = prepare_output_dir(args.outdir)

    decoded_audio = load_audio(audio_path)
    align_result, asr_segments = transcribe_and_align(
        decoded_audio, args.model_dir, batch_size=args.batch_size,
    )

    output_path = save_transcript(align_result, aligned_path(output_dir, audio_path.stem))
    report(format_alignment_summary(align_result, asr_segments) + [f"Wrote {output_path}"])


if __name__ == "__main__":
    main()
