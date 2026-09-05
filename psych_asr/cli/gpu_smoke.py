"""Smallest end-to-end check that the GPU, torch and ctranslate2 all work together.

    python -m psych_asr.cli.gpu_smoke <a short .wav>

Loads faster-whisper-tiny in float16 on the GPU and transcribes. Deliberately the TINY
model: this answers "does the CUDA stack work at all," and paying for large-v3 to learn
that wastes several minutes of an accelerator node.

The job that drives it synthesizes a 3-second sine tone, so it needs no session audio.
"""

from argparse import ArgumentParser

import torch
from faster_whisper import WhisperModel

from .. import config

SMOKE_MODEL_DIR = config.MODELS_ROOT / "faster-whisper-tiny"


def build_parser():
    parser = ArgumentParser(description="GPU / ctranslate2 sanity check.")
    parser.add_argument("audio", type=str, help="any short WAV; the job synthesizes a sine tone")
    parser.add_argument("--model-dir", type=str, default=str(SMOKE_MODEL_DIR),
                        help="staged faster-whisper-tiny directory (default: %(default)s)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    print(torch.__version__)
    print(torch.cuda.get_device_name(device=0))

    model = WhisperModel(args.model_dir, device="cuda", compute_type="float16")
    segments, info = model.transcribe(beam_size=5, audio=args.audio)
    print(info.language)
    for segment in segments:
        print(segment.text)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
