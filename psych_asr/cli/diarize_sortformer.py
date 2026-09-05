"""Stage 1b, arms B and C: NVIDIA Sortformer -> RTTM.

    python -m psych_asr.cli.diarize_sortformer <audio.wav> --mode offline     (arm B)
    python -m psych_asr.cli.diarize_sortformer <audio.wav> --mode streaming   (arm C)

Audio in, RTTM out, same contract as every other 1b step. Runs in nemo_env on a GPU. Both
checkpoints load through the same NeMo class, so one entry point covers both arms.

--num-speakers is ACCEPTED AND DELIBERATELY IGNORED, so the bake-off driver can pass the
same flags to every arm; the log says plainly that it was ignored. Sortformer is end-to-end
with a hard four-speaker ceiling and no clustering stage to constrain.
"""

from argparse import ArgumentParser
from pathlib import Path

from .. import config
from ..artifacts.naming import rttm_path
from ..artifacts.rttm_io import format_turn_summary, read_rttm, summarize_turns, write_rttm
from ..diarize import sortformer_arm
from ..diarize.windowing import MAX_SPEAKERS
from ._common import add_arm, add_num_speakers, add_output_dir, prepare_output_dir, report

MODES = {
    "offline": {"arm": config.ARM_SORTFORMER, "checkpoint": config.SORTFORMER_OFFLINE_CHECKPOINT},
    "streaming": {"arm": config.ARM_SORTFORMER_STREAMING, "checkpoint": config.SORTFORMER_STREAMING_CHECKPOINT},
}


def build_parser():
    parser = ArgumentParser(description="Stage 1b arms B/C: diarize one WAV with NVIDIA Sortformer.")
    parser.add_argument("audio", type=str)
    parser.add_argument("--mode", choices=sorted(MODES), default="offline",
                        help="offline = arm B (diar_sortformer_4spk-v1); "
                             "streaming = arm C (diar_streaming_sortformer_4spk-v2.1)")
    add_output_dir(parser)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="absolute path to the .nemo archive (default: the staged one for --mode)")
    add_arm(parser, help_text="name used in the output filenames (default: the arm for --mode)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--window-seconds", type=float, default=600.0,
                        help="offline mode only: window length. NVIDIA reports ~12 min as the "
                             "ceiling on a 48 GB card, so 10 min leaves headroom on the 46 GB A40")
    parser.add_argument("--overlap-seconds", type=float, default=60.0,
                        help="offline mode only: how much consecutive windows share. This is the "
                             "only evidence the stitcher has for deciding that two windows' "
                             "speaker indices refer to the same person")
    add_num_speakers(parser, help_text=(
        "accepted for flag compatibility with the pyannote arms and IGNORED: Sortformer is "
        "end-to-end with a 4-speaker ceiling and no clustering stage to constrain"))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    defaults = MODES[args.mode]
    arm = args.arm or defaults["arm"]
    checkpoint = args.checkpoint or defaults["checkpoint"]

    audio_path = Path(args.audio)
    output_dir = prepare_output_dir(args.outdir)
    uri = audio_path.stem

    report([
        f"Arm {arm} ({args.mode}) from {checkpoint}",
        f"NOTE: --num-speakers={args.num_speakers} is IGNORED; Sortformer cannot be pinned "
        f"to a speaker count. Whatever it emits, up to {MAX_SPEAKERS}, is the result.",
    ])

    model = sortformer_arm.load_model(checkpoint)

    if args.mode == "streaming":
        preset = sortformer_arm.apply_streaming_preset(model)
        report(f"Streaming preset: {preset}")
        # Arbitrary length by construction -- no windowing, no stitcher, no seam error.
        turns = sortformer_arm.diarize_whole(model, audio_path, args.batch_size)
    else:
        # Window WAVs are cut from session audio and are therefore PHI. They are written
        # under the (gitignored) output directory and removed before this returns.
        turns = sortformer_arm.diarize_windowed(
            model, audio_path, output_dir / f".{uri}.{arm}.windows",
            args.window_seconds, args.overlap_seconds, args.batch_size,
            log=lambda line: report(line),
        )

    written = rttm_path(output_dir, uri, arm)
    num_turns = write_rttm(turns, uri, written)
    report(f"Wrote {written} ({num_turns} turns)")
    report(format_turn_summary(arm, uri, summarize_turns(read_rttm(written))))


if __name__ == "__main__":
    main()
