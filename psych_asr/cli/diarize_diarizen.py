"""Stage 1b, arm A: DiariZen -> RTTM.

    python -m psych_asr.cli.diarize_diarizen <audio.wav> --num-speakers 2   (pinned)
    python -m psych_asr.cli.diarize_diarizen <audio.wav> --num-speakers 0   (shipped config)

Audio in, RTTM out, same contract as every other 1b step. Runs in diarizen_env on a GPU.

BOTH WAYS ARE RUN AND REPORTED SEPARATELY, under the arm names "diarizen" and
"diarizen-free", so the two never overwrite each other. Pinning is a deliberate deviation
from the checkpoint's published configuration and it is the fair one -- the pyannote
baseline is already told there are exactly two people -- but hiding that the deviation
happened would be worse than not making it.

WEIGHTS ARE CC BY-NC 4.0. Fine for feasibility research at a nonprofit institute; not
shippable in a translation path.
"""

from argparse import ArgumentParser
from pathlib import Path

from .. import config
from ..artifacts.naming import rttm_path
from ..artifacts.rttm_io import format_turn_summary, read_rttm, summarize_turns, write_rttm_from_annotation
from ..diarize import diarizen_arm
from ._common import add_arm, add_num_speakers, add_output_dir, prepare_output_dir, report


def build_parser():
    parser = ArgumentParser(description="Stage 1b arm A: diarize one WAV with DiariZen.")
    parser.add_argument("audio", type=str)
    add_output_dir(parser)
    parser.add_argument("--model-dir", type=str, default=str(config.DIARIZEN_MODEL_DIR),
                        help="staged DiariZen hub directory (default: %(default)s)")
    parser.add_argument("--embedding-model", type=str, default=str(config.WESPEAKER_EMBEDDING_FILE),
                        help="WeSpeaker pytorch_model.bin, as a FILE path (default: %(default)s)")
    add_num_speakers(parser, help_text=(
        "exact speaker count, pinned through the AHC seeding. Pass 0 to leave the "
        "checkpoint's shipped configuration alone and let the clustering decide for itself"))
    add_arm(parser, help_text=("name used in the output filenames (default: 'diarizen' when "
                               "pinned, 'diarizen-free' when not, so the two runs never "
                               "overwrite each other)"))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    pinned = args.num_speakers > 0
    arm = args.arm or (config.ARM_DIARIZEN if pinned else f"{config.ARM_DIARIZEN}-free")

    audio_path = Path(args.audio)
    output_dir = prepare_output_dir(args.outdir)
    uri = audio_path.stem

    pipeline = diarizen_arm.load_pipeline(args.model_dir, args.embedding_model)
    if pinned:
        report(diarizen_arm.pin_speaker_count(pipeline, args.num_speakers))
    else:
        report("NOT pinned: the checkpoint's shipped clustering config decides the speaker "
               "count for itself.")

    annotation = diarizen_arm.diarize(pipeline, audio_path, uri)

    written = rttm_path(output_dir, uri, arm)
    num_turns = write_rttm_from_annotation(annotation, uri, written)
    report(f"Wrote {written} ({num_turns} turns)")

    turns = read_rttm(written)
    report(format_turn_summary(arm, uri, summarize_turns(turns)))

    warning = diarizen_arm.sentinel_warning(turns, args.num_speakers if pinned else None)
    if warning:
        report(warning)


if __name__ == "__main__":
    main()
