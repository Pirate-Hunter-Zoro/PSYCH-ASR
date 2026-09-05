"""Argument-parser pieces every entry point shares.

STDLIB ONLY.

Six parsers used to re-declare --outdir, --num-speakers and --arm with their own help text
and their own hardcoded absolute model paths. The flags are declared once here, so a
default cannot drift between two jobs that are supposed to agree.
"""

from pathlib import Path

from .. import config


def add_output_dir(parser, default=config.STAGE1_DIR):
    """--outdir, defaulting to data/stage1. Session content is PHI and data/ is gitignored
    wholesale; nothing in Stage 1 may be written outside it."""
    parser.add_argument("--outdir", type=str, default=str(default),
                        help="where to write the artifacts (default: %(default)s)")


def add_num_speakers(parser, default=2, help_text=None):
    """--num-speakers. Every pilot session is a known dyad, so pinning the count removes
    both clustering failure modes -- splitting one person in two, merging two into one --
    for free. The flag exists rather than a hardcoded 2 in case a session turns out to have
    a third person present."""
    parser.add_argument(
        "--num-speakers", type=int, default=default,
        help=help_text or ("exact speaker count; every pilot session is a known dyad, and "
                           "pinning it removes both clustering failure modes for free"),
    )


def add_arm(parser, default=None, help_text=None):
    """--arm: the name written into every output filename from Stage 1b onward, so which
    model produced which transcript is a property of the file rather than of a note."""
    parser.add_argument("--arm", type=str, default=default,
                        help=help_text or "name used in the output filenames; identifies which "
                                          "diarizer produced them")


def prepare_output_dir(outdir):
    """IN: the --outdir string   OUT: the Path, created if it did not exist."""
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def report(lines):
    """IN: a list of log lines (or one string)   OUT: nothing; printed, flushed.

    Flushed because these are read out of a Slurm log while the job is still running, and
    an unflushed buffer makes a live job look hung.
    """
    if isinstance(lines, str):
        lines = [lines]
    print("\n".join(lines), flush=True)
