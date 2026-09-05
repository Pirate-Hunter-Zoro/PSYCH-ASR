"""QC: bucket every unlabeled word by which of the three causes produced it. NOT YET BUILT.

    python -m psych_asr.cli.audit_speakers <stem>.<arm>.diarized.json <stem>.<arm>.rttm

An unlabeled word looks like a diarization failure and usually is not. Reading
whisperx/diarize.py and whisperx/alignment.py in the installed 3.8.6, there are exactly
three ways "speaker" fails to appear on a word, and they call for three different fixes:

1. THE WORD HAS NO "start". assign_word_speakers opens its per-word loop with a skip: no
   start key, continue, never queried against the turn table at all. Alignment produces
   such a word when none of its characters received a timestamp -- digits, symbols and
   foreign script go through the wildcard emission column and can come back NaN -- AND the
   sentence-level interpolation fallback could not fill it. An ALIGNMENT artifact.
2. THE WORD HAS ZERO DURATION. The overlap test requires the intersection of word and turn
   to be strictly greater than zero, so a word whose start and end round to the same
   millisecond matches nothing even when it sits squarely inside a speaker turn. Also an
   alignment artifact, and invisible unless you look for it.
3. THE WORD IS TIMESTAMPED, REAL, AND OVERLAPS NO TURN. Only this one is a genuine
   diarization-coverage story.

A fourth case hides from a naive count entirely: a segment that fails alignment outright is
appended with an EMPTY words list, so its words never exist to be counted as missing.

WHY THIS IS STILL A STUB. Cause 3 cannot be confirmed from the .diarized.json alone -- it
needs the turn table, which the bake-off's RTTM now provides. The remaining work is the
cross-tab against whether the parent SEGMENT got a speaker, which is what distinguishes
micro-gaps between turns from a genuinely uncovered stretch of audio. It is tracked in
Research-Journey/planning/PSYCH-ASR_TODO.txt, not here.

The parser below is complete so the intended interface is on record; running it says
plainly that the analysis is not implemented rather than printing an empty result.
"""

from argparse import ArgumentParser

from ._common import report


def build_parser():
    parser = ArgumentParser(description="Bucket unlabeled words by cause (not yet implemented).")
    parser.add_argument("diarized", type=str, help="path to a <stem>.<arm>.diarized.json")
    parser.add_argument("rttm", type=str,
                        help="that arm's turn table -- required, because cause 3 cannot be "
                             "distinguished from causes 1 and 2 without it")
    return parser


def main(argv=None):
    build_parser().parse_args(argv)
    report("audit_speakers is not implemented yet. See the module docstring for the three "
           "causes it is meant to separate and what is left to build.")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
