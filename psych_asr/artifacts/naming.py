"""The Stage 1 filename convention, in one place.

STDLIB ONLY.

Every Stage 1 artifact is named "<stem>.<arm>.<kind>", where stem is the audio file's
stem and arm is the diarizer that produced it. THE ARM NAME IS CARRIED IN THE FILENAME
FROM STAGE 1b ONWARD, so which model produced which transcript is a property of the file
rather than of a note somewhere -- and Stage 1c derives the arm from the RTTM's own name,
which is why adding a fifth arm needs no change to the join.

    <stem>.aligned.json                 Stage 1a, no speaker keys, ONE per session
    <stem>.<arm>.rttm                   Stage 1b, that arm's speaker turn table
    <stem>.<arm>.exclusive.rttm         Stage 1b baseline only, the overlap-free view
    <stem>.<arm>.diarized.json          Stage 1c, the machine artifact
    <stem>.<arm>.transcript.txt         Stage 1c, the readable copy
    <stem>.arm_comparison.json          the word-level diff between arms (PHI: carries text)
    <stem>.arm_scores.json              DER and the therapy measures (numbers only)

Four scripts each used to re-derive these suffixes with their own slicing, and they
disagreed about edge cases -- one inferred an arm by splitting on ".", which turns the arm
"community-1" into "community-1" only by luck and would have broken on any arm name with a
dot in it. The parsing is here now, and it is done by exact suffix rather than by split.
"""

from pathlib import Path

ALIGNED_SUFFIX = ".aligned.json"
DIARIZED_SUFFIX = ".diarized.json"
TRANSCRIPT_SUFFIX = ".transcript.txt"
RTTM_SUFFIX = ".rttm"
EXCLUSIVE_RTTM_SUFFIX = ".exclusive.rttm"
COMPARISON_SUFFIX = ".arm_comparison.json"
SCORES_SUFFIX = ".arm_scores.json"


def _strip_suffix(name, suffix):
    """IN: filename + expected suffix  OUT: the name without it, or the name unchanged."""
    return name[: -len(suffix)] if name.endswith(suffix) else name


def stem_from_aligned(path):
    """IN: path to <stem>.aligned.json   OUT: "<stem>"."""
    return _strip_suffix(Path(path).name, ALIGNED_SUFFIX)


def stem_from_diarized(path):
    """IN: path to <stem>.diarized.json or <stem>.<arm>.diarized.json   OUT: what precedes
    the suffix, arm included. Callers that want the bare stem pass the stem in."""
    return _strip_suffix(Path(path).name, DIARIZED_SUFFIX)


def arm_from(path, stem, suffix):
    """IN: an artifact path, the session stem, and the artifact's suffix   OUT: the arm name.

    "<stem>.<arm><suffix>" -> "<arm>". A file that does not start with "<stem>." keeps its
    whole basename as the arm, which is what makes a hand-placed RTTM with an unexpected
    name still join rather than raise -- it just labels itself oddly, visibly.
    """
    base = _strip_suffix(Path(path).name, suffix)
    prefix = stem + "."
    return base[len(prefix):] if base.startswith(prefix) else base


def aligned_path(directory, stem):
    """IN: output directory + stem   OUT: Path to <stem>.aligned.json."""
    return Path(directory) / f"{stem}{ALIGNED_SUFFIX}"


def rttm_path(directory, stem, arm):
    """IN: output directory + stem + arm   OUT: Path to <stem>.<arm>.rttm."""
    return Path(directory) / f"{stem}.{arm}{RTTM_SUFFIX}"


def exclusive_rttm_path(directory, stem, arm):
    """IN: output directory + stem + arm   OUT: Path to <stem>.<arm>.exclusive.rttm."""
    return Path(directory) / f"{stem}.{arm}{EXCLUSIVE_RTTM_SUFFIX}"


def diarized_path(directory, stem, arm=None):
    """IN: output directory + stem + optional arm   OUT: Path to the machine artifact.

    With an arm: <stem>.<arm>.diarized.json, what Stage 1c writes. Without one:
    <stem>.diarized.json, the un-armed name the SINGLE-JOB Stage 1 has always written. The
    regression gate identifies the fixture by exactly that absence of an arm, so the two
    names are not interchangeable.
    """
    name = f"{stem}.{arm}" if arm else stem
    return Path(directory) / f"{name}{DIARIZED_SUFFIX}"


def transcript_path(directory, stem, arm=None):
    """IN: output directory + stem + optional arm   OUT: Path to the readable .txt.

    The single-job Stage 1 path writes <stem>.transcript.txt with no arm in it; the
    bake-off path writes one per arm. Both name it here.
    """
    name = f"{stem}.{arm}" if arm else stem
    return Path(directory) / f"{name}{TRANSCRIPT_SUFFIX}"


def comparison_path(directory, stem):
    """IN: output directory + stem   OUT: Path to <stem>.arm_comparison.json.

    THIS ARTIFACT CARRIES VERBATIM TRANSCRIPT TEXT and is therefore PHI. It is written
    under data/, which is gitignored wholesale, and the assistant's read guard refuses it.
    """
    return Path(directory) / f"{stem}{COMPARISON_SUFFIX}"


def scores_path(directory, stem):
    """IN: output directory + stem   OUT: Path to <stem>.arm_scores.json.

    Metrics only, no transcript text -- which is why this one file inside data/ is
    deliberately left readable by the assistant's guard.
    """
    return Path(directory) / f"{stem}{SCORES_SUFFIX}"


def find_sole_stem(stage1_dir):
    """IN: the Stage 1 directory   OUT: the one session stem present, by its aligned JSON.

    Raises SystemExit naming the count when it is not exactly one, because "which session"
    is not something to guess at when the answer decides which files get overwritten.
    """
    aligned = sorted(Path(stage1_dir).glob(f"*{ALIGNED_SUFFIX}"))
    if len(aligned) != 1:
        raise SystemExit(
            f"Found {len(aligned)} {ALIGNED_SUFFIX} files in {stage1_dir}; pass --stem explicitly."
        )
    return stem_from_aligned(aligned[0])


def find_arm_rttms(stage1_dir, stem):
    """IN: Stage 1 directory + stem   OUT: sorted list of (arm, Path) for every arm's RTTM.

    The .exclusive.rttm is skipped: it is the baseline's overlap-free view, a diagnostic on
    an existing arm rather than an arm of its own. Discovery is by glob rather than by a
    hardcoded arm list, so an arm whose job crashed is simply absent -- which is a visible
    result rather than a raised exception.
    """
    found = []
    for path in sorted(Path(stage1_dir).glob(f"{stem}.*{RTTM_SUFFIX}")):
        if path.name.endswith(EXCLUSIVE_RTTM_SUFFIX):
            continue
        found.append((arm_from(path, stem, RTTM_SUFFIX), path))
    return found


def find_arm_transcripts(stage1_dir, stem):
    """IN: Stage 1 directory + stem   OUT: sorted list of (arm, Path) for every joined JSON.

    Same discovery-by-glob reasoning as find_arm_rttms.
    """
    return [
        (arm_from(path, stem, DIARIZED_SUFFIX), path)
        for path in sorted(Path(stage1_dir).glob(f"{stem}.*{DIARIZED_SUFFIX}"))
    ]
