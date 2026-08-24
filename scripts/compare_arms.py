"""Compare the bake-off arms word by word, with no model and no judgement.

This is the step that has to happen BEFORE anything -- human or LLM -- reads four
transcripts looking for differences, and it makes most of that reading unnecessary.

THE STRUCTURAL FACT THIS EXPLOITS. Stage 1a runs once, so every arm is joined onto the
IDENTICAL word sequence with IDENTICAL word timings. The only thing that can differ
between two arms' transcripts is the speaker label attached to each word. "Compare four
50-minute transcripts" is therefore not a reading task at all; it is an exact diff over one
column, computable in a second.

WHAT IT DOES ABOUT ARBITRARY LABELS. SPEAKER_00 in one arm and speaker_1 in another may be
the same person -- diarizers have no idea who anyone is, and every arm numbers its clusters
independently. Each arm's labels are therefore remapped onto the baseline's namespace by
whichever one-to-one pairing maximizes word-level agreement, before any agreement number is
computed. Without that step two arms that agree perfectly could score 0%.

WHAT IT DELIBERATELY DOES NOT DO. It does not say which arm is right. Agreement is not
accuracy: four arms can agree and all be wrong. Only the hand-corrected reference RTTM from
Stage 2 scores the arms. What this produces is the map of WHERE they disagree, which is what
makes both the human listening pass and any downstream LLM triage affordable.

OUTPUT is PHI (it carries transcript text) and is written under data/stage1/, which is
gitignored wholesale.

CPU only, runs in asr_env, loads no models.
"""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path
import json

# Sibling module in scripts/; `python scripts/compare_arms.py` puts that dir on sys.path[0].
from render_transcript import format_timestamp

# The arm every other arm's labels are mapped onto, when it is present. It is the incumbent
# and the regression reference, so its namespace is the one already written up.
DEFAULT_BASELINE = "community-1"

# Words on either side of a disagreement run, carried into the report so a run is readable
# as dialogue rather than as a list of tokens. Also the unit a later LLM triage pass would
# be handed -- a speaker label is uncodeable without the surrounding exchange.
CONTEXT_WORDS = 25

# Runs shorter than this are single-word boundary jitter at a speaker change, which is what
# a diarizer disagreement looks like when nothing interesting happened. Counted, but not
# listed individually.
MIN_INTERESTING_RUN = 3


def load_arms(stage1_dir, stem):
    """IN: directory + file stem   OUT: dict of arm name -> loaded transcript dict.

    Discovers arms from the filenames rather than a hardcoded list, so an arm that failed
    is simply absent instead of raising.
    """
    arms = {}
    for path in sorted(stage1_dir.glob(f"{stem}.*.diarized.json")):
        # "<stem>.<arm>.diarized.json" -> "<arm>"
        arm = path.name[len(stem) + 1: -len(".diarized.json")]
        with open(path) as f:
            arms[arm] = json.load(f)
    return arms


def word_labels(transcript):
    """IN: a .diarized.json dict   OUT: (list of word strings, list of speaker labels).

    A word with no "speaker" key becomes None rather than being skipped, because WHICH
    words an arm declined to label is itself one of the differences worth seeing.
    """
    words, labels = [], []
    for word in transcript.get("word_segments", []):
        words.append(word.get("word", ""))
        labels.append(word.get("speaker"))
    return words, labels


def best_label_mapping(reference_labels, candidate_labels):
    """Find the relabeling of one arm that agrees most with another.

    IN:  two equal-length lists of per-word labels (either may hold None)
    OUT: dict mapping candidate label -> reference label

    Solved by brute force over permutations rather than with a linear-assignment solver:
    every diarizer here is capped at four speakers, so the search is at most a few hundred
    pairings, and it costs asr_env no new dependency.
    """
    reference_names = sorted({label for label in reference_labels if label is not None})
    candidate_names = sorted({label for label in candidate_labels if label is not None})
    if not candidate_names:
        return {}

    agreement = Counter()
    for left, right in zip(reference_labels, candidate_labels):
        if left is not None and right is not None:
            agreement[(right, left)] += 1

    # Pad the shorter side so every candidate label gets somewhere to go, even a name the
    # reference never used -- an arm that found a third speaker must not have it silently
    # folded onto one of the baseline's two.
    targets = list(reference_names)
    for index in range(len(candidate_names) - len(reference_names)):
        targets.append(f"EXTRA_{index}")

    best_mapping, best_score = {}, -1
    for assignment in permutations(targets, len(candidate_names)):
        mapping = dict(zip(candidate_names, assignment))
        score = sum(agreement[(source, target)] for source, target in mapping.items())
        if score > best_score:
            best_mapping, best_score = mapping, score
    return best_mapping


def disagreement_runs(labels_by_arm, arms, num_words):
    """Find contiguous stretches where the arms do not all say the same thing.

    IN:  dict arm -> canonicalized label list, the arm ordering, and the word count
    OUT: list of (first_index, last_index_exclusive) runs

    Words no arm labeled are not a disagreement -- every arm said the same nothing.
    """
    runs = []
    start = None
    for index in range(num_words):
        values = {labels_by_arm[arm][index] for arm in arms}
        differs = len(values) > 1
        if differs and start is None:
            start = index
        elif not differs and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, num_words))
    return runs


def word_time(transcript, index):
    """IN: transcript dict + word index   OUT: start time in seconds, or None.

    Alignment leaves "start" off a word whose characters never received a timestamp, so
    this is a tolerant lookup that walks backwards to the nearest timed word.
    """
    words = transcript.get("word_segments", [])
    for probe in range(index, max(index - 50, -1), -1):
        if "start" in words[probe]:
            return words[probe]["start"]
    return None


def main():
    parser = ArgumentParser(description="Word-level diff of every diarization arm's transcript.")
    parser.add_argument("--stage1-dir", type=str, default="data/stage1")
    parser.add_argument("--stem", type=str, default=None,
                        help="session stem (default: inferred from the single .aligned.json present)")
    parser.add_argument("--baseline", type=str, default=DEFAULT_BASELINE,
                        help="arm whose speaker names every other arm is mapped onto")
    parser.add_argument("--max-runs-listed", type=int, default=25,
                        help="how many disagreement regions to print in full; all of them are written to the JSON")
    args = parser.parse_args()

    stage1_dir = Path(args.stage1_dir)

    stem = args.stem
    if stem is None:
        aligned = sorted(stage1_dir.glob("*.aligned.json"))
        if len(aligned) != 1:
            raise SystemExit(f"Found {len(aligned)} .aligned.json files in {stage1_dir}; pass --stem explicitly.")
        stem = aligned[0].name[: -len(".aligned.json")]

    arms = load_arms(stage1_dir, stem)
    if len(arms) < 2:
        raise SystemExit(f"Need at least two arms to compare; found {sorted(arms)} for stem '{stem}'.")

    # Every arm must sit on the identical word sequence -- that is the whole premise. If it
    # does not, someone re-ran Stage 1a between arms and the comparison is meaningless.
    words_by_arm, raw_labels = {}, {}
    for arm, transcript in arms.items():
        words_by_arm[arm], raw_labels[arm] = word_labels(transcript)
    reference_arm = args.baseline if args.baseline in arms else sorted(arms)[0]
    reference_words = words_by_arm[reference_arm]
    for arm, words in words_by_arm.items():
        if words != reference_words:
            raise SystemExit(
                f"Arm '{arm}' has a different word sequence from '{reference_arm}' "
                f"({len(words)} vs {len(reference_words)} words). Every arm must be joined onto "
                f"the SAME Stage 1a output, or a label difference cannot be attributed to the diarizer."
            )
    num_words = len(reference_words)

    ordered_arms = [reference_arm] + sorted(arm for arm in arms if arm != reference_arm)

    rule = "=" * 78
    print(rule)
    print(f"DIARIZATION ARM COMPARISON — {stem}")
    print(rule)
    print(f"Words in common : {num_words}")
    print(f"Baseline arm    : {reference_arm}")
    print("")

    # ---- canonicalize every arm's labels onto the baseline's namespace ----
    labels_by_arm = {reference_arm: raw_labels[reference_arm]}
    mappings = {}
    for arm in ordered_arms[1:]:
        mapping = best_label_mapping(raw_labels[reference_arm], raw_labels[arm])
        mappings[arm] = mapping
        labels_by_arm[arm] = [mapping.get(label) if label is not None else None for label in raw_labels[arm]]
        rendered = ", ".join(f"{source} -> {target}" for source, target in sorted(mapping.items()))
        print(f"relabel {arm:<24}: {rendered}")

    # ---- per-arm shape ----
    print("")
    print(f"{'arm':<24}{'labels':>8}{'unlabeled':>11}   talk-time split")
    per_arm_summary = {}
    for arm in ordered_arms:
        labels = labels_by_arm[arm]
        counts = Counter(label for label in labels if label is not None)
        total = sum(counts.values())
        split = "  ".join(
            f"{label}:{count / total * 100:.0f}%" for label, count in sorted(counts.items(), key=lambda kv: -kv[1])
        ) if total else "(none)"
        unlabeled = sum(1 for label in labels if label is None)
        per_arm_summary[arm] = {
            "distinct_labels": len(counts),
            "unlabeled_words": unlabeled,
            "word_share": {label: count / total for label, count in counts.items()} if total else {},
        }
        print(f"{arm:<24}{len(counts):>8}{unlabeled:>11}   {split}")

    # ---- pairwise agreement ----
    # Computed over words BOTH arms labeled. Counting an unlabeled word as a disagreement
    # would conflate "these two arms disagree about who spoke" with "one arm did not cover
    # this stretch of audio," and those call for different fixes.
    print("")
    print("pairwise word-level agreement (over words both arms labeled)")
    print(f"{'':<24}" + "".join(f"{arm[:11]:>13}" for arm in ordered_arms))
    pairwise = defaultdict(dict)
    for left in ordered_arms:
        row = f"{left:<24}"
        for right in ordered_arms:
            shared = [
                (a, b) for a, b in zip(labels_by_arm[left], labels_by_arm[right])
                if a is not None and b is not None
            ]
            agree = sum(1 for a, b in shared if a == b) / len(shared) * 100 if shared else float("nan")
            pairwise[left][right] = agree
            row += f"{agree:>12.1f}%"
        print(row)

    # ---- where they disagree ----
    runs = disagreement_runs(labels_by_arm, ordered_arms, num_words)
    interesting = [run for run in runs if run[1] - run[0] >= MIN_INTERESTING_RUN]
    disputed_words = sum(end - start for start, end in runs)

    print("")
    print(f"Disagreement regions      : {len(runs)}  ({disputed_words} words, "
          f"{disputed_words / num_words * 100:.1f}% of the transcript)")
    print(f"Of those, >= {MIN_INTERESTING_RUN} words long : {len(interesting)}  "
          f"(shorter runs are boundary jitter at a speaker change)")
    print(rule)

    baseline_transcript = arms[reference_arm]
    records = []
    for start, end in interesting:
        context_start = max(start - CONTEXT_WORDS, 0)
        context_end = min(end + CONTEXT_WORDS, num_words)
        records.append({
            "word_range": [start, end],
            "context_range": [context_start, context_end],
            "start_time": word_time(baseline_transcript, start),
            # whisperx stores each word WITHOUT a leading space, so joining on "" welds
            # the sentence into one unreadable token.
            "text": " ".join(reference_words[context_start:context_end]).strip(),
            "disputed_text": " ".join(reference_words[start:end]).strip(),
            "labels": {arm: labels_by_arm[arm][start:end] for arm in ordered_arms},
        })

    for record in sorted(records, key=lambda r: -(r["word_range"][1] - r["word_range"][0]))[: args.max_runs_listed]:
        start, end = record["word_range"]
        stamp = format_timestamp(record["start_time"]) if record["start_time"] is not None else "--:--"
        print(f"[{stamp}] words {start}-{end} ({end - start})")
        for arm in ordered_arms:
            said = Counter(label or "UNLABELED" for label in record["labels"][arm])
            print(f"    {arm:<24}{'  '.join(f'{k}x{v}' for k, v in said.most_common())}")
        print(f"    \"{record['disputed_text']}\"")
        print("")

    output_path = stage1_dir / f"{stem}.arm_comparison.json"
    with open(output_path, "w") as f:
        json.dump({
            "stem": stem,
            "baseline_arm": reference_arm,
            "arms": ordered_arms,
            "num_words": num_words,
            "label_mappings": mappings,
            "per_arm": per_arm_summary,
            "pairwise_agreement": {left: dict(right) for left, right in pairwise.items()},
            "num_disagreement_regions": len(runs),
            "disputed_words": disputed_words,
            "regions": records,
        }, f, indent=2, ensure_ascii=False)
    print(f"Wrote {output_path}  ({len(records)} regions with context, for downstream triage)")


if __name__ == "__main__":
    main()
