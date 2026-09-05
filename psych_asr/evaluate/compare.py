"""Compare the bake-off arms word by word, with no model and no judgement.

STDLIB ONLY.

This is the step that has to happen BEFORE anything -- human or LLM -- reads four
transcripts looking for differences, and it makes most of that reading unnecessary.

THE STRUCTURAL FACT THIS EXPLOITS. Stage 1a runs once, so every arm is joined onto the
IDENTICAL word sequence with IDENTICAL word timings. The only thing that can differ
between two arms' transcripts is the speaker label attached to each word. "Compare four
50-minute transcripts" is therefore not a reading task at all; it is an exact diff over one
column, computable in a second.

WHAT IT DELIBERATELY DOES NOT DO. It does not say which arm is right. Agreement is not
accuracy: four arms can agree and all be wrong. Only the hand-corrected reference RTTM from
Stage 2 scores the arms. What this produces is the map of WHERE they disagree, which is what
makes both the human listening pass and any downstream LLM triage affordable.

THE REGION RECORDS CARRY VERBATIM TRANSCRIPT TEXT and are therefore PHI.
"""

from collections import Counter

from .labels import best_label_mapping

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


def order_arms(arms, baseline=DEFAULT_BASELINE):
    """IN: the arm names present + the preferred baseline   OUT: (ordered names, baseline).

    The baseline leads and everything else follows alphabetically, so two runs over the
    same arms print their columns in the same order. When the preferred baseline is absent
    -- its job crashed -- the alphabetically first arm stands in rather than the comparison
    refusing to run.
    """
    reference = baseline if baseline in arms else sorted(arms)[0]
    return [reference] + sorted(arm for arm in arms if arm != reference), reference


def require_identical_words(words_by_arm, reference_arm):
    """Enforce the premise. IN: dict arm -> word list + the reference arm   OUT: word count.

    Every arm must sit on the identical word sequence -- that is the whole basis for
    treating this as a one-column diff. If it does not, someone re-ran Stage 1a between
    arms and no label difference can be attributed to a diarizer.
    """
    reference_words = words_by_arm[reference_arm]
    for arm, words in words_by_arm.items():
        if words != reference_words:
            raise SystemExit(
                f"Arm '{arm}' has a different word sequence from '{reference_arm}' "
                f"({len(words)} vs {len(reference_words)} words). Every arm must be joined onto "
                f"the SAME Stage 1a output, or a label difference cannot be attributed to the diarizer."
            )
    return len(reference_words)


def canonicalize_labels(raw_labels, ordered_arms, reference_arm):
    """Remap every arm's labels onto the baseline's namespace.

    IN:  dict arm -> raw label column, the arm ordering, the reference arm
    OUT: (dict arm -> canonicalized label column, dict arm -> the mapping used)

    Without this step two arms that agree perfectly could score 0%.
    """
    labels_by_arm = {reference_arm: raw_labels[reference_arm]}
    mappings = {}
    for arm in ordered_arms:
        if arm == reference_arm:
            continue
        mapping = best_label_mapping(raw_labels[reference_arm], raw_labels[arm])
        mappings[arm] = mapping
        labels_by_arm[arm] = [
            mapping.get(label) if label is not None else None for label in raw_labels[arm]
        ]
    return labels_by_arm, mappings


def per_arm_shape(labels):
    """IN: one arm's canonicalized label column   OUT: its label count, unlabeled count,
    and each label's share of the arm's labeled words.

    Word share rather than duration: the words are identical across arms, so counting them
    compares the arms on exactly the same denominator.
    """
    counts = Counter(label for label in labels if label is not None)
    total = sum(counts.values())
    return {
        "distinct_labels": len(counts),
        "unlabeled_words": sum(1 for label in labels if label is None),
        "word_share": {label: count / total for label, count in counts.items()} if total else {},
    }


def pairwise_agreement(labels_by_arm, ordered_arms):
    """IN: canonicalized label columns + the arm ordering   OUT: nested dict of percentages.

    Computed over words BOTH arms labeled. Counting an unlabeled word as a disagreement
    would conflate "these two arms disagree about who spoke" with "one arm did not cover
    this stretch of audio," and those call for different fixes.
    """
    matrix = {}
    for left in ordered_arms:
        row = {}
        for right in ordered_arms:
            shared = [
                (a, b) for a, b in zip(labels_by_arm[left], labels_by_arm[right])
                if a is not None and b is not None
            ]
            row[right] = sum(1 for a, b in shared if a == b) / len(shared) * 100 if shared else float("nan")
        matrix[left] = row
    return matrix


def disagreement_runs(labels_by_arm, ordered_arms, num_words):
    """Find contiguous stretches where the arms do not all say the same thing.

    IN:  canonicalized label columns, the arm ordering, and the word count
    OUT: list of (first_index, last_index_exclusive) runs

    Words no arm labeled are not a disagreement -- every arm said the same nothing.
    """
    runs = []
    start = None
    for index in range(num_words):
        differs = len({labels_by_arm[arm][index] for arm in ordered_arms}) > 1
        if differs and start is None:
            start = index
        elif not differs and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, num_words))
    return runs


def build_region_records(runs, words, labels_by_arm, ordered_arms, start_time_of,
                         context_words=CONTEXT_WORDS):
    """Turn disagreement runs into the work queue for whoever adjudicates them.

    IN:  the runs to record, the shared word list, the canonicalized label columns, the arm
         ordering, and a callable index -> start seconds (or None)
    OUT: list of record dicts, each carrying the disputed span, its surrounding context,
         and what every arm said inside it

    EACH RECORD CARRIES VERBATIM TRANSCRIPT TEXT. This is the PHI-bearing half of the
    comparison, and it is why the whole artifact is written under data/.
    """
    num_words = len(words)
    records = []
    for start, end in runs:
        context_start = max(start - context_words, 0)
        context_end = min(end + context_words, num_words)
        records.append({
            "word_range": [start, end],
            "context_range": [context_start, context_end],
            "start_time": start_time_of(start),
            # whisperx stores each word WITHOUT a leading space, so joining on "" welds
            # the sentence into one unreadable token.
            "text": " ".join(words[context_start:context_end]).strip(),
            "disputed_text": " ".join(words[start:end]).strip(),
            "labels": {arm: labels_by_arm[arm][start:end] for arm in ordered_arms},
        })
    return records


def summarize_region(record, ordered_arms):
    """IN: one region record + the arm ordering   OUT: dict arm -> {label: word count}.

    A compact "who said what" for the region, used for printing and usable without reading
    the disputed text itself.
    """
    return {
        arm: dict(Counter(label or "UNLABELED" for label in record["labels"][arm]))
        for arm in ordered_arms
    }
