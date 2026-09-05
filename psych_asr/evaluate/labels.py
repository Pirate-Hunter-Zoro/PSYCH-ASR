"""Aligning one system's arbitrary speaker labels onto another's.

STDLIB ONLY.

SPEAKER_00 in one arm and speaker_1 in another may be the same person: diarizers have no
idea who anyone is, and every arm numbers its clusters independently. ANY comparison that
is not itself a DER -- word-level agreement, the talk-time ratio, the backchannel check --
has to align the label sets first, or two systems that agree perfectly score zero.

Two mappers live here because they have genuinely different inputs. The word-level one
compares two positionally-aligned label columns and is exact; the annotation-level one
compares two interval sets and is greedy on co-occurrence duration. DER does its own
optimal mapping internally and uses neither.
"""

from collections import Counter
from itertools import permutations


def best_label_mapping(reference_labels, candidate_labels):
    """Find the relabeling of one arm's word column that agrees most with another's.

    IN:  two equal-length lists of per-word labels (either may hold None)
    OUT: dict mapping candidate label -> reference label

    Solved by brute force over permutations rather than with a linear-assignment solver:
    every diarizer here is capped at four speakers, so the search is at most a few hundred
    pairings, and it costs asr_env no new dependency.

    The shorter side is PADDED with placeholder targets so every candidate label gets
    somewhere to go, even a name the reference never used -- an arm that found a third
    speaker must not have it silently folded onto one of the baseline's two.
    """
    reference_names = sorted({label for label in reference_labels if label is not None})
    candidate_names = sorted({label for label in candidate_labels if label is not None})
    if not candidate_names:
        return {}

    agreement = Counter()
    for left, right in zip(reference_labels, candidate_labels):
        if left is not None and right is not None:
            agreement[(right, left)] += 1

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


def map_labels_to_reference(reference, hypothesis):
    """Find which hypothesis speaker is which reference speaker, from two Annotations.

    IN:  two pyannote Annotations   OUT: dict hypothesis label -> reference label

    Greedy on co-occurrence duration, which is unambiguous at the two-or-three speaker
    scale here: take the highest-overlap pair, then the highest remaining pair whose two
    labels are both still free, and so on.

    Duck-typed on itertracks rather than imported from pyannote, so this module stays
    stdlib-only and testable without a diarization stack.
    """
    overlaps = {}
    for hypothesis_turn, _, hypothesis_label in hypothesis.itertracks(yield_label=True):
        for reference_turn, _, reference_label in reference.itertracks(yield_label=True):
            shared = min(hypothesis_turn.end, reference_turn.end) - max(hypothesis_turn.start, reference_turn.start)
            if shared > 0:
                key = (hypothesis_label, reference_label)
                overlaps[key] = overlaps.get(key, 0.0) + shared

    mapping, taken = {}, set()
    for (hypothesis_label, reference_label), _ in sorted(overlaps.items(), key=lambda item: -item[1]):
        if hypothesis_label not in mapping and reference_label not in taken:
            mapping[hypothesis_label] = reference_label
            taken.add(reference_label)
    return mapping


def dominant_speaker(annotation, start, end):
    """IN: an Annotation + a time span   OUT: the label covering most of it, or None.

    The same intersection-duration argmax whisperx.assign_word_speakers uses for words, so
    the attribution question is asked exactly the way the pipeline answers it.
    """
    best_label, best_overlap = None, 0.0
    for turn, _, label in annotation.itertracks(yield_label=True):
        overlap = min(turn.end, end) - max(turn.start, start)
        if overlap > best_overlap:
            best_label, best_overlap = label, overlap
    return best_label
