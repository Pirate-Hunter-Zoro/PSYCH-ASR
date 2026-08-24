"""Score every diarization arm against a hand-corrected reference RTTM.

One scorer, one implementation, every arm -- run from diar_eval_env, which holds
pyannote.metrics and deliberately no torch. Folding this into asr_env would risk bumping
pyannote.core underneath the locked whisperx pin set, and it would quietly couple "how we
measure" to "what we measure with."

THIS SCRIPT CANNOT BE RUN UNTIL THE REFERENCE EXISTS. The reference is the hand-corrected
RTTM from Stage 2's stratified subset -- the same subset already planned for WER/DER
estimation, used twice rather than built twice. Scoring one arm against another measures
divergence, not accuracy: four arms can agree and all be wrong.

WHAT IS REPORTED, AND WHY EACH ONE:

  * DER AT COLLAR 0.25 s AND COLLAR 0 s, OVERLAP INCLUDED. A collar forgives boundary error
    at every speaker change, and the same system looks far worse without one -- on this
    corpus the two differ by roughly a factor of two. Reporting one number without saying
    which collar produced it is how published DERs become incomparable. Overlap is included
    because excluding it would discard the exact regime the challengers exist to handle.
  * THE DECOMPOSITION into missed speech, false alarm and speaker confusion. Published
    benchmarking finds missed speech dominates, and the decomposition is what says which
    knob to turn -- a false-alarm problem and a confusion problem call for opposite fixes.
  * TALK-TIME RATIO ERROR. DER is a duration-weighted average and can look acceptable while
    failing exactly where this project cares. Talk-time share is the first Stage 3a
    structural feature and an input to therapist/patient role assignment, so an arm that
    gets DER right and the ratio wrong is useless here.
  * BACKCHANNEL ATTRIBUTION ACCURACY. Therapist backchannels over patient speech are
    constant in this corpus, they are the overlap the challengers claim to model, and they
    are short enough to vanish inside a duration-weighted average. Whether they land on the
    right person is a separate question from DER and has to be asked separately.

A UEM MATTERS AND IS NOT OPTIONAL FOR A REAL NUMBER. Without one pyannote.metrics warns that
it approximated the evaluation region as the union of reference and hypothesis extents --
which lets an arm be scored over stretches the reference never annotated. Pass --uem once the
hand-corrected subset defines its own boundaries; the warning is the reminder.
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

# Whole turns that are nothing but one of these are listener signals, not content. Kept
# deliberately short and high-frequency: the point is to catch the "mm-hm" over the
# patient's narrative, not to build a discourse-marker lexicon.
BACKCHANNEL_TOKENS = {
    "mm-hm", "mhm", "mm", "uh-huh", "uhhuh", "hmm", "mm-hmm",
    "okay", "ok", "right", "yeah", "yep", "yes", "sure", "gotcha", "got it",
}

COLLARS = (0.25, 0.0)


def normalize(text):
    """IN: a turn's text  OUT: lowercased, stripped of surrounding punctuation."""
    return text.strip().strip(".,!?;:-—\"'").lower()


def talk_time_shares(annotation):
    """IN: a pyannote Annotation  OUT: dict speaker -> share of total speaker time.

    Uses each speaker's own summed turn duration over the sum across speakers, so silence
    does not dilute the split -- the same denominator the transcript renderer uses.
    """
    per_speaker = {label: annotation.label_duration(label) for label in annotation.labels()}
    total = sum(per_speaker.values())
    return {label: duration / total for label, duration in per_speaker.items()} if total else {}


def dominant_speaker(annotation, start, end):
    """IN: Annotation + a time span  OUT: the label covering most of it, or None.

    Same intersection-duration argmax that whisperx.assign_word_speakers uses for words, so
    the attribution question is asked the same way the pipeline answers it.
    """
    best_label, best_overlap = None, 0.0
    for turn, _, label in annotation.itertracks(yield_label=True):
        overlap = min(turn.end, end) - max(turn.start, start)
        if overlap > best_overlap:
            best_label, best_overlap = label, overlap
    return best_label


def map_labels_to_reference(reference, hypothesis):
    """Find which hypothesis speaker is which reference speaker.

    IN:  two Annotations   OUT: dict hypothesis label -> reference label

    Speaker labels are arbitrary per system, so any comparison that is not itself a DER --
    the talk-time ratio, the backchannel check -- has to align them first. Greedy on
    co-occurrence duration, which is unambiguous at the two-or-three speaker scale here;
    DER does its own optimal mapping internally and does not use this.
    """
    overlaps = {}
    for hypothesis_turn, _, hypothesis_label in hypothesis.itertracks(yield_label=True):
        for reference_turn, _, reference_label in reference.itertracks(yield_label=True):
            shared = min(hypothesis_turn.end, reference_turn.end) - max(hypothesis_turn.start, reference_turn.start)
            if shared > 0:
                overlaps[(hypothesis_label, reference_label)] = overlaps.get((hypothesis_label, reference_label), 0.0) + shared

    mapping, taken = {}, set()
    for (hypothesis_label, reference_label), _ in sorted(overlaps.items(), key=lambda item: -item[1]):
        if hypothesis_label not in mapping and reference_label not in taken:
            mapping[hypothesis_label] = reference_label
            taken.add(reference_label)
    return mapping


def backchannel_turns(transcript_path):
    """IN: path to a <stem>.<arm>.diarized.json  OUT: list of (start, end) spans.

    A backchannel is a WHOLE turn -- a run of consecutive same-speaker segments -- whose
    entire text is one listener token. Taking it at segment level instead would catch every
    "right" that opens a sentence, which is not the same thing at all.
    """
    with open(transcript_path) as f:
        transcript = json.load(f)

    turns, spans = [], []
    for segment in transcript.get("segments", []):
        speaker = segment.get("speaker")
        text = segment.get("text", "").strip()
        if not text:
            continue
        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["text"] += " " + text
            turns[-1]["end"] = segment["end"]
        else:
            turns.append({"speaker": speaker, "start": segment["start"], "end": segment["end"], "text": text})

    for turn in turns:
        if normalize(turn["text"]) in BACKCHANNEL_TOKENS:
            spans.append((turn["start"], turn["end"]))
    return spans


def main():
    parser = ArgumentParser(description="Score every diarization arm against a reference RTTM.")
    parser.add_argument("reference", type=str, help="hand-corrected reference RTTM from Stage 2")
    parser.add_argument("--stage1-dir", type=str, default="data/stage1")
    parser.add_argument("--stem", type=str, default=None,
                        help="session stem (default: inferred from the reference filename)")
    parser.add_argument("--backchannel-source", type=str, default="community-1",
                        help="which arm's joined transcript defines the backchannel spans. ONE arm "
                             "defines them for ALL arms, so every arm is scored on the same set; "
                             "letting each arm nominate its own would change the denominator per arm "
                             "and make the percentages incomparable")
    parser.add_argument("--uem", type=str, default=None,
                        help="evaluation-region UEM. Without it pyannote.metrics approximates the "
                             "region as the union of reference and hypothesis extents, which scores "
                             "arms over stretches the reference never annotated")
    args = parser.parse_args()

    reference_path = Path(args.reference)
    stage1_dir = Path(args.stage1_dir)
    stem = args.stem or reference_path.name.split(".")[0]

    references = load_rttm(str(reference_path))
    if stem not in references:
        raise SystemExit(f"{reference_path} holds no annotation for uri '{stem}' (found {sorted(references)}).")
    reference = references[stem]

    uem = None
    if args.uem:
        from pyannote.database.util import load_uem
        uem = load_uem(args.uem)[stem]
    else:
        print("WARNING: no --uem given. DER is being computed over the union of reference and\n"
              "         hypothesis extents, so an arm is scored on audio the reference never\n"
              "         annotated. Fine for a smoke test, NOT for a reported number.\n")

    # ONE backchannel span set for every arm. The spans come from a transcript rather than
    # from the reference RTTM because identifying a backchannel needs the WORDS, and an RTTM
    # carries none -- it is turns and speaker labels only. Which arm supplies them barely
    # matters (the arms agree on >99% of words); that it is the SAME arm for all of them is
    # what makes the resulting percentages comparable.
    backchannel_path = stage1_dir / f"{stem}.{args.backchannel_source}.diarized.json"
    if backchannel_path.exists():
        backchannel_spans = backchannel_turns(backchannel_path)
        print(f"backchannel spans: {len(backchannel_spans)}, taken from '{args.backchannel_source}'")
    else:
        backchannel_spans = []
        print(f"backchannel spans: none -- {backchannel_path.name} not found, so that column is n/a")

    reference_shares = talk_time_shares(reference)
    print(f"reference talk-time: " + "  ".join(f"{k}:{v * 100:.1f}%" for k, v in sorted(reference_shares.items())))
    print("")

    rule = "=" * 96
    print(rule)
    print(f"{'arm':<24}{'collar':>7}{'DER':>9}{'miss':>9}{'FA':>9}{'conf':>9}{'ratio err':>11}{'backchannel':>13}")
    print(rule)

    results = {}
    for rttm_path in sorted(stage1_dir.glob(f"{stem}.*.rttm")):
        if rttm_path.name.endswith(".exclusive.rttm"):
            continue
        arm = rttm_path.name[len(stem) + 1: -len(".rttm")]
        hypotheses = load_rttm(str(rttm_path))
        if stem not in hypotheses:
            print(f"{arm:<24}  (no annotation for '{stem}' -- skipped)")
            continue
        hypothesis = hypotheses[stem]

        # --- talk-time ratio error, on the largest reference speaker ---
        mapping = map_labels_to_reference(reference, hypothesis)
        hypothesis_shares = {}
        for label, share in talk_time_shares(hypothesis).items():
            target = mapping.get(label)
            if target is not None:
                hypothesis_shares[target] = hypothesis_shares.get(target, 0.0) + share
        ratio_error = max(
            (abs(hypothesis_shares.get(label, 0.0) - share) for label, share in reference_shares.items()),
            default=float("nan"),
        )

        # --- backchannel attribution, on the shared span set ---
        backchannel_report = "n/a"
        if backchannel_spans:
            correct = sum(
                1 for start, end in backchannel_spans
                if mapping.get(dominant_speaker(hypothesis, start, end)) == dominant_speaker(reference, start, end)
            )
            backchannel_report = f"{correct}/{len(backchannel_spans)} {correct / len(backchannel_spans) * 100:.0f}%"

        results[arm] = {"talk_time_ratio_error": ratio_error, "backchannel": backchannel_report, "der": {}}
        for collar in COLLARS:
            metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
            detail = metric(reference, hypothesis, uem=uem, detailed=True)
            results[arm]["der"][str(collar)] = {
                "der": detail["diarization error rate"],
                "missed_detection": detail["missed detection"],
                "false_alarm": detail["false alarm"],
                "confusion": detail["confusion"],
            }
            # The per-arm columns that do not depend on the collar print once, on the first row.
            extra = f"{ratio_error * 100:>10.1f}%{backchannel_report:>13}" if collar == COLLARS[0] else ""
            print(f"{arm if collar == COLLARS[0] else '':<24}{collar:>7}"
                  f"{detail['diarization error rate'] * 100:>8.2f}%"
                  f"{detail['missed detection']:>8.1f}s"
                  f"{detail['false alarm']:>8.1f}s"
                  f"{detail['confusion']:>8.1f}s{extra}")
    print(rule)
    print("DER is overlap-INCLUSIVE at both collars. 'ratio err' is the largest absolute error in any\n"
          "reference speaker's share of speech. 'backchannel' is whole listener-token turns landing on\n"
          "the right person. Agreement between arms is not accuracy -- only this reference is.")

    output_path = stage1_dir / f"{stem}.arm_scores.json"
    with open(output_path, "w") as f:
        json.dump({"stem": stem, "reference": str(reference_path), "uem": args.uem, "arms": results}, f, indent=2)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
