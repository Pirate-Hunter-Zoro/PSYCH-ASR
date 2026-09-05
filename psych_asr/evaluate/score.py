"""Score every diarization arm against a hand-corrected reference RTTM.

Needs pyannote.metrics, so this imports only in diar_eval_env -- which holds
pyannote.metrics and DELIBERATELY NO TORCH. Folding it into asr_env would risk bumping
pyannote.core underneath the locked whisperx pin set, and it would quietly couple "how we
measure" to "what we measure with."

THIS CANNOT PRODUCE A REAL NUMBER UNTIL THE REFERENCE EXISTS. The reference is the
hand-corrected RTTM from Stage 2's stratified subset -- the same subset already planned for
WER/DER estimation, used twice rather than built twice. Scoring one arm against another
measures divergence, not accuracy: four arms can agree and all be wrong.

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
"""

from pyannote.metrics.diarization import DiarizationErrorRate

from ..artifacts.transcripts import load_transcript
from ..transcript.turns import group_into_turns
from .labels import dominant_speaker, map_labels_to_reference

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

    Each speaker's own summed turn duration over the sum across speakers, so silence does
    not dilute the split -- the same denominator the transcript renderer uses.
    """
    per_speaker = {label: annotation.label_duration(label) for label in annotation.labels()}
    total = sum(per_speaker.values())
    return {label: duration / total for label, duration in per_speaker.items()} if total else {}


def backchannel_spans(transcript_path):
    """IN: path to a <stem>.<arm>.diarized.json  OUT: list of (start, end) spans.

    A backchannel is a WHOLE TURN -- a run of consecutive same-speaker segments -- whose
    entire text is one listener token. Taking it at segment level instead would catch every
    "right" that opens a sentence, which is not the same thing at all.

    Turns are grouped with unknown=None so two genuinely unlabeled stretches are not welded
    into one turn by a shared placeholder they never carried.
    """
    transcript = load_transcript(transcript_path)
    turns = group_into_turns(transcript.get("segments", []), unknown=None)
    return [(turn["start"], turn["end"]) for turn in turns if normalize(turn["text"]) in BACKCHANNEL_TOKENS]


def talk_time_ratio_error(reference, hypothesis, reference_shares, mapping):
    """IN: two Annotations, the reference's shares, and the hypothesis->reference mapping
    OUT: the LARGEST absolute error in any reference speaker's share of speech.

    The maximum rather than the mean: with two speakers the two errors are the same number,
    and with three the one that matters is the worst one, not the average that hides it.
    """
    hypothesis_shares = {}
    for label, share in talk_time_shares(hypothesis).items():
        target = mapping.get(label)
        if target is not None:
            hypothesis_shares[target] = hypothesis_shares.get(target, 0.0) + share
    return max(
        (abs(hypothesis_shares.get(label, 0.0) - share) for label, share in reference_shares.items()),
        default=float("nan"),
    )


def backchannel_accuracy(reference, hypothesis, spans, mapping):
    """IN: two Annotations, the shared backchannel spans, the label mapping
    OUT: (correct, total) -- how many spans landed on the right person.

    ONE ARM'S TRANSCRIPT DEFINES THE SPANS FOR EVERY ARM. The spans need the WORDS, which
    an RTTM does not carry, and letting each arm nominate its own would change the
    denominator per arm and make the percentages incomparable.
    """
    correct = sum(
        1 for start, end in spans
        if mapping.get(dominant_speaker(hypothesis, start, end)) == dominant_speaker(reference, start, end)
    )
    return correct, len(spans)


def score_arm(reference, hypothesis, reference_shares, spans, uem=None, collars=COLLARS):
    """Score one arm against the reference on all four measures.

    IN:  reference and hypothesis Annotations, the reference's talk-time shares, the shared
         backchannel spans, an optional UEM, and the collars to report
    OUT: dict with "talk_time_ratio_error", "backchannel" (correct, total) and "der",
         the last keyed by collar as a string

    A UEM MATTERS AND IS NOT OPTIONAL FOR A REPORTED NUMBER. Without one pyannote.metrics
    approximates the evaluation region as the union of reference and hypothesis extents,
    which lets an arm be scored over stretches the reference never annotated.
    """
    mapping = map_labels_to_reference(reference, hypothesis)
    result = {
        "talk_time_ratio_error": talk_time_ratio_error(reference, hypothesis, reference_shares, mapping),
        "backchannel": backchannel_accuracy(reference, hypothesis, spans, mapping) if spans else None,
        "der": {},
    }
    for collar in collars:
        metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
        detail = metric(reference, hypothesis, uem=uem, detailed=True)
        result["der"][str(collar)] = {
            "der": detail["diarization error rate"],
            "missed_detection": detail["missed detection"],
            "false_alarm": detail["false alarm"],
            "confusion": detail["confusion"],
        }
    return result
