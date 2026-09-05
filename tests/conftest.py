"""Shared synthetic fixtures. Nothing here touches session data."""

import random

import pytest

VOCAB = "so how has the week been since we last talked I don't know it was fine".split()


@pytest.fixture
def rng():
    """A seeded RNG, so a failure is reproducible rather than a coin flip."""
    return random.Random(1234)


def make_words(rng, count, clock=0.0, untimed_rate=0.0, zero_duration_rate=0.0):
    """Build a word list shaped like whisperx.align's output.

    untimed_rate and zero_duration_rate reproduce the two ALIGNMENT artifacts that make a
    word unlabelable: no "start" key at all, and a word whose start and end are equal.
    """
    words = []
    for _ in range(count):
        duration = rng.uniform(0.06, 0.45)
        word = {"word": rng.choice(VOCAB), "start": round(clock, 3),
                "end": round(clock + duration, 3), "score": round(rng.uniform(0.3, 1.0), 3)}
        if rng.random() < untimed_rate:
            word.pop("start"); word.pop("end")
        elif rng.random() < zero_duration_rate:
            word["end"] = word["start"]
        clock += duration
        words.append(word)
    return words, clock


@pytest.fixture
def aligned_transcript(rng):
    """A Stage 1a-shaped transcript, INCLUDING the aliasing whisperx.align produces.

    "word_segments" holds THE SAME dicts the segments' "words" lists hold -- one object,
    two references. That aliasing is what relink_word_segments exists to restore after a
    JSON round trip, so a fixture that did not have it would test nothing.
    """
    segments, flat, clock = [], [], 0.0
    for _ in range(60):
        words, clock = make_words(rng, rng.randint(1, 8), clock,
                                  untimed_rate=0.03, zero_duration_rate=0.03)
        segments.append({
            "start": round(words[0].get("start", clock), 3),
            "end": round(clock, 3),
            "text": " ".join(w["word"] for w in words),
            "avg_logprob": round(rng.uniform(-1.4, -0.1), 4),
            "words": words,
        })
        flat.extend(words)
        clock += rng.uniform(0, 2.0)
    return {"segments": segments, "word_segments": flat, "language": "en"}


@pytest.fixture
def turn_table(rng, aligned_transcript):
    """Speaker turns over the same span, with real gaps and real overlap."""
    end = aligned_transcript["segments"][-1]["end"]
    turns, clock = [], 0.0
    while clock < end:
        duration = rng.uniform(0.8, 9.0)
        turns.append((round(clock, 3), round(min(clock + duration, end), 3),
                      rng.choice(["SPEAKER_00", "SPEAKER_01"])))
        clock = max(clock + duration + rng.uniform(-1.0, 2.0), 0.0)
    return turns
