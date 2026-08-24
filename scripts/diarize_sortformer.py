"""Stage 1b, arms B and C: NVIDIA Sortformer -> RTTM.

Audio in, RTTM out, same contract as every other 1b script. Both checkpoints load through
the same NeMo class, so one script and one env cover both arms:

  arm B  --mode offline    nvidia/diar_sortformer_4spk-v1            CC BY-NC 4.0
  arm C  --mode streaming  nvidia/diar_streaming_sortformer_4spk-v2.1  NVIDIA Open Model License

Arm C's license is the reason to care which one wins: arms A and B are non-commercial-only,
and the stated deliverable is an R21/R01, so an NC weight license is the kind of thing that
is invisible for two years and then blocks a translation path.

TWO THINGS ABOUT THESE MODELS DIFFER FROM THE PYANNOTE ARMS, AND BOTH ARE FINDINGS RATHER
THAN IMPLEMENTATION DETAILS.

1. THEY CANNOT BE PINNED TO TWO SPEAKERS. Sortformer is end-to-end with a hard four-speaker
   ceiling and no clustering stage to constrain, so there is no num_speakers to pass. The
   pyannote path takes an exact count and the pipeline pins it to 2. On a known dyad that
   constraint is worth real DER, and losing it is a genuine cost of switching, not a
   footnote. --num-speakers is accepted here and deliberately ignored, so the bake-off
   driver can pass the same flags to every arm; the log says plainly that it was ignored.

2. THE OFFLINE ARM DOES NOT SURVIVE A 50-MINUTE FILE UNAIDED. NVIDIA's own model card puts
   the limit near 12 minutes on a 48 GB RTX A6000; the node's A40 has 46 GB and a pilot
   session is ~50 minutes. Arm B is therefore run over overlapping windows and stitched
   back together here (see stitch_windows below). Arm C handles arbitrary length by
   construction -- that is the streaming variant's whole point, and it may turn out to be
   its decisive advantage independent of accuracy.

   The stitcher is EXTERNAL MACHINERY THAT ONLY ARM B NEEDS. When the arms are scored,
   that asymmetry belongs in the writeup: some of arm B's error will be seam error, not
   model error.

Runs in nemo_env. NeMo carries its own torch, hydra, lightning and transformers pins,
which is exactly the set that would break asr_env's locked whisperx pin chain.
"""

from argparse import ArgumentParser
from pathlib import Path
import shutil

import numpy as np
import soundfile as sf
from nemo.collections.asr.models import SortformerEncLabelModel

# Sibling module in scripts/; `python scripts/diarize_sortformer.py` puts that dir on sys.path[0].
from rttm_io import write_rttm, read_rttm, summarize_turns, format_turn_summary

MODELS_ROOT = "/media/studies/ehr_study/analysis/mferguson/models"

# The model emits one activity decision per 0.08 s of audio; matching windows on that same
# grid means the stitcher never invents a resolution the model does not have.
FRAME_SECONDS = 0.08

# Hard four-speaker ceiling, and the number of global label slots the stitcher carries.
MAX_SPEAKERS = 4

# NVIDIA's "very high latency" streaming preset, all counts in 80 ms frames. It is the
# accuracy-favouring end of their published table (RTF 0.002); the low-latency preset
# exists to hit a 1 s input-buffer latency, which is irrelevant for batch processing of
# recorded sessions.
STREAMING_PRESET = {
    "chunk_len": 340,
    "chunk_right_context": 40,
    "fifo_len": 40,
    "spkcache_update_period": 300,
    "spkcache_len": 188,
}

ARMS = {
    "offline": {
        "arm": "sortformer",
        "checkpoint": f"{MODELS_ROOT}/diar_sortformer_4spk-v1/diar_sortformer_4spk-v1.nemo",
    },
    "streaming": {
        "arm": "sortformer-streaming",
        "checkpoint": f"{MODELS_ROOT}/diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1.nemo",
    },
}


def parse_segments(raw_segments, time_offset=0.0):
    """Normalize NeMo's diarize() output into (start, end, local_speaker) tuples.

    IN:  raw_segments -- one audio file's entry from diarize(), whose elements are either
         "start end speaker" strings or (start, end, speaker) sequences depending on NeMo
         version; time_offset -- seconds to add, for a window cut out of a longer file
    OUT: list of (float, float, str)

    Both shapes are accepted rather than pinned to one, because the two model cards
    document the return value only as "begin_seconds, end_seconds, speaker_index".
    """
    turns = []
    for item in raw_segments:
        if isinstance(item, str):
            fields = item.split()
        else:
            fields = list(item)
        start, end, speaker = float(fields[0]), float(fields[1]), str(fields[2])
        turns.append((start + time_offset, end + time_offset, speaker))
    return turns


def activity_grid(turns, labels, grid_start, grid_end):
    """Rasterize turns onto the model's own 80 ms frame grid.

    IN:  turns  -- (start, end, label) tuples in absolute seconds
         labels -- the label ordering that becomes the rows
         grid_start / grid_end -- the window to rasterize, in seconds
    OUT: bool array of shape (len(labels), num_frames)

    Used only to compare two windows on the stretch they share.
    """
    num_frames = max(int(round((grid_end - grid_start) / FRAME_SECONDS)), 0)
    grid = np.zeros((len(labels), num_frames), dtype=bool)
    index_of = {label: i for i, label in enumerate(labels)}
    for start, end, label in turns:
        if label not in index_of:
            continue
        first = int(np.floor((max(start, grid_start) - grid_start) / FRAME_SECONDS))
        last = int(np.ceil((min(end, grid_end) - grid_start) / FRAME_SECONDS))
        if last > first:
            grid[index_of[label], max(first, 0):min(last, num_frames)] = True
    return grid


def match_labels(previous_turns, window_turns, overlap_start, overlap_end):
    """Decide which of the new window's local speakers is which of the running speakers.

    IN:  previous_turns -- turns accepted so far, carrying GLOBAL labels
         window_turns   -- the new window's turns, carrying LOCAL labels
         overlap_start / overlap_end -- the stretch both windows saw, in seconds
    OUT: dict mapping local label -> global label

    A window's speaker indices are arbitrary and independent of every other window's, so
    the stitch has to be made on evidence rather than on index. The evidence is the overlap
    region: whichever local speaker's activity best coincides with a global speaker's
    activity there is the same person. The pairing is solved as a linear assignment over
    frame agreement, so it is one-to-one -- two local speakers can never collapse onto the
    same global label, which is the failure that would silently merge two people.

    A local speaker with no overlap evidence at all gets a free global slot instead of a
    guess; that is what keeps a speaker who only appears late in the session from being
    grafted onto an existing label.
    """
    from scipy.optimize import linear_sum_assignment

    global_labels = [f"speaker_{i}" for i in range(MAX_SPEAKERS)]
    local_labels = sorted({label for _, _, label in window_turns})
    if not local_labels:
        return {}

    previous_grid = activity_grid(previous_turns, global_labels, overlap_start, overlap_end)
    window_grid = activity_grid(window_turns, local_labels, overlap_start, overlap_end)

    # Agreement counted as frames where both are active; negated because the solver minimizes.
    agreement = window_grid.astype(np.int32) @ previous_grid.astype(np.int32).T
    rows, columns = linear_sum_assignment(-agreement)

    mapping = {}
    taken = set()
    for row, column in zip(rows, columns):
        if agreement[row, column] > 0:
            mapping[local_labels[row]] = global_labels[column]
            taken.add(global_labels[column])
    for label in local_labels:
        if label not in mapping:
            free = [g for g in global_labels if g not in taken]
            if not free:
                break
            mapping[label] = free[0]
            taken.add(free[0])
    return mapping


def stitch_windows(window_results, stride, overlap):
    """Splice per-window diarizations into one timeline with consistent speaker labels.

    IN:  window_results -- list of (window_start_seconds, turns) in window order, each
         turns list carrying that window's own local labels in ABSOLUTE seconds
         stride / overlap -- the windowing used, in seconds
    OUT: list of (start, end, global_label) tuples covering the whole recording

    Each window contributes the timeline up to the midpoint of its overlap with the next
    one. Cutting at the midpoint rather than at a window edge keeps every accepted second
    away from the region where a model has the least context -- the first and last moments
    of its input, where an end-to-end diarizer is at its worst.
    """
    accepted = []
    # The PREVIOUS window's full mapped turns, not `accepted`. Accepted turns have already
    # been truncated at the overlap midpoint, so matching against them would throw away
    # half the evidence the overlap exists to provide.
    previous_mapped = []
    for index, (window_start, turns) in enumerate(window_results):
        if index == 0:
            # Normalize the first window's local labels onto the global slot names too, so
            # every label in the output comes from one namespace regardless of what the
            # model happened to call its speakers.
            first_labels = sorted({label for _, _, label in turns})
            first_mapping = {label: f"speaker_{i}" for i, label in enumerate(first_labels)}
            mapped = [(s, e, first_mapping[label]) for s, e, label in turns]
        else:
            overlap_start = window_start
            overlap_end = window_start + overlap
            mapping = match_labels(previous_mapped, turns, overlap_start, overlap_end)
            mapped = [(s, e, mapping.get(label, label)) for s, e, label in turns]
        previous_mapped = mapped

        if index + 1 < len(window_results):
            next_start = window_results[index + 1][0]
            cut = next_start + overlap / 2.0
        else:
            cut = float("inf")
        keep_from = (window_start + overlap / 2.0) if index > 0 else 0.0

        accepted = [(s, min(e, keep_from), label) for s, e, label in accepted if s < keep_from]
        accepted.extend(
            (max(s, keep_from), min(e, cut), label)
            for s, e, label in mapped
            if e > keep_from and s < cut
        )
    return merge_adjacent(t for t in accepted if t[1] > t[0])


def merge_adjacent(turns, tolerance=0.01):
    """Rejoin same-speaker turns that the seam cut in half.

    IN:  turns -- (start, end, label) tuples; tolerance -- the largest gap still treated
         as "touching", in seconds
    OUT: the same turns with touching or overlapping same-speaker pairs merged

    A turn spanning a window boundary is truncated by one window and re-emitted by the
    next, so without this it lands in the RTTM as two consecutive turns by the same
    speaker. That is a pure artifact of the stitcher, and it would inflate arm B's turn
    count and its between-turn latency against arms that never got windowed. The tolerance
    is deliberately far below any real pause in speech, so nothing but a seam is joined.
    """
    by_label = {}
    for start, end, label in turns:
        by_label.setdefault(label, []).append((start, end))

    merged = []
    # Grouped by speaker first, so a seam split is still rejoined when the OTHER speaker
    # happens to be talking across it. Sorting the mixed list instead would put that
    # speaker's turn between the two halves and block the merge.
    for label, spans in by_label.items():
        current = None
        for start, end in sorted(spans):
            if current is not None and start - current[1] <= tolerance:
                current[1] = max(current[1], end)
            else:
                if current is not None:
                    merged.append((current[0], current[1], label))
                current = [start, end]
        if current is not None:
            merged.append((current[0], current[1], label))
    return sorted(merged)


def diarize_windowed(model, audio_path, scratch_dir, window, overlap, batch_size):
    """Run the offline model over overlapping windows and stitch the results.

    IN:  loaded model, path to the 16 kHz mono WAV, a scratch directory for window WAVs,
         window and overlap lengths in seconds, batch size
    OUT: list of (start, end, speaker) tuples over the whole recording

    Window WAVs are written under the output directory (which is gitignored) rather than
    anywhere else, because they are cut from session audio and are therefore PHI. They are
    deleted before the function returns.
    """
    info = sf.info(str(audio_path))
    duration = info.frames / info.samplerate
    stride = window - overlap

    if duration <= window:
        segments = model.diarize(audio=[str(audio_path)], batch_size=batch_size)
        return parse_segments(segments[0])

    scratch_dir.mkdir(parents=True, exist_ok=True)
    audio, sample_rate = sf.read(str(audio_path), dtype="float32")
    starts = list(np.arange(0.0, max(duration - overlap, 0.0), stride))
    print(f"Windowing {duration:.1f}s into {len(starts)} windows of {window:.0f}s "
          f"(stride {stride:.0f}s, overlap {overlap:.0f}s)", flush=True)

    window_results = []
    try:
        for index, start in enumerate(starts):
            first_sample = int(round(start * sample_rate))
            last_sample = min(int(round((start + window) * sample_rate)), len(audio))
            window_path = scratch_dir / f"window_{index:03d}.wav"
            sf.write(str(window_path), audio[first_sample:last_sample], sample_rate)

            segments = model.diarize(audio=[str(window_path)], batch_size=batch_size)
            turns = parse_segments(segments[0], time_offset=start)
            print(f"  window {index + 1}/{len(starts)} [{start:.0f}s]: {len(turns)} turns, "
                  f"{len({t[2] for t in turns})} local speakers", flush=True)
            window_results.append((start, turns))
            window_path.unlink()
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    return stitch_windows(window_results, stride, overlap)


def main():
    parser = ArgumentParser(description="Stage 1b arms B/C: diarize one WAV with NVIDIA Sortformer.")
    parser.add_argument("audio", type=str)
    parser.add_argument("--mode", choices=sorted(ARMS), default="offline",
                        help="offline = arm B (diar_sortformer_4spk-v1); "
                             "streaming = arm C (diar_streaming_sortformer_4spk-v2.1)")
    parser.add_argument("--outdir", type=str, default="data/stage1")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="absolute path to the .nemo archive (default: the staged one for --mode)")
    parser.add_argument("--arm", type=str, default=None,
                        help="name used in the output filenames (default: the arm for --mode)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--window-seconds", type=float, default=600.0,
                        help="offline mode only: window length. NVIDIA reports ~12 min as the "
                             "ceiling on a 48 GB card, so 10 min leaves headroom on the 46 GB A40")
    parser.add_argument("--overlap-seconds", type=float, default=60.0,
                        help="offline mode only: how much consecutive windows share. This is the "
                             "only evidence the stitcher has for deciding that two windows' "
                             "speaker indices refer to the same person")
    parser.add_argument("--num-speakers", type=int, default=2,
                        help="accepted for flag compatibility with the pyannote arms and IGNORED: "
                             "Sortformer is end-to-end with a 4-speaker ceiling and no clustering "
                             "stage to constrain")
    args = parser.parse_args()

    arm_defaults = ARMS[args.mode]
    arm = args.arm or arm_defaults["arm"]
    checkpoint = args.checkpoint or arm_defaults["checkpoint"]

    audio_path = Path(args.audio)
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    uri = audio_path.stem

    print(f"Arm {arm} ({args.mode}) from {checkpoint}", flush=True)
    print(f"NOTE: --num-speakers={args.num_speakers} is IGNORED; Sortformer cannot be pinned "
          f"to a speaker count. Whatever it emits, up to {MAX_SPEAKERS}, is the result.", flush=True)

    # restore_from loads the .nemo archive straight off disk -- no Hub call, so this needs
    # no token and works with HF_HUB_OFFLINE=1 set. strict=False per NVIDIA's model cards.
    model = SortformerEncLabelModel.restore_from(restore_path=checkpoint, map_location="cuda", strict=False)
    model.eval()

    if args.mode == "streaming":
        for name, value in STREAMING_PRESET.items():
            setattr(model.sortformer_modules, name, value)
        print(f"Streaming preset: {STREAMING_PRESET}", flush=True)
        # Arbitrary length by construction -- no windowing, no stitcher, no seam error.
        segments = model.diarize(audio=[str(audio_path)], batch_size=args.batch_size)
        turns = parse_segments(segments[0])
    else:
        turns = diarize_windowed(
            model,
            audio_path,
            output_dir / f".{uri}.{arm}.windows",
            args.window_seconds,
            args.overlap_seconds,
            args.batch_size,
        )

    rttm_path = output_dir / f"{uri}.{arm}.rttm"
    num_turns = write_rttm(turns, uri, rttm_path)

    print(f"Wrote {rttm_path} ({num_turns} turns)", flush=True)
    print("\n".join(format_turn_summary(arm, uri, summarize_turns(read_rttm(rttm_path)))), flush=True)


if __name__ == "__main__":
    main()
