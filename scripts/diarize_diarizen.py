"""Stage 1b, arm A: DiariZen (BUT-FIT/diarizen-wavlm-large-s80-md-v2) -> RTTM.

Audio in, RTTM out, same contract as every other 1b script.

Why this arm exists: DiariZen's segmentation model is powerset over up to four concurrent
speakers, so it models overlapping speech directly, where community-1's config sets
embedding_exclude_overlap and protects its speaker profiles by discarding the overlapped
regions instead of resolving them. Therapist backchannels over patient speech are constant
in this corpus and are themselves a Stage 3a feature.

WEIGHTS ARE CC BY-NC 4.0 (the code is MIT). Fine for feasibility work at a nonprofit
institute; not shippable in a translation path. Record which arm wins and, separately,
which arm can ship.

Runs in diarizen_env: python 3.10, torch 2.1.1+cu121, DiariZen installed from source
together with its vendored pyannote-audio fork. That torch pin is why this cannot live in
asr_env.

OFFLINE LOADING: DiariZenPipeline.from_pretrained() calls snapshot_download and
hf_hub_download, so it wants either the network or an HF-cache-shaped directory. We
sidestep both and construct the pipeline directly from absolute local paths, which is what
from_pretrained does anyway once its two downloads resolve. Two paths are needed:

  * the model hub directory -- config.toml, pytorch_model.bin, and plda/ .  Its config.toml
    names wavlm_src "wavlm_large_s80_md", which resolves to a HARDCODED dict in
    diarizen/models/module/wavlm_config.py. That is NOT a third download.
  * the speaker embedder, pyannote/wespeaker-voxceleb-resnet34-LM's pytorch_model.bin,
    already staged for the community-1 baseline and passed as a plain file path.
"""

from argparse import ArgumentParser
from pathlib import Path

from diarizen.pipelines.inference import DiariZenPipeline

# Sibling module in scripts/; `python scripts/diarize_diarizen.py` puts that dir on sys.path[0].
from rttm_io import write_rttm_from_annotation, read_rttm, summarize_turns, format_turn_summary

ARM = "diarizen"
MODELS_ROOT = "/media/studies/ehr_study/analysis/mferguson/models"


def main():
    parser = ArgumentParser(description="Stage 1b arm A: diarize one WAV with DiariZen.")
    parser.add_argument("audio", type=str)
    parser.add_argument("--outdir", type=str, default="data/stage1")
    parser.add_argument("--model-dir", type=str, default=f"{MODELS_ROOT}/diarizen-wavlm-large-s80-md-v2")
    parser.add_argument("--embedding-model", type=str,
                        default=f"{MODELS_ROOT}/pyannote-wespeaker-voxceleb-resnet34-LM/pytorch_model.bin")
    parser.add_argument("--num-speakers", type=int, default=2,
                        help="exact speaker count, pinned through the AHC seeding (see below). "
                             "Pass 0 to leave the checkpoint's shipped configuration alone and let "
                             "the clustering decide for itself")
    parser.add_argument("--arm", type=str, default=None,
                        help="name used in the output filenames (default: 'diarizen' when pinned, "
                             "'diarizen-free' when not, so the two runs never overwrite each other)")
    args = parser.parse_args()

    pinned = args.num_speakers > 0
    arm = args.arm or (ARM if pinned else f"{ARM}-free")

    audio_path = Path(args.audio)
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    uri = audio_path.stem

    pipeline = DiariZenPipeline(
        diarizen_hub=Path(args.model_dir).expanduser().absolute(),
        embedding_model=args.embedding_model,
    )

    # PINNING THE SPEAKER COUNT HERE IS NOT WHERE IT LOOKS. DiariZen's __call__ passes
    # min_clusters/max_clusters into the clustering, which reads like the knob and is not:
    # VBxClustering.__call__ declares all three count arguments "not used but kept for
    # compatibility" and ignores them outright. Clamping min_speakers/max_speakers alone
    # returned FOUR clusters on a two-person session.
    #
    # The count actually falls out of VBx, which is seeded by an agglomerative pass:
    #   fcluster(dendrogram, self.ahc_threshold, criterion=self.ahc_criterion)
    # With the shipped criterion "distance" the threshold is a dendrogram cut height (0.6)
    # and the count is whatever that produces. With criterion "maxclust", scipy reads the
    # threshold as A MAXIMUM NUMBER OF CLUSTERS instead -- so seeding VBx with exactly
    # num_speakers is what pins it. VBx then prunes components whose weight decays to zero
    # and never creates new ones, so the count can only come out at or below the seed.
    #
    # This is a deliberate deviation from the checkpoint's published configuration, and it
    # is the fair one: the pyannote baseline is already told there are exactly two people,
    # so a challenger denied the same information would be compared on worse footing.
    # Run both ways -- --num-speakers 0 leaves the shipped config alone -- and say which is
    # which, because "can it be pinned at all" is itself one of the questions.
    if pinned:
        pipeline.clustering.ahc_criterion = "maxclust"
        pipeline.clustering.ahc_threshold = args.num_speakers
        # Still clamped: this one governs the per-frame count of CONCURRENT speakers, which
        # is a different quantity from how many people are in the room.
        pipeline.min_speakers = args.num_speakers
        pipeline.max_speakers = args.num_speakers
        print(f"Pinned to {args.num_speakers} speakers by seeding AHC with criterion "
              f"'maxclust' at threshold {args.num_speakers}.", flush=True)
    else:
        print("NOT pinned: the checkpoint's shipped clustering config decides the speaker "
              "count for itself.", flush=True)

    # IN: path to the 16 kHz WAV      OUT: pyannote.core Annotation of speaker turns
    # __call__ reads the file with torchaudio and forces channel 0, so the standardized
    # mono WAV goes in unchanged.
    annotation = pipeline(str(audio_path), sess_name=uri)

    rttm_path = output_dir / f"{uri}.{arm}.rttm"
    num_turns = write_rttm_from_annotation(annotation, uri, rttm_path)

    print(f"Wrote {rttm_path} ({num_turns} turns)", flush=True)
    turns = read_rttm(rttm_path)
    print("\n".join(format_turn_summary(arm, uri, summarize_turns(turns))), flush=True)
    warn_sentinel_cluster(turns, args.num_speakers if pinned else None)


def warn_sentinel_cluster(turns, pinned_count):
    """Flag the phantom speaker DiariZen's clustering can invent, without deleting it.

    IN:  turns from read_rttm; the pinned speaker count, or None if unpinned
    OUT: nothing; prints a diagnostic

    THE BUG, READ OUT OF THE VENDORED FORK. BaseClustering.constrained_argmax fills
    hard_clusters with -2 and then solves a one-to-one assignment per chunk, so a local
    speaker with no cluster left to take KEEPS the -2 sentinel. VBxClustering.__call__ then
    ends with

        _, hard_clusters = np.unique(hard_clusters, return_inverse=True)

    which renumbers the label set -- and -2 is a value like any other, so it becomes
    cluster 0 and every real cluster shifts up by one. Upstream pyannote's own clustering
    classes do not do this; it is specific to the fork DiariZen vendors.

    The consequence is a speaker who is not a person. It is always label "0", because -2 is
    the minimum and np.unique sorts, and it is always tiny -- a few seconds of overlap where
    the powerset segmentation found more concurrent speakers than there were clusters to
    assign them to. It is why pinning this arm to two speakers still emits three labels.

    Nothing is deleted here. The RTTM stays a faithful record of what the model emitted, the
    phantom shows up in the rendered transcript as a speaker with ~0% talk time, and the
    reader gets told what they are looking at. Silently dropping a cluster would be a guess
    wearing the same shape as a measurement.
    """
    by_label = {}
    for turn in turns:
        by_label[turn["speaker"]] = by_label.get(turn["speaker"], 0.0) + turn["end"] - turn["start"]
    if pinned_count is None or len(by_label) <= pinned_count:
        return

    total = sum(by_label.values())
    sentinel = by_label.get("0", 0.0)
    print(
        f"WARNING: pinned to {pinned_count} speakers but {len(by_label)} labels came out.\n"
        f"  Label \"0\" holds {sentinel:.1f}s ({sentinel / total * 100:.2f}% of speech) and is very "
        f"probably NOT A PERSON.\n"
        f"  Cause: the vendored fork's VBxClustering renumbers constrained_argmax's -2 "
        f"\"unassigned\" sentinel\n"
        f"  into a real cluster index. See warn_sentinel_cluster in this file. Not deleted -- "
        f"the RTTM stays faithful.",
        flush=True,
    )


if __name__ == "__main__":
    main()
