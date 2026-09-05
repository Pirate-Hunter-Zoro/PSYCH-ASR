"""Stage 1b, arm A: DiariZen (BUT-FIT/diarizen-wavlm-large-s80-md-v2).

Runs in diarizen_env: python 3.10, torch 2.1.1+cu121, DiariZen installed from source
together with its vendored pyannote-audio fork. That torch pin is why this cannot live in
asr_env.

WHY THIS ARM EXISTS: DiariZen's segmentation model is powerset over up to four concurrent
speakers, so it models overlapping speech directly, where community-1's config sets
embedding_exclude_overlap and protects its speaker profiles by DISCARDING the overlapped
regions instead of resolving them. Therapist backchannels over patient speech are constant
in this corpus and are themselves a Stage 3a feature.

WEIGHTS ARE CC BY-NC 4.0 (the code is MIT). Fine for feasibility work at a nonprofit
institute; not shippable in a translation path. Record which arm wins and, separately,
which arm can ship.

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

from pathlib import Path

from diarizen.pipelines.inference import DiariZenPipeline

# The phantom cluster described in warn_sentinel_cluster is always this label, because the
# -2 sentinel is the minimum value np.unique sorts to position zero.
SENTINEL_LABEL = "0"


def load_pipeline(model_dir, embedding_model):
    """IN: the DiariZen hub directory + the WeSpeaker .bin file   OUT: a DiariZenPipeline."""
    return DiariZenPipeline(
        diarizen_hub=Path(model_dir).expanduser().absolute(),
        embedding_model=str(embedding_model),
    )


def pin_speaker_count(pipeline, num_speakers):
    """Force the clustering to at most num_speakers, and say how in one line.

    IN:  the pipeline + an exact speaker count   OUT: a log line describing what was done

    PINNING THE SPEAKER COUNT HERE IS NOT WHERE IT LOOKS. DiariZen's __call__ passes
    min_clusters/max_clusters into the clustering, which reads like the knob and is not:
    VBxClustering.__call__ declares all three count arguments "not used but kept for
    compatibility" and ignores them outright. Clamping min_speakers/max_speakers alone
    returned FOUR clusters on a two-person session.

    The count actually falls out of VBx, which is seeded by an agglomerative pass:
        fcluster(dendrogram, self.ahc_threshold, criterion=self.ahc_criterion)
    With the shipped criterion "distance" the threshold is a dendrogram cut height (0.6)
    and the count is whatever that produces. With criterion "maxclust", scipy reads the
    threshold as A MAXIMUM NUMBER OF CLUSTERS instead -- so seeding VBx with exactly
    num_speakers is what pins it. VBx then prunes components whose weight decays to zero
    and never creates new ones, so the count can only come out at or below the seed.

    This is a deliberate deviation from the checkpoint's published configuration, and it is
    the fair one: the pyannote baseline is already told there are exactly two people, so a
    challenger denied the same information would be compared on worse footing. Both ways
    are run and reported separately, because "can it be pinned at all" is itself one of the
    bake-off's questions.
    """
    pipeline.clustering.ahc_criterion = "maxclust"
    pipeline.clustering.ahc_threshold = num_speakers
    # Still clamped: this one governs the per-frame count of CONCURRENT speakers, which is
    # a different quantity from how many people are in the room.
    pipeline.min_speakers = num_speakers
    pipeline.max_speakers = num_speakers
    return (f"Pinned to {num_speakers} speakers by seeding AHC with criterion "
            f"'maxclust' at threshold {num_speakers}.")


def diarize(pipeline, audio_path, uri):
    """IN: the pipeline, the 16 kHz WAV path, the recording id   OUT: pyannote Annotation.

    __call__ reads the file with torchaudio and forces channel 0, so the standardized mono
    WAV goes in unchanged. Note this arm takes a PATH, where the pyannote arm takes the
    decoded array -- DiariZen does its own reading and offers no array entry point.
    """
    return pipeline(str(audio_path), sess_name=uri)


def sentinel_warning(turns, pinned_count):
    """Flag the phantom speaker DiariZen's clustering can invent, without deleting it.

    IN:  turns from read_rttm; the pinned speaker count, or None if unpinned
    OUT: a warning string, or None when there is nothing to warn about

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

    NOTHING IS DELETED. The RTTM stays a faithful record of what the model emitted, the
    phantom shows up in the rendered transcript as a speaker with ~0% talk time, and the
    reader gets told what they are looking at. Silently dropping a cluster would be a guess
    wearing the same shape as a measurement.
    """
    by_label = {}
    for turn in turns:
        by_label[turn["speaker"]] = by_label.get(turn["speaker"], 0.0) + turn["end"] - turn["start"]
    if pinned_count is None or len(by_label) <= pinned_count:
        return None

    total = sum(by_label.values())
    sentinel = by_label.get(SENTINEL_LABEL, 0.0)
    share = (sentinel / total * 100.0) if total else 0.0
    return (
        f"WARNING: pinned to {pinned_count} speakers but {len(by_label)} labels came out.\n"
        f"  Label \"{SENTINEL_LABEL}\" holds {sentinel:.1f}s ({share:.2f}% of speech) and is very "
        f"probably NOT A PERSON.\n"
        f"  Cause: the vendored fork's VBxClustering renumbers constrained_argmax's -2 "
        f"\"unassigned\" sentinel\n"
        f"  into a real cluster index. See sentinel_warning in this file. Not deleted -- "
        f"the RTTM stays faithful."
    )
