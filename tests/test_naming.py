"""The Stage 1 filename convention. Getting this wrong mislabels which model produced what."""

from psych_asr.artifacts import naming


def test_stem_and_arm_survive_a_hyphenated_arm_name():
    assert naming.stem_from_aligned("d/S1.aligned.json") == "S1"
    assert naming.arm_from("d/S1.sortformer-streaming.rttm", "S1", naming.RTTM_SUFFIX) == "sortformer-streaming"
    assert naming.arm_from("d/S1.community-1.diarized.json", "S1", naming.DIARIZED_SUFFIX) == "community-1"


def test_an_arm_name_containing_a_dot_is_not_truncated():
    """Splitting on "." would return "v2" here. Suffix stripping returns the whole arm."""
    assert naming.arm_from("d/S1.sortformer-v2.1.rttm", "S1", naming.RTTM_SUFFIX) == "sortformer-v2.1"


def test_a_stem_containing_dots_still_resolves():
    """Real stems carry a participant code and a session number. The placeholder here is
    deliberately not code-shaped -- the pre-commit hook refuses a staged BL### and it is
    right to, since a fixture that looks like a real one is how a real one gets committed."""
    assert naming.stem_from_aligned("d/PILOT.session1.aligned.json") == "PILOT.session1"
    assert naming.arm_from("d/PILOT.session1.diarizen.rttm", "PILOT.session1",
                           naming.RTTM_SUFFIX) == "diarizen"


def test_an_unexpected_filename_labels_itself_visibly_rather_than_raising():
    assert naming.arm_from("d/handmade.rttm", "S1", naming.RTTM_SUFFIX) == "handmade"


def test_the_single_job_path_and_the_split_do_not_collide():
    """The gate identifies the fixture by the ABSENCE of an arm, so the two names are not
    interchangeable."""
    assert naming.diarized_path("d", "S1").name == "S1.diarized.json"
    assert naming.diarized_path("d", "S1", "community-1").name == "S1.community-1.diarized.json"
    assert naming.transcript_path("d", "S1").name == "S1.transcript.txt"
    assert naming.transcript_path("d", "S1", "diarizen").name == "S1.diarizen.transcript.txt"


def test_discovery_skips_the_overlap_free_view(tmp_path):
    """.exclusive.rttm is a diagnostic on the baseline arm, not an arm of its own -- joining
    it would produce a fifth transcript that no model actually produced."""
    for name in ["S1.community-1.rttm", "S1.community-1.exclusive.rttm",
                 "S1.diarizen.rttm", "S1.sortformer-streaming.rttm"]:
        (tmp_path / name).write_text("")
    assert [arm for arm, _ in naming.find_arm_rttms(tmp_path, "S1")] == [
        "community-1", "diarizen", "sortformer-streaming"]


def test_a_crashed_arm_is_absent_rather_than_fatal(tmp_path):
    (tmp_path / "S1.community-1.diarized.json").write_text("{}")
    assert [arm for arm, _ in naming.find_arm_transcripts(tmp_path, "S1")] == ["community-1"]


def test_more_than_one_session_in_the_directory_is_refused(tmp_path):
    import pytest
    (tmp_path / "A.aligned.json").write_text("{}")
    (tmp_path / "B.aligned.json").write_text("{}")
    with pytest.raises(SystemExit):
        naming.find_sole_stem(tmp_path)
