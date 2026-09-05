# tests

Pure-logic tests over **synthetic** transcripts and turn tables. No session audio, no
session artifacts, no PHI — every fixture in here is generated in the test that uses it,
so the suite runs anywhere and can be read by anyone.

What it deliberately does not cover: anything that needs a model. Whisper's decode,
pyannote's clustering, DiariZen's VBx and Sortformer's forward pass are all exercised only
by an actual Slurm run, and the artifact that proves the pipeline still works is the
regression gate in `psych_asr/evaluate/regression.py`, run by `stage1c_join.sbatch` against
the job-2032471 fixture.

Run from the repo root:

    conda activate <venvs>/asr_env && python -m pytest tests -q       # everything
    conda activate <venvs>/diar_eval_env && python -m pytest tests -q # skips the whisperx ones

`test_join.py` needs whisperx and skips itself where it is absent, which is what lets the
same suite run in the torch-free scoring env.
