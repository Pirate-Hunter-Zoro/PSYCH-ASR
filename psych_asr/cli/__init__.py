"""Command-line entry points, one module per job step.

Each is run as `python -m psych_asr.cli.<name>` FROM THE REPO ROOT. Nothing is pip
installed into any env, so the repo root landing on sys.path is what makes the import
work, and that is what "submit from the repo root" has always meant.

Every module here is argument parsing, path handling and printing only. The work itself
lives in the library modules, so a step can be exercised from a test without a parser.

Empty on purpose -- see psych_asr/__init__.py for why nothing is re-exported.
"""
