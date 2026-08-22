"""Single source of truth for account-specific paths.

Mirrors env.sh (the bash equivalent, sourced by job scripts). Nothing here is
hardcoded to a specific Snellius account: PROJECT_ROOT and SCRATCH_DATA_ROOT
are derived from the home directory and $USER, so this module needs no edits
if the project is ever migrated to a new account again.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT        = Path(os.environ.get(
    "MOE_PROJECT_ROOT", str(Path.home() / "projects" / "MoE-ShapeRefine-MedicalSeg")
))
MEDSAM_ROOT          = Path(os.environ.get(
    "MOE_MEDSAM_ROOT", str(Path.home() / "projects" / "MedSAM")
))
TOPLEVEL_DATA_ROOT   = Path(os.environ.get(
    "MOE_DATA_ROOT", str(Path.home() / "projects" / "data")
))

SCRATCH_DATA_ROOT    = Path(os.environ.get("SCRATCH_DATA_ROOT", f"/scratch-shared/{os.environ.get('USER', '')}"))
SCRATCH_DATA         = SCRATCH_DATA_ROOT / "data"
SCRATCH_CHECKPOINTS  = SCRATCH_DATA_ROOT / "checkpoints"
