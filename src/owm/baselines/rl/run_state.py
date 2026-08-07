"""Run-directory persistence: what a crashed run needs to resume.

Layout of a run dir:
    config.yaml                     resolved hydra config (written at launch)
    wandb_run_id.txt                wandb id, so resume reattaches to the run
    checkpoints/model_<N>_steps.zip           SB3 CheckpointCallback output
    checkpoints/model_replay_buffer_<N>_steps.pkl   (off-policy algos)
    checkpoints/model_vecnormalize_<N>_steps.pkl
    final_model.zip / vecnormalize.pkl        end-of-training artifacts
"""

from __future__ import annotations

import re
from pathlib import Path

CHECKPOINT_DIR = "checkpoints"
NAME_PREFIX = "model"
FINAL_MODEL = "final_model.zip"
FINAL_VECNORM = "vecnormalize.pkl"
_WANDB_ID_FILE = "wandb_run_id.txt"
_STEP_RE = re.compile(rf"^{NAME_PREFIX}_(\d+)_steps\.zip$")


def save_wandb_id(run_dir: Path, run_id: str) -> None:
    (run_dir / _WANDB_ID_FILE).write_text(run_id + "\n")


def load_wandb_id(run_dir: Path) -> str | None:
    path = run_dir / _WANDB_ID_FILE
    return path.read_text().strip() if path.exists() else None


def latest_checkpoint(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / CHECKPOINT_DIR
    if not ckpt_dir.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for path in ckpt_dir.iterdir():
        match = _STEP_RE.match(path.name)
        if match:
            steps = int(match.group(1))
            if best is None or steps > best[0]:
                best = (steps, path)
    return best[1] if best else None


def vecnormalize_name_for(ckpt_name: str) -> str | None:
    """Name of the VecNormalize pickle that belongs beside a model zip.

    Name-only, so a remote checkpoint can be resolved to its sibling before
    either file exists locally.
    """
    if ckpt_name == FINAL_MODEL:
        return FINAL_VECNORM
    match = _STEP_RE.match(ckpt_name)
    return f"{NAME_PREFIX}_vecnormalize_{match.group(1)}_steps.pkl" if match else None


def _sibling(ckpt: Path, kind: str) -> Path | None:
    match = _STEP_RE.match(ckpt.name)
    if match is None:
        return None
    steps = match.group(1)
    path = ckpt.parent / f"{NAME_PREFIX}_{kind}_{steps}_steps.pkl"
    return path if path.exists() else None


def replay_buffer_for(ckpt: Path) -> Path | None:
    return _sibling(ckpt, "replay_buffer")


def vecnormalize_for(ckpt: Path) -> Path | None:
    return _sibling(ckpt, "vecnormalize")
