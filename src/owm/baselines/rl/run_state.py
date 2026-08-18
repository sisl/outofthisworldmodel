"""Run-directory persistence: what a crashed run needs to resume.

Layout of a run dir:
    config.yaml                     resolved hydra config (written at launch)
    env_config.yaml                 concrete env config the run trained on
    wandb_run_id.txt                wandb id, so resume reattaches to the run
    checkpoints/model_<N>_steps.zip           SB3 CheckpointCallback output
    checkpoints/model_replay_buffer_<N>_steps.pkl   (off-policy algos)
    checkpoints/model_vecnormalize_<N>_steps.pkl
    final_model.zip / vecnormalize.pkl        end-of-training artifacts
    final_replay_buffer.pkl         off-policy buffer beside the finals
    final_steps.txt                 num_timesteps the finals were saved at
"""

from __future__ import annotations

import re
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

CHECKPOINT_DIR = "checkpoints"
RUN_CONFIG = "config.yaml"
NAME_PREFIX = "model"
FINAL_MODEL = "final_model.zip"
FINAL_VECNORM = "vecnormalize.pkl"
# Gigabytes of transitions that only a local resume can use: never uploaded.
FINAL_REPLAY_BUFFER = "final_replay_buffer.pkl"
FINAL_STEPS = "final_steps.txt"
_WANDB_ID_FILE = "wandb_run_id.txt"
_STEP_RE = re.compile(rf"^{NAME_PREFIX}_(\d+)_steps\.zip$")


def load_run_config(run_dir: Path) -> DictConfig | None:
    """The run's resolved hydra config, or None on a run dir predating it.

    None rather than an error: `env_config.yaml` is the record a consumer
    actually needs, and a run that has one but no saved hydra config is still
    evaluable — the caller falls back for the handful of fields this file is
    the only source of.
    """
    path = run_dir / RUN_CONFIG
    return OmegaConf.load(path) if path.exists() else None


def save_wandb_id(run_dir: Path, run_id: str) -> None:
    (run_dir / _WANDB_ID_FILE).write_text(run_id + "\n")


def load_wandb_id(run_dir: Path) -> str | None:
    path = run_dir / _WANDB_ID_FILE
    return path.read_text().strip() if path.exists() else None


def save_final_steps(run_dir: Path, steps: int) -> None:
    """Record what num_timesteps the finals hold.

    Read back on resume to tell a finished run's finals from an older periodic
    checkpoint; the alternative, loading final_model.zip just to read its
    counter, costs a full model deserialization on every resume.
    """
    (run_dir / FINAL_STEPS).write_text(f"{int(steps)}\n")


def load_final_steps(run_dir: Path) -> int | None:
    path = run_dir / FINAL_STEPS
    return int(path.read_text().strip()) if path.exists() else None


def clear_final_steps(run_dir: Path) -> None:
    """Withdraw the marker before rewriting the finals it vouches for.

    The count describes the artifact set as a whole, so it stops being true
    the moment the first of those files is replaced: a crash partway through
    an extension's re-save would otherwise leave the old count blessing a mix
    of two legs' finals.
    """
    (run_dir / FINAL_STEPS).unlink(missing_ok=True)


def checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """Every checkpoint zip in the run dir, highest step count first."""
    ckpt_dir = run_dir / CHECKPOINT_DIR
    if not ckpt_dir.is_dir():
        return []
    found = [
        (int(match.group(1)), path)
        for path in ckpt_dir.iterdir()
        if (match := _STEP_RE.match(path.name))
    ]
    return sorted(found, key=lambda pair: pair[0], reverse=True)


def latest_checkpoint(run_dir: Path) -> Path | None:
    found = checkpoints(run_dir)
    return found[0][1] if found else None


def checkpoint_steps(ckpt: Path) -> int | None:
    match = _STEP_RE.match(ckpt.name)
    return int(match.group(1)) if match else None


def missing_siblings(ckpt: Path, need_replay_buffer: bool) -> list[str]:
    """Which of the checkpoint's companion files are absent.

    CheckpointCallback writes the siblings after the zip, so a run killed
    mid-save leaves a checkpoint that cannot be resumed from.
    """
    missing = []
    if vecnormalize_for(ckpt) is None:
        missing.append("vecnormalize sibling")
    if need_replay_buffer and replay_buffer_for(ckpt) is None:
        missing.append("replay_buffer sibling")
    return missing


def latest_complete_checkpoint(run_dir: Path, need_replay_buffer: bool) -> Path | None:
    """Highest-step checkpoint that still has every file a resume needs."""
    for _, path in checkpoints(run_dir):
        if not missing_siblings(path, need_replay_buffer):
            return path
    return None


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
