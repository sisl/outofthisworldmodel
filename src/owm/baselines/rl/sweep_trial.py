"""One trial of a wandb Bayesian sweep: train a budget, report eval return.

The wandb agent runs this module once per trial (see ``sweeps/ppo.yaml`` and
``sweeps/sac.yaml``); it takes the trial's hyperparameters from wandb.config,
maps them onto the hydra config they belong to, and trains inside the run the
agent already opened.

    wandb agent <entity>/<project>/<sweep_id>
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import wandb
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from owm.baselines.rl.run_state import CHECKPOINT_DIR, FINAL_REPLAY_BUFFER
from owm.baselines.rl.sweep_callbacks import EvalReportCallback, TrialTimeoutCallback
from owm.baselines.rl.train import run_training
from owm.envs.factory import iss_config

load_dotenv()

CONF_DIR = str(Path(__file__).resolve().parents[4] / "conf")
SWEEP_RUNS_DIR = Path("runs/sweeps")

TOTAL_TIMESTEPS = 500_000
EVAL_EVERY_STEPS = 100_000
EVAL_EPISODES = 5
FINAL_EVAL_EPISODES = 20
DEFAULT_MAX_SECONDS = 7200.0

# wandb.config carries the swept hyperparameters plus these two, which say how
# to run the trial rather than what to train with. Everything else is an SB3
# keyword argument for the chosen algorithm.
CONTROL_KEYS = frozenset({"algo", "seed"})

# Fixed per algorithm, not swept: they are a property of the machine the sweep
# runs on, not of the policy. cuda:0 is deliberate — GPU 1 is somebody else's.
RESOURCES = {
    "ppo": {"n_envs": 8, "vec": "subproc", "device": "cpu"},
    "sac": {"n_envs": 4, "vec": "subproc", "device": "cuda:0"},
}


def build_cfg(config: Mapping[str, Any], run_dir: Path) -> DictConfig:
    """Map a trial's wandb.config onto the hydra config training reads."""
    algo = str(config.get("algo", ""))
    if algo not in RESOURCES:
        raise SystemExit(
            f"sweep config has algo={algo!r}; expected one of {sorted(RESOURCES)}"
        )

    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"rl={algo}"])

    # Assigned rather than passed as hydra overrides: the sweeps tune SB3
    # arguments the group defaults do not all spell out (clip_range, tau, ...),
    # and both hydra's override grammar and struct mode reject a new key.
    OmegaConf.set_struct(cfg, False)
    for key, value in config.items():
        if key not in CONTROL_KEYS:
            cfg.rl.hyperparams[key] = value
    OmegaConf.set_struct(cfg, True)

    cfg.seed = int(config.get("seed", 0))
    cfg.run_dir = str(run_dir)
    cfg.rl.total_timesteps = TOTAL_TIMESTEPS
    cfg.rl.n_envs = RESOURCES[algo]["n_envs"]
    cfg.rl.vec = RESOURCES[algo]["vec"]
    cfg.rl.device = RESOURCES[algo]["device"]
    # A trial is disposable and never resumed, so periodic checkpoints would
    # only cost disk — SAC's carry a replay buffer of hundreds of MB each.
    cfg.rl.checkpoint.save_freq = TOTAL_TIMESTEPS
    # The agent owns the run; a trial publishes nothing and renders nothing.
    cfg.external_wandb = True
    cfg.hub.upload = False
    cfg.video.enabled = False
    return cfg


def prune_trial_artifacts(run_dir: Path) -> None:
    """Drop what only a resume would want, keeping the final model and stats.

    Tens of trials each leaving a full SAC replay buffer behind fills the disk
    and stops the sweep; nothing resumes a trial, so the buffer and the
    checkpoints it came with are dead weight the moment training ends.
    """
    shutil.rmtree(run_dir / CHECKPOINT_DIR, ignore_errors=True)
    (run_dir / FINAL_REPLAY_BUFFER).unlink(missing_ok=True)


def main() -> None:
    # sync_tensorboard here and nowhere else: it is a wandb.init argument, and
    # this init is the trial's only one, so SB3's TB scalars reach wandb only
    # if it asks for them.
    run = wandb.init(sync_tensorboard=True)
    config = dict(run.config)
    if os.environ.get("WANDB_SWEEP_ID") and not config:
        # Training the group defaults instead would look like a working sweep
        # of identical trials rather than a broken one.
        raise SystemExit(
            "running under a sweep agent but wandb.config is empty: the agent "
            "passed no hyperparameters for this trial"
        )

    run_dir = SWEEP_RUNS_DIR / str(config.get("algo", "unknown")) / run.id
    cfg = build_cfg(config, run_dir)
    callbacks = [
        EvalReportCallback(
            env_conf=iss_config(cfg.environments).model_dump(mode="json"),
            every_steps=EVAL_EVERY_STEPS,
            episodes=EVAL_EPISODES,
            final_episodes=FINAL_EVAL_EPISODES,
            seed=cfg.seed + 10_000,  # never the training seeds
        ),
        TrialTimeoutCallback(
            max_seconds=float(
                os.environ.get("SWEEP_TRIAL_MAX_SECONDS", DEFAULT_MAX_SECONDS)
            )
        ),
    ]

    try:
        run_training(cfg, extra_callbacks=callbacks)
    finally:
        # A trial that raised still leaves its buffer on disk, and the agent
        # will start the next one regardless.
        prune_trial_artifacts(run_dir)
        wandb.finish()


if __name__ == "__main__":
    main()
