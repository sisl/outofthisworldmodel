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

import torch
import wandb
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from hydra.errors import MissingConfigException
from omegaconf import DictConfig, OmegaConf

from owm.baselines.rl.run_state import CHECKPOINT_DIR, FINAL_REPLAY_BUFFER
from owm.baselines.rl.sweep_callbacks import (
    EvalReportCallback,
    GracefulStopCallback,
    TrialTimeoutCallback,
)
from owm.baselines.rl.val_episodes import ValEpisodeCallback
from owm.envs.factory import DEFAULT_ENV_NAME, ENV_NAME_KEY
from owm.baselines.rl.train import run_training

load_dotenv()

CONF_DIR = str(Path(__file__).resolve().parents[4] / "conf")
SWEEP_RUNS_DIR = Path("runs/sweeps")

# How many times a trial reports its objective before the final report. This
# is the resolution hyperband bands on: a spec's min_iter counts reports, and
# wandb's default eta of 3 puts its rungs at min_iter, 3 * min_iter, ... So
# five reports leave room for a single rung at 60% of the horizon, which
# prices a bad draw at most of a full trial. Ten put the first rung at 20%.
#
# A report costs one deterministic eval round -- EVAL_EPISODES episodes -- and
# nothing else, which is seconds against the hours a trial spends training.
EVAL_REPORTS = 10
EVAL_EPISODES = 5
FINAL_EVAL_EPISODES = 20
# Val rounds keep their own, coarser cadence. A round flies episodes and draws
# matplotlib figures for each, so it costs minutes where an eval report costs
# seconds, and it is read by a human looking at a trajectory rather than by
# hyperband looking at a series -- four points over a trial is what that
# reading needs.
VAL_ROUNDS = 4
DEFAULT_MAX_SECONDS = 7200.0

# Every trial's val rounds fly known-seed episodes, seeded SWEEP_VAL_SEED,
# SWEEP_VAL_SEED+1, ... A fixed base rather than the trial's own (possibly
# swept) seed, so every trial in every sweep flies the same episodes and the
# diagnostics read side by side.
#
# The rounds are plot-only by default -- 3D trajectory, reward and control
# traces on the objective-report cadence -- so they cost env rollouts and
# matplotlib, cheap enough to fly a handful of episodes per round; the count
# is an env knob like the trial timeout. Video is opt-in via
# SWEEP_VAL_VIDEO=1, because a rendered round draws six views per frame and
# costs minutes where a plot round costs seconds; opted in, ONE episode (the
# plot rounds' first, seed SWEEP_VAL_SEED) is rendered at the trial's
# mid-point and end, the two like-for-like points to watch across trials --
# one, because video cost scales per episode rendered.
SWEEP_VAL_SEED = 20_000
DEFAULT_SWEEP_VAL_EPISODES = 5
SWEEP_VAL_EPISODES_VAR = "SWEEP_VAL_EPISODES"
SWEEP_VAL_VIDEO_VAR = "SWEEP_VAL_VIDEO"

# How to add a hyperparameter to a sweep: add it to the spec's `parameters`
# (sweeps/<name>.yaml) with a wandb distribution — see
# https://docs.wandb.ai/guides/sweeps/define-sweep-configuration for the
# distribution types. Nothing needs to change here or in the spec's `command`.
# A key not listed in RESERVED_KEYS below routes straight to
# `rl.hyperparams.<key>` and must be a real constructor argument of the
# chosen algorithm's SB3 class — `test_every_swept_parameter_is_a_real_sb3_
# argument` (tests/test_sweep_specs.py) checks every spec against that
# algorithm's signature, so a typo'd or unsupported key fails at test time,
# not hours into a trial.
#
# wandb.config keys that are not SB3 arguments, and the config path each one
# writes instead:
#   algo           -- selects the `rl=<algo>` config group (not routable: it
#                      names a group to compose, not a value to write)
#   environments   -- selects the `environments=<name>` config group (not
#                      routable, same reason)
#   trial_timesteps -> rl.total_timesteps
#   obs            -> rl.obs
#   seed           -> seed
ALGO_KEY = "algo"
# Sweeps tune hyperparameters for the real training distribution, the
# random-port goal one, so that is the default; a pixel spec pins the variant
# of it that renders at the extractor's input size.
ENVIRONMENTS_KEY = "environments"
DEFAULT_ENVIRONMENTS = "iss_numerical_ports"
#   demo_*         -> rl.demo.*  (the pre-flown warm-start block)
#   dock_success   -> environments.reward_weights.dock_success
#   collision_penalty -> environments.reward_weights.collision
#   progress       -> environments.reward_weights.progress
#
# The reward weights are routable because a sweep may legitimately search over
# them, but doing so only means anything against a reward-independent
# objective -- see CLOSURE_OBJECTIVE in sweep_callbacks.
#
# `progress` is the exception that stays comparable under mean return. It
# telescopes over an episode to progress * (start_range - end_range) /
# position_scale_m, so what it can add to a return is bounded by the start
# shell rather than growing with the horizon: at the widest weight a spec
# searches and a 500 m start, under 9 against returns of several hundred. The
# event weights have no such bound -- a trial handed a bigger dock bonus scores
# higher for identical flying -- which is why they are not swept beside it.
ROUTES = {
    "trial_timesteps": "rl.total_timesteps",
    "obs": "rl.obs",
    "seed": "seed",
    "demo_repo_id": "rl.demo.repo_id",
    "demo_split": "rl.demo.split",
    "demo_max_transitions": "rl.demo.max_transitions",
    "demo_protected_fraction": "rl.demo.protected_fraction",
    "demo_bc_steps": "rl.demo.bc_steps",
    "dock_success": "environments.reward_weights.dock_success",
    "collision_penalty": "environments.reward_weights.collision",
    "progress": "environments.reward_weights.progress",
}
RESERVED_KEYS = frozenset({ALGO_KEY, ENVIRONMENTS_KEY, *ROUTES})

_MISSING = object()

# Fixed per algorithm, not swept: they are a property of the machine the sweep
# runs on, not of the policy. cuda:0 means "the one GPU this agent was given":
# `just sweep-agent` narrows CUDA_VISIBLE_DEVICES to a single physical device
# (2 or 3 -- 0 and 1 belong to other tenants), and cuda:0 is that device.
RESOURCES = {
    "ppo": {"n_envs": 8, "vec": "subproc", "device": "cpu"},
    "sac": {"n_envs": 4, "vec": "subproc", "device": "cuda:0"},
}
# Every vector_resnet env renders, and the renderer takes a Vulkan device on
# the GPU worth roughly 1.9 GB per process. Vulkan does not honour
# CUDA_VISIBLE_DEVICES -- without PYGFX_WGPU_ADAPTER_NAME (which the justfile
# exports to point at this machine's own GPUs) every render context lands on
# GPU 0 whatever rl.device says -- so the CPU-learner PPO lane is no cheaper
# than the SAC one. Eight workers plus a five-wide eval pool put ~20 GB of
# one shared GPU behind a single trial, which is what OOM'd five SAC trials
# in a row when another tenant arrived.
PIXEL_N_ENVS = 4


def build_cfg(config: Mapping[str, Any], run_dir: Path) -> DictConfig:
    """Map a trial's wandb.config onto the hydra config training reads."""
    algo = str(config.get("algo", ""))
    if algo not in RESOURCES:
        raise SystemExit(
            f"sweep config has algo={algo!r}; expected one of {sorted(RESOURCES)}"
        )

    if "trial_timesteps" not in config:
        raise SystemExit(
            "sweep config has no trial_timesteps; every spec must pin its own "
            "horizon or trials would silently run the multi-million-step "
            "conf/rl default"
        )
    # A horizon shorter than the vec width trains nothing and reports nothing,
    # so the trial would return an objective from an untrained policy and the
    # sweep would rank it against real ones.
    if int(config["trial_timesteps"]) < EVAL_REPORTS * RESOURCES[algo]["n_envs"]:
        raise SystemExit(
            f"trial_timesteps={config['trial_timesteps']!r} is too short for "
            f"{algo}: it takes at least {EVAL_REPORTS * RESOURCES[algo]['n_envs']} "
            f"steps to report the objective {EVAL_REPORTS} times"
        )

    environments = str(config.get(ENVIRONMENTS_KEY, DEFAULT_ENVIRONMENTS))
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        try:
            cfg = compose(
                config_name="config",
                overrides=[f"rl={algo}", f"environments={environments}"],
            )
        except MissingConfigException as exc:
            # A named group that does not exist is a typo or a spec written
            # against a checkout that lacks the config. Hydra's own error is
            # loud but arrives as a stack trace mid-agent; the trial should
            # say which key was wrong and what the alternatives are.
            available = sorted(p.stem for p in Path(CONF_DIR).glob("environments/*.yaml"))
            raise SystemExit(
                f"sweep config sets {ENVIRONMENTS_KEY}={environments!r}, but "
                f"conf/environments has no such config — available: {available}"
            ) from exc

    # Assigned rather than passed as hydra overrides: the sweeps tune SB3
    # arguments the group defaults do not all spell out (clip_range, tau, ...),
    # and both hydra's override grammar and struct mode reject a new key.
    OmegaConf.set_struct(cfg, False)
    for key, value in config.items():
        if key in (ALGO_KEY, ENVIRONMENTS_KEY):
            # Both were spent on the compose above; writing them into the
            # composed config would replace a whole node with its own name.
            continue
        path = ROUTES.get(key)
        if path is None:
            cfg.rl.hyperparams[key] = value
            continue
        # A routed key names an option that must already exist: unlike a
        # hyperparameter, which SB3 validates the moment the model is built, a
        # typo'd or not-yet-landed option would be created here and quietly
        # ignored by everything downstream. This is what a sweep sweeping obs=
        # hits in a checkout where rl.obs has not landed.
        if OmegaConf.select(cfg, path, default=_MISSING) is _MISSING:
            raise SystemExit(
                f"sweep config sets {key}={value!r}, but this checkout's config "
                f"has no {path} for it to write — that option is not available "
                "yet, so the trial would train something other than what the "
                "sweep asked for"
            )
        OmegaConf.update(cfg, path, value)
    OmegaConf.set_struct(cfg, True)

    cfg.run_dir = str(run_dir)
    cfg.rl.n_envs = (
        PIXEL_N_ENVS if str(cfg.rl.obs) == "vector_resnet" else RESOURCES[algo]["n_envs"]
    )
    cfg.rl.vec = RESOURCES[algo]["vec"]
    cfg.rl.device = RESOURCES[algo]["device"]
    # SB3's get_device falls back to CPU without failing, so an agent launched
    # against the wrong sweep id -- `sweep-agent <sac-id> ppo_vector` exports
    # CUDA_VISIBLE_DEVICES="" -- would train every SAC trial on CPU and report
    # the results as if they were the GPU run that was asked for.
    if str(cfg.rl.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            f"{algo} trials need {cfg.rl.device} but torch reports no CUDA "
            "device. The usual cause is an agent started against another "
            "sweep's id: `just sweep-agent <id> ppo_vector` hides the GPUs, so "
            "check that this agent's spec name matches the sweep it joined."
        )
    # A trial is disposable and never resumed, so periodic checkpoints would
    # only cost disk — SAC's carry a replay buffer of hundreds of MB each.
    cfg.rl.checkpoint.save_freq = int(cfg.rl.total_timesteps)
    # The agent owns the run and a trial publishes nothing. Training's own
    # val cadence is off too: the trial schedules its own two val rounds
    # (mid-point and end) in main(), at a seed every trial shares.
    cfg.external_wandb = True
    cfg.hub.upload = False
    cfg.val.enabled = False
    return cfg


def eval_cadence(total_timesteps: int) -> int:
    """Steps between objective reports, derived so a trial can be banded.

    Fixed at 100k this was fine for a 500k horizon and useless for a shorter
    one: hyperband only bands a trial that has reported min_iter times, so the
    cadence has to follow whatever horizon its spec pinned.
    """
    return max(total_timesteps // EVAL_REPORTS, 1)


def val_cadence(total_timesteps: int) -> int:
    """Steps between val rounds, on their own cadence rather than the eval one."""
    return max(total_timesteps // VAL_ROUNDS, 1)


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
    # Everything past the init goes through finish, including the config checks
    # below: a trial that dies on its way to training still has an open run,
    # and the agent will not start the next one until this one is closed out.
    try:
        config = dict(run.config)
        if os.environ.get("WANDB_SWEEP_ID") and not config:
            # Training the group defaults instead would look like a working
            # sweep of identical trials rather than a broken one.
            raise SystemExit(
                "running under a sweep agent but wandb.config is empty: the "
                "agent passed no hyperparameters for this trial"
            )

        run_dir = SWEEP_RUNS_DIR / str(config.get("algo", "unknown")) / run.id
        cfg = build_cfg(config, run_dir)
        obs_mode = str(cfg.rl.obs)
        resnet = None
        if obs_mode == "vector_resnet":
            # Imported here, not at the top: owm.envs.resnet_obs pulls in
            # torchvision. See the note in owm/envs/factory.py.
            from owm.envs.resnet_obs import extractor_kwargs

            resnet = extractor_kwargs(cfg.rl)
        callbacks = [
            EvalReportCallback(
                run_dir=run_dir,
                every_steps=eval_cadence(int(cfg.rl.total_timesteps)),
                episodes=EVAL_EPISODES,
                final_episodes=FINAL_EVAL_EPISODES,
                seed=cfg.seed + 10_000,  # never the training seeds
                obs_mode=obs_mode,
                resnet=resnet,
                env_name=str(cfg.environments.get(ENV_NAME_KEY, DEFAULT_ENV_NAME)),
            ),
            TrialTimeoutCallback(
                max_seconds=float(
                    os.environ.get("SWEEP_TRIAL_MAX_SECONDS", DEFAULT_MAX_SECONDS)
                )
            ),
            GracefulStopCallback(),
        ]
        if obs_mode == "vector":
            # Val rounds for the trial. Vector trials only: the val env
            # observes vectors, which a vector_resnet policy cannot read
            # (train.py refuses the same combination for the training
            # cadence).
            env_name = str(cfg.environments.get(ENV_NAME_KEY, DEFAULT_ENV_NAME))
            total = int(cfg.rl.total_timesteps)
            video = os.environ.get(SWEEP_VAL_VIDEO_VAR, "").lower() in (
                "1", "true", "yes",
            )
            episodes = int(
                os.environ.get(SWEEP_VAL_EPISODES_VAR, DEFAULT_SWEEP_VAL_EPISODES)
            )
            # Plot-only trajectory diagnostics. The final plot round is owed
            # by whichever callback runs last: the video one when video is on,
            # this one otherwise.
            callbacks.append(
                ValEpisodeCallback(
                    run_dir=run_dir,
                    env_name=env_name,
                    seed=SWEEP_VAL_SEED,
                    episodes=episodes,
                    video_episodes=0,
                    every_steps=val_cadence(total),
                    final=not video,
                )
            )
            if video:
                callbacks.append(
                    ValEpisodeCallback(
                        run_dir=run_dir,
                        env_name=env_name,
                        seed=SWEEP_VAL_SEED,
                        episodes=1,
                        video_episodes=1,
                        at_steps=(total // 2,),
                        final=True,
                    )
                )
        try:
            run_training(cfg, extra_callbacks=callbacks)
        finally:
            # A trial that raised still leaves its buffer on disk, and the
            # agent will start the next one regardless.
            prune_trial_artifacts(run_dir)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
