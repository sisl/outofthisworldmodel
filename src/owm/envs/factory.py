"""Build owm-envs docking envs for SB3 from the hydra environments config.

Which environment of the suite a config describes is carried by one reserved
key, `env_name`, alongside the task fields -- "iss" when absent, so every
config written before the suite existed still means the env it always meant.
The name is a key into owm-envs' own `ENV_REGISTRY`, which is what says both
which config class validates the remaining fields and which gym.Env flies
them. Nothing here holds a second table of that correspondence: an env added
upstream is reachable from this repo as soon as its config group file names
it.
"""

from __future__ import annotations

import os

# The env dynamics are JAX on CPU; the RL learner is torch. Default JAX to
# CPU so env workers never grab GPU memory (an explicit JAX_PLATFORMS wins).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from typing import TYPE_CHECKING

import gymnasium as gym
import torch
from gymnasium.envs.registration import load_env_creator
from gymnasium.wrappers import RescaleAction
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
from owm_envs.envs import ENV_REGISTRY
from owm_envs.envs.common.config import BaseTaskConfig
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

if TYPE_CHECKING:
    from owm.envs.resnet_obs import FrozenResnetExtractor

OBS_MODES = ("vector", "vector_resnet")

# The reserved key naming which env of the suite a config group describes, and
# what it means when absent. Reserved because everything else in an
# `environments` group is a field of the task config itself, which is
# extra="forbid": this key has to be lifted out before validation and put back
# on the way to a worker (see `env_conf_dict`).
ENV_NAME_KEY = "env_name"
DEFAULT_ENV_NAME = "iss"

# owm.envs.resnet_obs is imported inside the vector_resnet branches below
# rather than here, which is the one sanctioned exception to imports-at-top in
# this package: it pulls in torchvision, and this module is on the critical
# path of every env worker of every run. A vector run has no use for it, and a
# torch pin bumped past its matching torchvision would otherwise stop all
# training rather than just the mode that needs the network.


def env_spec(env_name: str):
    """The upstream EnvSpec for `env_name`, or a ValueError naming the suite."""
    try:
        return ENV_REGISTRY[env_name]
    except KeyError:
        raise ValueError(
            f"unknown {ENV_NAME_KEY} {env_name!r}; owm-envs registers "
            f"{sorted(ENV_REGISTRY)}"
        ) from None


def env_name_of(cfg: BaseTaskConfig) -> str:
    """Which env of the suite `cfg` configures, by its config class.

    The reverse of `env_spec(...).config_cls`. Callers that hold only a
    resolved config -- the video and eval envs, the render preflight -- read
    the env off it here rather than being handed the name separately, so a
    config and the env it is flown on cannot drift apart in transit.

    Exact type, not isinstance: every env's config class extends
    `BaseTaskConfig` and `HCWConfig`-style subclassing is how the suite grows,
    so isinstance would match a base class against a derived config and pick
    whichever env the registry happened to list first. Two envs sharing one
    config class would make the reverse direction genuinely ambiguous, which is
    an upstream registry change rather than anything a caller can fix -- so it
    is raised here rather than resolved by iteration order.
    """
    matches = [spec.name for spec in ENV_REGISTRY.values() if type(cfg) is spec.config_cls]
    if len(matches) > 1:
        raise ValueError(
            f"{type(cfg).__name__} is the config class of more than one env "
            f"owm-envs registers ({matches}), so which env a config of that "
            "class describes cannot be recovered from the config alone"
        )
    if not matches:
        raise ValueError(
            f"{type(cfg).__name__} is not the config class of any env owm-envs "
            f"registers ({sorted(ENV_REGISTRY)})"
        )
    return matches[0]


def env_class(env_name: str) -> type[gym.Env]:
    """The gym.Env class for `env_name`, resolved through its gym registration.

    Via the registry's `gym_id` and gymnasium's own entry-point loader rather
    than an import of the class here, so the suite's registration stays the
    only place a name is tied to a class. `load_env_creator` rather than
    `gym.make`, deliberately: `gym.make` would wrap the env in
    OrderEnforcing/PassiveEnvChecker/TimeLimit, and the env's own max_steps
    already truncates.
    """
    return load_env_creator(gym.spec(env_spec(env_name).gym_id).entry_point)


# Reward fields the pre-reshape schema carried and the reshaped one dropped.
# A record holding them predates owm-envs' reward reshape, and no config class
# on the current pin can validate it: the reward it describes no longer exists
# in the code, so the run or dataset behind it cannot be reproduced as
# recorded. Detected up front so the failure is a sentence naming that
# situation rather than an extra_forbidden trace from inside validation.
PRE_RESHAPE_REWARD_KEYS = frozenset({"angular_velocity", "control_effort"})


def task_config_from_yaml(env_name: str, path) -> BaseTaskConfig:
    """Validate a stored task-config record against `env_name`'s config class.

    The one loader for every stored env_config.yaml a run consumes — the
    resume path, the eval and val callbacks, and the from_dataset derivation —
    so a record that predates the reward reshape fails the same loud way at
    every one of them.
    """
    payload = OmegaConf.to_container(OmegaConf.load(path))
    stale = PRE_RESHAPE_REWARD_KEYS & set(payload.get("reward_weights") or {})
    if stale:
        raise SystemExit(
            f"{path} carries pre-reshape reward_weights keys {sorted(stale)}: "
            "the reward they configured no longer exists on this owm-envs pin. "
            "A run recorded under it cannot be resumed or evaluated as the run "
            "it was, and a dataset shipping it needs regenerating under the "
            "current reward."
        )
    return env_spec(env_name).config_cls.model_validate(payload)


def env_config_from_dataset(
    repo_id: str, revision: str | None = None, env_name: str = DEFAULT_ENV_NAME
) -> BaseTaskConfig:
    path = hf_hub_download(
        repo_id=repo_id, filename="env_config.yaml", repo_type="dataset",
        revision=revision,
    )
    return task_config_from_yaml(env_name, path)


def env_config(env_conf: DictConfig | dict) -> BaseTaskConfig:
    """Resolve an `environments` config group to a validated task config."""
    if isinstance(env_conf, DictConfig):
        env_conf = OmegaConf.to_container(env_conf, resolve=True)
    env_conf = dict(env_conf)
    env_name = str(env_conf.pop(ENV_NAME_KEY, DEFAULT_ENV_NAME))
    if "from_dataset_repo" in env_conf:
        return env_config_from_dataset(
            env_conf["from_dataset_repo"],
            env_conf.get("from_dataset_revision"),
            env_name,
        )
    return env_spec(env_name).config_cls.model_validate(env_conf)


def env_conf_dict(cfg: BaseTaskConfig) -> dict:
    """`cfg` as the plain dict `env_config` round-trips back to the same config.

    What training hands its workers and its video/eval envs: the concrete
    config rather than the repo name or group file it came from, with
    `env_name` restored so the far side knows which env to build it on.
    """
    return {**cfg.model_dump(mode="json"), ENV_NAME_KEY: env_name_of(cfg)}


def require_renderable(env_name: str) -> None:
    """Fail at launch if `env_name` has no frames to give.

    The registry's `renderable` is the suite's own answer to "can the video
    path draw this env at all", which is exactly what both frame-consuming
    callers -- rl.obs=vector_resnet and val-episode video capture -- are
    asking.
    Every env in the suite today answers yes, iss-numerical included; this
    guard is what keeps a future one that does not from failing deep inside
    the renderer hours into a run instead of here.
    """
    spec = env_spec(env_name)
    if not spec.renderable:
        raise SystemExit(
            f"env {spec.name!r} cannot be rendered (owm-envs registers it with "
            "renderable=False, meaning it ships no render adapter), so it "
            "supports neither rl.obs=vector_resnet nor val-episode video "
            "capture (set val.video_episodes=0 or val.enabled=false)"
        )


def make_env(
    cfg: BaseTaskConfig,
    seed: int,
    render: bool = False,
    obs_mode: str = "vector",
    extractor: FrozenResnetExtractor | None = None,
    action_repeat: int = 1,
) -> gym.Env:
    if obs_mode not in OBS_MODES:
        raise ValueError(f"unknown obs_mode {obs_mode!r}; expected one of {OBS_MODES}")
    if obs_mode == "vector_resnet" and extractor is None:
        raise ValueError("obs_mode='vector_resnet' needs an extractor to embed with")
    spec = env_spec(env_name_of(cfg))
    # The frame is an observation in this mode, not an optional recording, so
    # the renderer is not the caller's to decline.
    needs_frames = render or obs_mode == "vector_resnet"
    if needs_frames:
        require_renderable(spec.name)
    env = env_class(spec.name)(cfg, render_mode="rgb_array" if needs_frames else None)
    if obs_mode == "vector_resnet":
        from owm.envs.resnet_obs import ResnetObservationWrapper

        env = ResnetObservationWrapper(env, extractor)
    # Innermost of the RL wrappers, so everything above it -- the action
    # rescale, Monitor's episode statistics, SB3's step counter -- works in
    # decisions rather than env steps, which is the cadence the policy is
    # actually trained and evaluated at. Applied here rather than per call
    # site so a val or eval env can never end up running a different cadence
    # than the policy was trained on.
    if action_repeat > 1:
        from owm.baselines.rl.action_repeat import ActionRepeat

        env = ActionRepeat(env, action_repeat)
    # SB3's Gaussian (PPO) samples in raw action units; +-1600 N would need an
    # absurd init std, so policies act in [-1, 1] and the wrapper rescales.
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def preflight_render(cfg: BaseTaskConfig) -> None:
    """Render one frame up front, so a GPU that cannot serve one says so here.

    Rendering fails deep inside pygfx with `Request device failed (3):
    Validation Error / Parent device is lost`, which names neither the GPU nor
    the memory it wanted. Paying one render at launch turns that into a
    sentence, before a trial spends its budget getting there.
    """
    env_name = env_name_of(cfg)
    require_renderable(env_name)
    try:
        env = env_class(env_name)(cfg, render_mode="rgb_array")
        try:
            env.reset(seed=0)
            env.render()
        finally:
            env.close()
    except Exception as exc:
        raise SystemExit(
            "rl.obs=vector_resnet observes a rendered frame every step, and "
            f"this run could not render a single one: {exc}\n"
            "The usual cause is GPU memory pressure. The renderer takes a "
            "Vulkan device worth roughly 1.9 GB per process, on the GPU "
            "regardless of rl.device -- Vulkan does not honour "
            "CUDA_VISIBLE_DEVICES -- and one process per env. Check nvidia-smi "
            "for other tenants, and note that a pixel trial's own workers are "
            "counted there too."
        ) from exc


def make_vec_env(
    env_conf: DictConfig | dict,
    n_envs: int,
    seed: int,
    vec: str = "subproc",
    obs_mode: str = "vector",
    resnet: dict | None = None,
    action_repeat: int = 1,
) -> VecEnv:
    conf_dict = (
        OmegaConf.to_container(env_conf, resolve=True)
        if isinstance(env_conf, DictConfig)
        else env_conf
    )

    def thunk(rank: int):
        def _init() -> gym.Env:
            extractor = None
            if obs_mode == "vector_resnet":
                if vec == "subproc":
                    # This process is one env and nothing else, and n_envs of
                    # them each defaulting to a thread per core turns one
                    # embedding into a machine-wide fight for the CPU. Set only
                    # here: a dummy vec runs _init in the learner's process,
                    # whose own thread count is not this function's to narrow.
                    torch.set_num_threads(1)
                # Built inside the worker: a torch module cannot cross a spawn
                # boundary, and each worker embeds only its own frames anyway.
                from owm.envs.resnet_obs import FrozenResnetExtractor

                extractor = FrozenResnetExtractor(**(resnet or {}))
            return make_env(
                env_config(conf_dict),
                seed=seed + rank,
                obs_mode=obs_mode,
                extractor=extractor,
                action_repeat=action_repeat,
            )

        return _init

    fns = [thunk(i) for i in range(n_envs)]
    if vec == "dummy":
        return DummyVecEnv(fns)
    if vec == "subproc":
        return SubprocVecEnv(fns, start_method="spawn")
    raise ValueError(f"unknown vec type {vec!r}; expected 'subproc' or 'dummy'")
