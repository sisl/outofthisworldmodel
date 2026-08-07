"""Build ISS docking envs for SB3 from the hydra environments config."""

from __future__ import annotations

import os

# The env dynamics are JAX on CPU; the RL learner is torch. Default JAX to
# CPU so env workers never grab GPU memory (an explicit JAX_PLATFORMS wins).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import gymnasium as gym
import torch
from gymnasium.wrappers import RescaleAction
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss.env import ISSEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from owm.envs.resnet_obs import FrozenResnetExtractor, ResnetObservationWrapper

OBS_MODES = ("vector", "vector_resnet")


def iss_config_from_dataset(repo_id: str, revision: str | None = None) -> ISSConfig:
    path = hf_hub_download(
        repo_id=repo_id, filename="env_config.yaml", repo_type="dataset",
        revision=revision,
    )
    return ISSConfig.from_yaml(path)


def iss_config(env_conf: DictConfig | dict) -> ISSConfig:
    if isinstance(env_conf, DictConfig):
        env_conf = OmegaConf.to_container(env_conf, resolve=True)
    if "from_dataset_repo" in env_conf:
        return iss_config_from_dataset(
            env_conf["from_dataset_repo"], env_conf.get("from_dataset_revision")
        )
    return ISSConfig.model_validate(env_conf)


def make_iss_env(
    cfg: ISSConfig,
    seed: int,
    render: bool = False,
    obs_mode: str = "vector",
    extractor: FrozenResnetExtractor | None = None,
) -> gym.Env:
    if obs_mode not in OBS_MODES:
        raise ValueError(f"unknown obs_mode {obs_mode!r}; expected one of {OBS_MODES}")
    if obs_mode == "vector_resnet" and extractor is None:
        raise ValueError("obs_mode='vector_resnet' needs an extractor to embed with")
    # The frame is an observation in this mode, not an optional recording, so
    # the renderer is not the caller's to decline.
    needs_frames = render or obs_mode == "vector_resnet"
    env = ISSEnv(cfg, render_mode="rgb_array" if needs_frames else None)
    if obs_mode == "vector_resnet":
        env = ResnetObservationWrapper(env, extractor)
    # SB3's Gaussian (PPO) samples in raw action units; +-1600 N would need an
    # absurd init std, so policies act in [-1, 1] and the wrapper rescales.
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def make_vec_env(
    env_conf: DictConfig | dict,
    n_envs: int,
    seed: int,
    vec: str = "subproc",
    obs_mode: str = "vector",
    resnet: dict | None = None,
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
                extractor = FrozenResnetExtractor(**(resnet or {}))
            return make_iss_env(
                iss_config(conf_dict),
                seed=seed + rank,
                obs_mode=obs_mode,
                extractor=extractor,
            )

        return _init

    fns = [thunk(i) for i in range(n_envs)]
    if vec == "dummy":
        return DummyVecEnv(fns)
    if vec == "subproc":
        return SubprocVecEnv(fns, start_method="spawn")
    raise ValueError(f"unknown vec type {vec!r}; expected 'subproc' or 'dummy'")
