"""Build ISS docking envs for SB3 from the hydra environments config."""

from __future__ import annotations

import os

# The env dynamics are JAX on CPU; the RL learner is torch. Default JAX to
# CPU so env workers never grab GPU memory (an explicit JAX_PLATFORMS wins).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss.env import ISSEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv


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


def make_iss_env(cfg: ISSConfig, seed: int, render: bool = False) -> gym.Env:
    env = ISSEnv(cfg, render_mode="rgb_array" if render else None)
    # SB3's Gaussian (PPO) samples in raw action units; +-1600 N would need an
    # absurd init std, so policies act in [-1, 1] and the wrapper rescales.
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def make_vec_env(
    env_conf: DictConfig | dict, n_envs: int, seed: int, vec: str = "subproc"
) -> VecEnv:
    conf_dict = (
        OmegaConf.to_container(env_conf, resolve=True)
        if isinstance(env_conf, DictConfig)
        else env_conf
    )

    def thunk(rank: int):
        def _init() -> gym.Env:
            return make_iss_env(iss_config(conf_dict), seed=seed + rank)

        return _init

    fns = [thunk(i) for i in range(n_envs)]
    if vec == "dummy":
        return DummyVecEnv(fns)
    if vec == "subproc":
        return SubprocVecEnv(fns, start_method="spawn")
    raise ValueError(f"unknown vec type {vec!r}; expected 'subproc' or 'dummy'")
