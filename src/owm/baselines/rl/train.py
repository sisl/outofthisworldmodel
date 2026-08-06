"""Train an SB3 PPO/SAC baseline on the ISS docking env.

    python -m owm.baselines.rl.train rl=ppo
    python -m owm.baselines.rl.train rl=sac run_dir=runs/sac_a seed=1
"""

from __future__ import annotations

from pathlib import Path

import hydra
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

from owm.baselines.rl.run_state import (
    CHECKPOINT_DIR,
    FINAL_MODEL,
    FINAL_VECNORM,
    NAME_PREFIX,
    save_wandb_id,
)
from owm.envs.factory import make_vec_env

load_dotenv()

ALGOS = {"ppo": PPO, "sac": SAC}


def run_training(cfg: DictConfig) -> Path:
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Save resolved, not raw: ${now:...} and ${oc.env:...} would re-resolve to
    # different values when a resume reloads this file.
    OmegaConf.save(
        OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)), run_dir / "config.yaml"
    )

    run_id = wandb.util.generate_id()
    save_wandb_id(run_dir, run_id)
    wandb.init(
        id=run_id,
        entity=cfg.logging.entity,
        project=cfg.logging.project,
        mode=cfg.logging.mode,
        name=run_dir.name,
        dir=str(run_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
        sync_tensorboard=True,  # SB3 writes losses to TB; wandb mirrors them
    )

    venv = make_vec_env(cfg.environments, cfg.rl.n_envs, cfg.seed, vec=cfg.rl.vec)
    # Position obs are O(100 m) while rates are O(1e-3); normalization is
    # load-bearing. Reward normalization also tames the -1e6 collision spike.
    venv = VecNormalize(
        venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=cfg.rl.hyperparams.gamma
    )

    algo_cls = ALGOS[cfg.rl.algo]
    model = algo_cls(
        "MlpPolicy",
        venv,
        seed=cfg.seed,
        device=cfg.rl.device,
        tensorboard_log=str(run_dir / "tb"),
        **OmegaConf.to_container(cfg.rl.hyperparams, resolve=True),
    )

    callbacks = [
        CheckpointCallback(
            # SB3 counts save_freq in per-env steps; divide to get total steps
            save_freq=max(cfg.rl.checkpoint.save_freq // cfg.rl.n_envs, 1),
            save_path=str(run_dir / CHECKPOINT_DIR),
            name_prefix=NAME_PREFIX,
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
    ]

    model.learn(
        total_timesteps=cfg.rl.total_timesteps,
        callback=callbacks,
        reset_num_timesteps=True,
    )

    model.save(run_dir / FINAL_MODEL)
    venv.save(str(run_dir / FINAL_VECNORM))
    venv.close()
    wandb.finish()
    return run_dir


@hydra.main(config_path="../../../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
