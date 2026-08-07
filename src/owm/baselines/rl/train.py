"""Train an SB3 PPO/SAC baseline on the ISS docking env.

    python -m owm.baselines.rl.train rl=ppo
    python -m owm.baselines.rl.train rl=sac run_dir=runs/sac_a seed=1
    python -m owm.baselines.rl.train run_dir=runs/sac_a resume=true
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

from owm.baselines.rl.hub import upload_run
from owm.baselines.rl.run_state import (
    CHECKPOINT_DIR,
    FINAL_MODEL,
    FINAL_VECNORM,
    NAME_PREFIX,
    latest_checkpoint,
    load_wandb_id,
    replay_buffer_for,
    save_wandb_id,
    vecnormalize_for,
)
from owm.baselines.rl.video import VideoEvalCallback
from owm.envs.factory import iss_config, make_vec_env

load_dotenv()

ALGOS = {"ppo": PPO, "sac": SAC}


def run_training(cfg: DictConfig) -> Path:
    run_dir = Path(cfg.run_dir)
    resume = bool(cfg.resume)

    if resume:
        # The run's own saved config is authoritative: hyperparameters cannot
        # silently diverge from the ones the checkpoint was trained under. Only
        # extend_timesteps carries over from the command line — rl.total_timesteps
        # always composes to its group default, so honouring it here would grow a
        # bare resume's budget to 5M without anyone asking.
        saved = OmegaConf.load(run_dir / "config.yaml")
        saved.resume = True
        if cfg.get("extend_timesteps") is not None:
            saved.rl.total_timesteps = int(cfg.extend_timesteps)
        cfg = saved
    elif run_dir.exists() and any(run_dir.iterdir()):
        # Refuse to write a second run's config and id over an existing run's,
        # which would leave the dir describing one run and holding another's
        # checkpoints.
        raise SystemExit(
            f"run_dir {run_dir} already contains a run; pass resume=true to "
            "continue it or choose a new run_dir"
        )

    # Checked before anything is written or launched: a run that trains for
    # hours and then finds it has nowhere to publish has wasted the run.
    if cfg.hub.upload and not cfg.hub.repo_id:
        raise SystemExit(
            "hub.upload=true but hub.repo_id is empty (set OWM_HF_MODEL_REPO in "
            ".env or pass hub.repo_id=... / hub.upload=false)"
        )

    if resume:
        run_id = load_wandb_id(run_dir)
        if run_id is None:
            raise SystemExit(f"resume=true but {run_dir} has no wandb_run_id.txt")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        # Save resolved, not raw: ${now:...} and ${oc.env:...} would re-resolve to
        # different values when a resume reloads this file.
        OmegaConf.save(
            OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)), run_dir / "config.yaml"
        )

        run_id = wandb.util.generate_id()
        save_wandb_id(run_dir, run_id)

    # No ckpt means the run crashed before its first checkpoint: there is no
    # training state to lose, so it restarts from scratch under the same wandb
    # id and keeps its history attached.
    ckpt = latest_checkpoint(run_dir) if resume else None
    if ckpt is not None:
        # CheckpointCallback writes both siblings alongside every checkpoint, so
        # a missing one means a damaged run dir. Substituting a fresh one would
        # corrupt the resumed run rather than fail it.
        if vecnormalize_for(ckpt) is None:
            raise SystemExit(
                f"{ckpt} has no vecnormalize sibling; resuming would run a trained "
                "policy against fresh normalization statistics"
            )
        if cfg.rl.algo == "sac" and replay_buffer_for(ckpt) is None:
            raise SystemExit(
                f"{ckpt} has no replay_buffer sibling; resuming SAC would restart "
                "from an empty buffer"
            )

    wandb.init(
        id=run_id,
        resume="must" if resume else None,
        entity=cfg.logging.entity,
        project=cfg.logging.project,
        mode=cfg.logging.mode,
        name=run_dir.name,
        dir=str(run_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
        sync_tensorboard=True,  # SB3 writes losses to TB; wandb mirrors them
    )

    # environments=from_dataset names a repo, not an env, so the hydra config
    # alone does not say what was trained on. Record the concrete config the
    # workers will each build, the way published datasets ship theirs.
    iss_config(cfg.environments).to_yaml(run_dir / "env_config.yaml")

    venv = make_vec_env(cfg.environments, cfg.rl.n_envs, cfg.seed, vec=cfg.rl.vec)
    if ckpt is not None:
        venv = VecNormalize.load(str(vecnormalize_for(ckpt)), venv)
    else:
        # Position obs are O(100 m) while rates are O(1e-3); normalization is
        # load-bearing. Reward normalization also tames the -1e6 collision spike.
        venv = VecNormalize(
            venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=cfg.rl.hyperparams.gamma
        )

    algo_cls = ALGOS[cfg.rl.algo]
    if ckpt is not None:
        model = algo_cls.load(
            ckpt, env=venv, device=cfg.rl.device, tensorboard_log=str(run_dir / "tb")
        )
        if cfg.rl.algo == "sac":
            model.load_replay_buffer(replay_buffer_for(ckpt))
    else:
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
    if cfg.video.enabled:
        callbacks.append(VideoEvalCallback(
            env_conf=OmegaConf.to_container(cfg.environments, resolve=True),
            every_steps=cfg.video.every_steps,
            max_frames=cfg.video.max_frames,
            seed=cfg.seed + 10_000,  # never the training seeds
        ))

    # rl.total_timesteps is the run's total budget, but SB3 adds the restored
    # counter to whatever it is given when reset_num_timesteps=False, so a
    # resumed leg must ask only for the steps still outstanding.
    remaining = int(cfg.rl.total_timesteps)
    if ckpt is not None:
        remaining = max(remaining - model.num_timesteps, 0)

    if remaining > 0:
        model.learn(
            total_timesteps=remaining,
            callback=callbacks,
            reset_num_timesteps=ckpt is None,
        )

    # This model was rebuilt from the latest checkpoint, which trails whatever
    # the finished leg saved as final: a re-issued resume (a requeued job, say)
    # would otherwise roll the finals back to the last checkpoint boundary.
    # Absent finals still have to be written — a crash between the last
    # checkpoint and the final save leaves the budget met but nothing final.
    # venv/wandb cleanup must run even if artifact logging or the hub upload
    # raises (e.g. a network error), or the run's env leaks and its wandb
    # history is never flushed; the exception still propagates after cleanup.
    try:
        finals_exist = (run_dir / FINAL_MODEL).exists() and (run_dir / FINAL_VECNORM).exists()
        if remaining == 0 and finals_exist:
            print(
                f"budget already met; final artifacts in {run_dir} left untouched, "
                "wandb artifact log and hub upload skipped"
            )
        else:
            model.save(run_dir / FINAL_MODEL)
            venv.save(str(run_dir / FINAL_VECNORM))

            artifact = wandb.Artifact(name=f"{run_dir.name}-model", type="model")
            artifact.add_file(str(run_dir / FINAL_MODEL))
            artifact.add_file(str(run_dir / FINAL_VECNORM))
            wandb.log_artifact(artifact)
            if cfg.hub.upload and cfg.hub.repo_id:
                url = upload_run(run_dir, cfg.hub.repo_id)
                print(f"[hub] uploaded final model: {url}")
    finally:
        venv.close()
        wandb.finish()

    return run_dir


@hydra.main(config_path="../../../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
