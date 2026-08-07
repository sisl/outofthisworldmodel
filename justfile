set dotenv-load

# Launch a fresh PPO / SAC training run (extra hydra overrides pass through)
train-ppo *ARGS:
    uv run python -m owm.baselines.rl.train rl=ppo {{ARGS}}

train-sac *ARGS:
    uv run python -m owm.baselines.rl.train rl=sac {{ARGS}}

# Resume a run from its dir (same wandb run); extend_timesteps=<N> raises its budget
resume RUN_DIR *ARGS:
    uv run python -m owm.baselines.rl.train run_dir={{RUN_DIR}} resume=true {{ARGS}}

# Short offline smoke run on tiny settings
smoke:
    WANDB_MODE=offline uv run python -m owm.baselines.rl.train rl=ppo \
        rl.n_envs=2 rl.vec=dummy rl.device=cpu rl.total_timesteps=2048 \
        rl.hyperparams.n_steps=256 rl.hyperparams.batch_size=256 \
        rl.checkpoint.save_freq=1024 hub.upload=false run_dir=runs/smoke

# Evaluate a checkpoint (local path or hf:org/repo/path.zip)
eval CKPT *ARGS:
    uv run python -m owm.baselines.rl.evaluate eval.checkpoint={{CKPT}} {{ARGS}}

test:
    uv run pytest

test-network:
    uv run pytest -m network
