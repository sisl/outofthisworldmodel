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

# Create a wandb sweep from sweeps/<SWEEP>.yaml; prints the sweep id
sweep-init SWEEP:
    uv run wandb sweep --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" \
        sweeps/{{SWEEP}}.yaml

# Run one sweep agent (PPO on CPU, SAC on GPU 0); stop it with Ctrl-C
sweep-agent SWEEP_ID SWEEP:
    #!/usr/bin/env bash
    set -euo pipefail
    # Keyed on the spec's algo prefix, so every obs mode of an algo lands on
    # the same device. SAC gets GPU 0 and only GPU 0: GPU 1 is someone else's.
    case "{{SWEEP}}" in
        ppo_*) export CUDA_VISIBLE_DEVICES="" ;;
        sac_*) export CUDA_VISIBLE_DEVICES="0" ;;
        *) echo "unknown sweep '{{SWEEP}}': expected ppo_* or sac_*" >&2; exit 1 ;;
    esac
    uv run wandb agent "$WANDB_ENTITY/$WANDB_PROJECT/{{SWEEP_ID}}"

test:
    uv run pytest

test-network:
    uv run pytest -m network
