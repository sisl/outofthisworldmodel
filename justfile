set dotenv-load

# Vulkan does not honour CUDA_VISIBLE_DEVICES, so without this the val-episode
# renderer lands on GPU 0 -- an H100 another tenant is using. The RTX PRO
# 6000s (GPUs 2/3) are ours, and are the only adapters matching this name;
# pygfx reads the variable and picks the first match.
export PYGFX_WGPU_ADAPTER_NAME := "RTX PRO 6000"

# CUDA's default enumeration is fastest-first by compute capability, which on
# this machine puts the Blackwell RTX cards at CUDA indices 0/1 and the H100s
# at 2/3 -- the reverse of nvidia-smi's PCI order that every GPU number in
# this file (and in people's heads) refers to. Pin PCI order so
# CUDA_VISIBLE_DEVICES=2 means the GPU nvidia-smi calls 2.
export CUDA_DEVICE_ORDER := "PCI_BUS_ID"

# owm-envs ships CUDA jax, but in this repo jax only flies env dynamics, and
# the learner (torch) owns the GPU. Left to itself, XLA initialises a backend
# on every visible card and pre-claims ~75% of each -- including cards other
# tenants are using. Exported here rather than trusted to owm.envs.factory's
# setdefault, which only wins when that module is imported before anything
# touches jax.
export JAX_PLATFORMS := "cpu"

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
        rl.checkpoint.save_freq=1024 hub.upload=false val.enabled=false \
        run_dir=runs/smoke

# Evaluate a checkpoint (local path or hf:org/repo/path.zip)
eval CKPT *ARGS:
    uv run python -m owm.baselines.rl.evaluate eval.checkpoint={{CKPT}} {{ARGS}}

# Create a wandb sweep from sweeps/<SWEEP>.yaml; prints the sweep id
sweep-init SWEEP:
    uv run wandb sweep --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" \
        sweeps/{{SWEEP}}.yaml

# Run one sweep agent (PPO on CPU, SAC on the given GPU); stop it with Ctrl-C
sweep-agent SWEEP_ID SWEEP GPU="2":
    #!/usr/bin/env bash
    set -euo pipefail
    # Keyed on the spec's algo prefix, so every obs mode of an algo lands on
    # the same device class. GPUs 0 and 1 (the H100s) belong to other
    # tenants; a SAC agent takes one of GPUs 2/3 (the RTX PRO 6000s) --
    # pass a trailing 3 to run a second agent beside the first.
    case "{{SWEEP}}" in
        ppo_*) export CUDA_VISIBLE_DEVICES="" ;;
        sac_*) export CUDA_VISIBLE_DEVICES="{{GPU}}" ;;
        *) echo "unknown sweep '{{SWEEP}}': expected ppo_* or sac_*" >&2; exit 1 ;;
    esac
    uv run wandb agent "$WANDB_ENTITY/$WANDB_PROJECT/{{SWEEP_ID}}"

test:
    uv run pytest

test-network:
    uv run pytest -m network
