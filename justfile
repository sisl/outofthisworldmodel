set dotenv-load

# Vulkan does not honour CUDA_VISIBLE_DEVICES, so without this the val-episode
# renderer lands on GPU 0 -- an H100 another tenant is using. The RTX PRO
# 6000s (GPUs 2/3) are ours, and are the only adapters matching this name;
# pygfx reads the variable and picks the first match.
#
# The default names THIS host's cards, so it is wrong on any other one, and a
# plain `export X := "..."` in just is unconditional -- it overwrites the
# caller's environment rather than deferring to it, so `PYGFX_WGPU_ADAPTER_NAME
# =... just train-ppo` would silently keep the default and fail with "Adapter
# with name 'RTX PRO 6000' not found". env_var_or_default defers instead. List
# a host's adapters with:
#
#   uv run python -c "import wgpu; [print(a.info['device']) for a in \
#       wgpu.gpu.enumerate_adapters_sync()]"
export PYGFX_WGPU_ADAPTER_NAME := env_var_or_default("PYGFX_WGPU_ADAPTER_NAME", "RTX PRO 6000")

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

# Launch COUNT detached agents for one sweep, round-robin over GPUS for SAC
sweep-fleet SWEEP_ID SWEEP COUNT GPUS="2,3":
    #!/usr/bin/env bash
    set -euo pipefail
    # One agent runs one trial at a time, so covering a search space in a
    # night means running several side by side. The binding resource is CPU:
    # every env worker is a process integrating the dynamics, and a SAC
    # learner asks little of a card that a PPO learner does not touch at all.
    mkdir -p runs/logs
    IFS=',' read -r -a gpus <<< "{{GPUS}}"
    for i in $(seq 0 $(( {{COUNT}} - 1 ))); do
        case "{{SWEEP}}" in
            # Threads for the LEARNER process; env workers pin themselves to
            # one each (owm.envs.factory). Left at torch's default every
            # learner would claim a thread per core, and a fleet of them would
            # spend the night contending rather than training. PPO's learner
            # does its gradient work here, SAC's does it on the card.
            ppo_*) device="" ; threads=4 ;;
            sac_*) device="${gpus[$(( i % ${#gpus[@]} ))]}" ; threads=2 ;;
            *) echo "unknown sweep '{{SWEEP}}': expected ppo_* or sac_*" >&2; exit 1 ;;
        esac
        log="runs/logs/{{SWEEP}}-{{SWEEP_ID}}-$i.log"
        # setsid puts each agent in a session of its own. That session is what
        # sweep-fleet-kill signals for a hard stop; the graceful stop signals
        # the recorded agent pid alone, and the two are different on purpose --
        # see sweep-fleet-stop.
        CUDA_VISIBLE_DEVICES="$device" \
        OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" \
        SWEEP_TRIAL_MAX_SECONDS="${SWEEP_TRIAL_MAX_SECONDS:-14400}" \
            setsid nohup uv run wandb agent \
            "$WANDB_ENTITY/$WANDB_PROJECT/{{SWEEP_ID}}" > "$log" 2>&1 &
        # Per spec, not one file for the whole machine: the lanes are launched
        # separately and are normally running side by side, so a single list
        # would leave no way to stop one without the other.
        echo "$!" >> "runs/logs/sweep-fleet-{{SWEEP}}.pids"
        echo "{{SWEEP}} agent $i: pid $!, CUDA_VISIBLE_DEVICES='$device', $log"
    done

# Stop pulling new trials; the trial in flight runs to its end and reports.
# SWEEP defaults to every lane; name one to stop just that lane.
sweep-fleet-stop SWEEP="*":
    #!/usr/bin/env bash
    set -euo pipefail
    shopt -s nullglob
    files=(runs/logs/sweep-fleet-{{SWEEP}}.pids)
    if [[ ${#files[@]} -eq 0 ]]; then echo "no fleet recorded for '{{SWEEP}}'"; exit 0; fi
    while read -r pid; do
        [[ -n "$pid" ]] || continue
        # The agent, which forwards the signal to the trial it is running
        # (wandb_agent.AgentProcess._forward_signal). The trial takes it
        # cooperatively -- GracefulStopCallback ends training so SB3 still runs
        # on_training_end, and the final eval that is the trial's objective
        # still happens. An agent that already exited is not an error.
        if kill -INT "$pid" 2>/dev/null; then echo "SIGINT -> agent $pid"; fi
    done < <(cat "${files[@]}")
    rm -f "${files[@]}"
    echo "each agent finishes its trial's final eval, then exits"

# Force a fleet down now, losing the in-flight trials' objectives entirely.
# SWEEP defaults to every lane; name one to kill just that lane.
sweep-fleet-kill SWEEP="*":
    #!/usr/bin/env bash
    set -euo pipefail
    shopt -s nullglob
    files=(runs/logs/sweep-fleet-{{SWEEP}}.pids)
    if [[ ${#files[@]} -eq 0 ]]; then echo "no fleet recorded for '{{SWEEP}}'"; exit 0; fi
    while read -r pid; do
        [[ -n "$pid" ]] || continue
        # SIGKILL across the whole session, which is the agent, the trial and
        # the trial's env workers. Nothing gets to report, which is the
        # difference between this and sweep-fleet-stop; reach for it only when
        # the graceful stop is already hung or the deadline has passed.
        sid=$(ps -o sess= -p "$pid" 2>/dev/null | tr -d ' ' || true)
        if [[ -n "$sid" ]] && pkill -KILL -s "$sid"; then
            echo "SIGKILL -> session $sid (agent $pid, its trial and workers)"
        fi
    done < <(cat "${files[@]}")
    rm -f "${files[@]}"

test:
    uv run pytest

test-network:
    uv run pytest -m network
