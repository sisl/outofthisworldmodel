set dotenv-load

# Which CUDA devices this checkout may use, as the comma-separated indices
# nvidia-smi prints. The default is the single card every machine with a GPU
# has; set OWM_GPUS in .env on a host with more, or one whose other cards
# belong to someone else. Recipes that place work on a card read this, so it is
# the one setting a new host normally has to change.
OWM_GPUS := env_var_or_default("OWM_GPUS", "0")

# Vulkan does not honour CUDA_VISIBLE_DEVICES, so the renderer picks its own
# adapter and OWM_GPUS cannot steer it. Empty lets pygfx choose, which asks for
# a high-performance adapter and so lands on a discrete GPU rather than the
# llvmpipe software one -- right on a machine whose cards are all yours.
#
# Set it in .env on a shared host, to a substring of the adapter you own: pygfx
# takes the first adapter whose summary contains it, and raises "Adapter with
# name '...' not found" rather than falling back if nothing matches. Note that
# identical cards share a summary, so a name selects a MODEL and not a slot --
# every render context in the fleet lands on the first card matching it. List a
# host's adapters with `just adapters`.
#
# env_var_or_default, not a plain `export X := "..."`, because the latter is
# unconditional in just: it overwrites the caller's environment rather than
# deferring to it, so `PYGFX_WGPU_ADAPTER_NAME=... just train-ppo` would
# silently keep the default.
export PYGFX_WGPU_ADAPTER_NAME := env_var_or_default("PYGFX_WGPU_ADAPTER_NAME", "")

# CUDA enumerates fastest-first by compute capability, so its indices reorder
# themselves around whatever mix of cards a host has and stop agreeing with
# nvidia-smi's PCI order -- which is the order OWM_GPUS, this file, and people's
# heads all mean. Pin PCI order so CUDA_VISIBLE_DEVICES=2 is the GPU nvidia-smi
# calls 2 on every host.
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

# Evaluate a checkpoint per port, under every success definition
eval-matrix CKPT *ARGS:
    uv run python -m owm.baselines.rl.eval_matrix \
        eval_matrix.checkpoint={{CKPT}} {{ARGS}}

# Rank a finished run's checkpoints on its wandb history, keep and publish the best
promote RUN_DIR *ARGS:
    uv run python -m owm.baselines.rl.promote {{RUN_DIR}} {{ARGS}}

# Difference two eval-matrix result dirs, episode by episode
compare RUN_A RUN_B *ARGS:
    uv run python -m owm.baselines.rl.compare {{RUN_A}} {{RUN_B}} {{ARGS}}

# Film the presentation manifest's rl rows: trajectory file, clips and plots per row
film MANIFEST OUT *ARGS:
    uv run python -m owm.baselines.rl.film --manifest {{MANIFEST}} --out {{OUT}} {{ARGS}}

# Tag a port's candidate seeds by lighting and by their recorded eval outcomes
scout PORT *ARGS:
    uv run python -m owm.baselines.rl.scout --port {{PORT}} {{ARGS}}

# Create a wandb sweep from sweeps/<SWEEP>.yaml; prints the sweep id
sweep-init SWEEP:
    uv run wandb sweep --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" \
        sweeps/{{SWEEP}}.yaml

# List this host's render adapters, to pick a PYGFX_WGPU_ADAPTER_NAME from
adapters:
    uv run python -c "import wgpu; [print(a.summary) for a in \
        wgpu.gpu.enumerate_adapters_sync()]"

# Run one sweep agent (vector PPO on CPU, everything else on a GPU); Ctrl-C stops it
sweep-agent SWEEP_ID SWEEP GPU="":
    #!/usr/bin/env bash
    set -euo pipefail
    # Defaults to the first card in OWM_GPUS; name one to place the agent
    # yourself, which is how a second agent is run beside the first.
    gpu="{{GPU}}"; [[ -n "$gpu" ]] || { gpu="{{OWM_GPUS}}"; gpu="${gpu%%,*}"; }
    # Keyed on the spec, most specific first. A vector PPO learner does its
    # gradient work on the CPU and needs no card at all, but a pixel one still
    # does: every obs=vector_resnet worker embeds its frame with a ResNet-18,
    # and that runs on a GPU whatever the learner is on.
    case "{{SWEEP}}" in
        *resnet*) export CUDA_VISIBLE_DEVICES="$gpu" ;;
        ppo_*) export CUDA_VISIBLE_DEVICES="" ;;
        sac_*) export CUDA_VISIBLE_DEVICES="$gpu" ;;
        *) echo "unknown sweep '{{SWEEP}}': expected ppo_* or sac_*" >&2; exit 1 ;;
    esac
    uv run wandb agent "$WANDB_ENTITY/$WANDB_PROJECT/{{SWEEP_ID}}"

# Launch COUNT detached agents for one sweep, round-robin over the given GPUs
sweep-fleet SWEEP_ID SWEEP COUNT GPUS=OWM_GPUS:
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
            #
            # Most specific first: a pixel lane takes a card whatever its algo,
            # because its workers embed frames on one. See sweep-agent.
            ppo_*resnet*) device="${gpus[$(( i % ${#gpus[@]} ))]}" ; threads=4 ;;
            sac_*resnet*) device="${gpus[$(( i % ${#gpus[@]} ))]}" ; threads=2 ;;
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
        # SIGKILL, and the AGENT's process group only. Two reasons it is not a
        # polite SIGINT to the pid.
        #
        # The pid is a `uv run` wrapper, which does not pass SIGINT to the
        # wandb agent underneath it, so signalling it alone stops nothing. And
        # signalling wider is worse than useless: wandb's agent forwards what
        # it receives to the trial (AgentProcess._forward_signal), where
        # GracefulStopCallback ends training early -- and the agent then pulls
        # a NEW trial rather than exiting, so the net effect is to truncate the
        # run in flight and carry on.
        #
        # SIGKILL cannot be caught, so nothing is forwarded. setsid put the
        # agent in a group of its own and the agent puts each trial in another,
        # so this reaches the wrapper and the agent and stops there: the trial
        # is orphaned, keeps running to its horizon, and still reports its
        # objective through its own wandb run. An agent that already exited is
        # not an error.
        if kill -KILL -- "-$pid" 2>/dev/null; then echo "stopped agent $pid"; fi
    done < <(cat "${files[@]}")
    rm -f "${files[@]}"
    echo "no new trials will start; those in flight run to their horizon and report"

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
