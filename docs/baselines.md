# RL baselines runbook

Reference runs, the sweep lifecycle, the winner-freezing convention, and the
per-episode telemetry that all of it logs. See `README.md`'s Sweeps section
for the deeper mechanics (`external_wandb`, eval-width invariance, run-dir
layout); this is the operational runbook.

## Reference runs

Two fixed-goal 5,000,000-step baselines, trained before the random-port goal
distribution (`iss_coop_goal_ports`) existed, on the single PMA-2 pose from
`environments=iss_coop_goal` with each algorithm's plain `conf/rl/ppo.yaml` /
`conf/rl/sac.yaml` hyperparameters (no tuning): `ppo_iss_coop_goal` and
`sac_iss_coop_goal`. They predate the sweep infra and stand as the
pre-ports reference point, not a tuned result.

Both finished their full 5M-step budget and their final artifacts are
uploaded to the HF Hub repo configured as `OWM_HF_MODEL_REPO`, under
`rl/ppo_iss_coop_goal/` and `rl/sac_iss_coop_goal/` respectively
(`final_model.zip`, `vecnormalize.pkl`, `config.yaml`) — the same
`rl/<run_name>/` layout every run publishes to, per `hub.py`.

## Sweep lifecycle

```bash
just sweep-init ppo_vector             # prints the sweep id
just sweep-agent <sweep_id> ppo_vector # one agent, CPU only
just sweep-init sac_vector
just sweep-agent <sweep_id> sac_vector # one agent, pinned to GPU 0
```

`sweep-init` creates a wandb Bayesian sweep from `sweeps/<name>.yaml`.
`sweep-agent` pins devices off the spec's `ppo_*`/`sac_*` prefix
(`CUDA_VISIBLE_DEVICES=""` for PPO, `CUDA_VISIBLE_DEVICES="0"` for SAC — GPU 1
is someone else's) and runs `wandb agent`, which pulls and runs one trial
(`owm.baselines.rl.sweep_trial`) at a time until stopped.

**Objective.** Each spec's `metric.name` is `sweep/eval_mean_return`
(`goal: maximize`), a deterministic-policy eval return, reported five times
over the trial plus once more at the end — not the SB3 training loss,
because the loss's scale and meaning change with the hyperparameters being
swept (a different `clip_range` or `tau` changes what the loss *is*, not
just its value), while a deterministic eval return measures the same thing
— actual docking behavior — regardless of what produced the policy.

**Hyperband pruning.** Both specs set `early_terminate: {type: hyperband,
min_iter: 3}` — a trial must have reported its objective 3 times before
hyperband will prune it against other trials at the same rung; that count is
in objective reports, not steps or epochs.

**Trial horizon.** Both `sweeps/ppo_vector.yaml` and `sweeps/sac_vector.yaml`
pin `trial_timesteps: 500000`.

**Forced settings**, from `sweep_trial.py`'s `build_cfg` (verified against
that source, not paraphrased from memory):

- `environments=iss_coop_goal_ports` is forced at compose time — every trial
  tunes hyperparameters against the actual random 5-port goal distribution,
  not the single-goal group default.
- Resources are fixed per algorithm and **not** tunable, overwritten after
  hydra composes the base `rl=<algo>` config: PPO gets `n_envs=8,
  vec=subproc, device=cpu`; SAC gets `n_envs=4, vec=subproc,
  device=cuda:0`.
- `rl.checkpoint.save_freq` is forced to `total_timesteps` — no periodic
  mid-training checkpoints, since a trial is disposable and never resumed
  (SAC's checkpoints each carry a replay buffer of hundreds of MB).
- `external_wandb=true` — the trial trains inside the run the wandb agent
  already opened, rather than opening its own.
- `hub.upload=false` and `video.enabled=false` — trials never publish to the
  HF Hub and never capture video.
- Everything in a trial's `wandb.config` besides `algo` (selects the `rl`
  group), `trial_timesteps` (→ `rl.total_timesteps`), `obs` (→ `rl.obs`),
  and `seed` routes straight to `rl.hyperparams.<key>` — so a spec tunes a
  new SB3 argument just by naming it.

**Per-trial timeout.** `TrialTimeoutCallback` bounds *training* wall-clock at
`SWEEP_TRIAL_MAX_SECONDS` (env var; `sweep_trial.py`'s `DEFAULT_MAX_SECONDS
= 7200.0`, i.e. 2 hours, if unset), with the clock starting at trial setup,
not at the first training step. On timeout, `model.learn()` stops
gracefully — the trial's final 20-episode eval and final save still run
afterward, since a trial with no reported objective is worthless to the
sweep. The wall-clock bound is on training only, not the whole trial
process.

**Stopping agents at a deadline.** Run agents under `nohup`/`tmux`; stop them
with `Ctrl-C` (SIGINT) rather than killing the process. SIGINT lets whichever
trial is in flight finish its final eval and save instead of losing that
trial's objective — budget past the deadline for that tail, roughly 8-9
minutes for the in-flight trial to wrap up.

## Winner-freezing convention

When a sweep concludes, its best trial's hyperparameters get copied into a
new committed config, same schema as `conf/rl/ppo.yaml` / `conf/rl/sac.yaml`
(`algo`, `n_envs`, `vec`, `total_timesteps`, `device`,
`checkpoint.save_freq`, `hyperparams: {...}`):

- `conf/rl/ppo_tuned.yaml`
- `conf/rl/sac_tuned.yaml`

**Finding the winner.** The winning trial is whichever run in the sweep has
the highest `sweep/final_mean_return` (the objective's value from its final,
20-episode eval, not an intermediate periodic report).

- *wandb UI:* open the sweep page, go to its Runs tab, sort by the
  `sweep/final_mean_return` column descending, and open the top run's Config
  tab for the hyperparameters to copy into `hyperparams:`.
- *Programmatic:* `wandb.Api()` reads the same data without the UI:

  ```python
  import wandb

  api = wandb.Api()
  sweep = api.sweep("<entity>/<project>/<sweep_id>")
  winner = max(sweep.runs, key=lambda r: r.summary["sweep/final_mean_return"])
  print(winner.id, winner.summary["sweep/final_mean_return"])
  print(dict(winner.config))
  ```

Each file's header comment records its provenance: the sweep id and its
wandb URL, the winning trial's run id and wandb URL, the objective value
(`sweep/eval_mean_return`) it achieved, and the date the sweep concluded.

Final baselines then launch from the frozen config against the production
environment:

```bash
just train-ppo rl=ppo_tuned environments=iss_coop_goal_ports
just train-sac rl=sac_tuned environments=iss_coop_goal_ports
```

`conf/rl/ppo_tuned.yaml` and `conf/rl/sac_tuned.yaml` are the paper-record
configs — any reported final baseline number should trace back to a run
launched from one of them, not from a sweep trial or the untuned defaults.

**Status.** Winners frozen for both vector sweeps (`ppo_vector` sweep
`h4be1smz`, `sac_vector` sweep `a5kxxtk2`) as of 2026-08-08. The pixel
sweeps (`ppo_resnet`, `sac_resnet`) are still running; their winners are not
yet frozen.

## Telemetry

Every run and every sweep trial logs `docking/*` metrics (via
`DockingMetricsCallback`, registered unconditionally in `run_training`) on
top of reward: per finished episode, a windowed rate of how it ended
(docked / collision / escaped / truncated) and that episode's closest true
approach to the goal (minimum position, velocity, attitude, and body-rate
error). Reward alone cannot tell a policy that is docking more often from
one that is just colliding less; `docking/*` can.
