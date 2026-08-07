# outofthisworldmodel

Out of this World Model (OWM): reference architecture and baselines for the
OWM ISS environments. This repo is the record for OWM's RL baselines today;
it reserves slots for a general reference architecture and a dreamer-v3
baseline (`conf/model/`, `src/owm/baselines/dreamer/`), which land in a
follow-up plan alongside the world-model side.

## Relationship to quickdraw and owm-envs

- **quickdraw** (world-model library) is planned as a pinned git dependency
  for the world-model side; it is not yet in `pyproject.toml` — integration
  lands in a follow-up plan alongside the owm-v1 and dreamer-v3
  implementations.
- **owm-envs** ships the `ISSEnv` simulator and its 3D render assets, and is
  pinned as a git dependency in `pyproject.toml`.

## Setup

```bash
cp .env.example .env   # fill in HF_TOKEN, OWM_HF_MODEL_REPO, WANDB_* etc.
uv sync
```

Python is pinned to 3.13 (`.python-version`): hydra 1.3.x's CLI is broken on
3.14, since 3.14's argparse eagerly validates help strings that hydra's
lazy `--shell-completion` help object doesn't satisfy.

## Commands

```bash
just train-ppo [ARGS...]     # fresh PPO run   (owm.baselines.rl.train rl=ppo)
just train-sac [ARGS...]     # fresh SAC run   (owm.baselines.rl.train rl=sac)
just resume RUN_DIR [ARGS...]  # resume a crashed/stopped run
just smoke                   # tiny offline PPO run, no hub upload
just eval CKPT [ARGS...]     # evaluate a checkpoint
just sweep-init SWEEP        # create a wandb sweep, print its id
just sweep-agent ID SWEEP    # run one sweep agent (see Sweeps below)
just test                    # pytest, network tests deselected
just test-network            # pytest -m network only
```

All recipes are thin wrappers over hydra entry points; extra `key=value`
overrides pass straight through, e.g.
`just train-ppo rl.total_timesteps=1000000 seed=1`.

To swap the training environment for one derived from a published dataset's
as-run config instead of the committed inline config, add
`environments=from_dataset` (optionally
`environments.from_dataset_repo=org/name`) to any command — training then
follows whatever horizon the data carries. Note: the `-trial` dataset
predates the 360 s horizon change (`max_steps` 12000 vs the current 7200),
so evaluating against it needs a matching env override.

## Sweeps

Bayesian hyperparameter search for both baselines, run by wandb:

```bash
just sweep-init ppo_vector               # prints a sweep id
just sweep-agent <sweep_id> ppo_vector   # one agent; run it under nohup/tmux
just sweep-init sac_vector
just sweep-agent <sweep_id> sac_vector
```

There is one spec per (algorithm, observation mode) pair — `sweeps/<name>.yaml`
— and the intended matrix is `{ppo, sac} x {vector, vector_pixels}`. Only the
two vector specs exist today; the pixel pair ships with the `rl.obs` option.

Each trial trains its pinned horizon (500k steps for the vector specs),
reports a deterministic 5-episode eval five times along the way, and
finishes with a 20-episode one. The sweep maximizes `sweep/eval_mean_return`
— eval return, not a training loss, because losses are not comparable across
hyperparameters (a small clip range or a large `tau` changes what the loss
*means*), while the deterministic return is the same measurement of docking
behaviour whatever produced the policy. Hyperband bands on those periodic
reports, so `min_iter: 3` counts reports, not epochs; the reporting cadence
is derived from the horizon rather than fixed, so a shorter sweep still
produces enough reports to be banded.

The specs live outside `conf/` deliberately: a `conf/sweep/` directory would
show up in hydra's config group discovery as a group nothing ever selects.
The per-trial entry point is `owm.baselines.rl.sweep_trial`, which reads
`wandb.config` and trains under the run the agent already opened
(`external_wandb=true` — see below). It maps keys by a small routing table:

| wandb.config key | goes to |
|---|---|
| `algo` | selects the `rl` config group |
| `trial_timesteps` | `rl.total_timesteps` |
| `obs` | `rl.obs` |
| `seed` | `seed` |
| anything else | `rl.hyperparams.<key>` |

So a spec tunes a new SB3 argument by naming it and nothing else. A routed
key whose target does not exist in the composed config fails the trial
loudly — a spec setting `obs` in a checkout without `rl.obs` stops rather
than training vector observations while reporting that it swept the mode.
A spec that pins no `trial_timesteps` is refused the same way, since it
would otherwise inherit `conf/rl`'s multi-million-step budget.

Eval episodes run five-at-a-time through their own `SubprocVecEnv`, built once
per trial from the run's `env_config.yaml`. One env at a time left the training
workers idle and made a report cost more than the training it reported on. The
width does not change what is measured — same seeds, same raw rewards, same
`normalize_obs` transform — which `pocs/eval_width_benchmark.py` checks by
scoring the same policy at both widths.

Trials write to `runs/sweeps/<algo>/<wandb_run_id>/`. Checkpoints and the
final replay buffer are deleted when the trial ends — nothing resumes a
trial, and a few dozen SAC buffers would fill the disk — leaving the final
model and its VecNormalize stats.

Two bounds keep a trial from running away: its pinned `trial_timesteps`, and
`SWEEP_TRIAL_MAX_SECONDS` (default 7200), whose clock starts when the trial
is set up, not when training does, and which ends training gracefully so the
trial still reports an objective and logs `sweep/timed_out=1`. It is a bound
on *training*, not on the process: the final 20-episode eval and the final
save still run afterwards, deliberately, since a trial with no objective is
worth nothing to the sweep. Budget a few minutes past it. Agents themselves
run until stopped; stop them at the deadline with `Ctrl-C` (SIGINT), which
lets the trial in flight finish its final eval.

Both vector specs fix `seed: 0`, so the search is over hyperparameters at one
training seed and a winner may partly have won on luck; re-run the finalists
across several seeds before believing the ranking. For SAC, `train_freq` and
`gradient_steps` are searched independently, which spans a 64x range of
gradient steps per env step — the cheap and expensive ends of that range are
not given equal wall-clock, so read its results with the time bound in mind.

`just sweep-agent` pins the devices: PPO runs with `CUDA_VISIBLE_DEVICES=""`
(SB3's `MlpPolicy` PPO is faster on CPU anyway), SAC with
`CUDA_VISIBLE_DEVICES="0"` and `rl.device=cuda:0`.

### external_wandb

`external_wandb=true` tells `run_training` that its caller already opened the
wandb run and will close it: it skips `wandb.init`/`wandb.finish` and the
`wandb_run_id.txt` bookkeeping, and logs into the active run instead. Sweep
trials need this because the run the agent creates *is* the trial — a second
run would split the history and hide the objective from the sweep
controller. Since `sync_tensorboard` is a `wandb.init` argument, the caller
owns that too: `sweep_trial` passes `sync_tensorboard=True`. Run dirs are
still written exactly as usual, so a trial's artifacts read like any other
run's; only the resume path is unavailable, which is what makes a trial
disposable.

### Manual publish retry

`train.py` uploads the final model to the HF Hub automatically. If that
upload failed or was skipped, republish a finished run directly:

```bash
uv run python -m owm.baselines.rl.hub <run_dir> [repo_id]
```

`repo_id` defaults to `$OWM_HF_MODEL_REPO`.

## Run-dir layout and resume semantics

```
runs/<run_name>/
  config.yaml                    resolved hydra config, written at launch
  env_config.yaml                 concrete ISSConfig the run trained on
  wandb_run_id.txt                wandb id, so resume reattaches to the run
  checkpoints/model_<N>_steps.zip (+ vecnormalize/replay_buffer siblings)
  final_model.zip / vecnormalize.pkl
  final_replay_buffer.pkl         off-policy buffer, local only (never uploaded)
  final_steps.txt                 num_timesteps the finals hold
```

A resume rebuilds from the final artifacts when they are at least as far
along as the newest complete checkpoint — a finished run's finals sit past
its last checkpoint, and rebuilding from that checkpoint would throw the
difference away. Otherwise it takes the newest checkpoint that still has
every sibling it needs, warning about any newer one a kill left
half-written. A run dir whose checkpoints are *all* incomplete fails
loudly *unless* the finals are usable, in which case they are the source;
only a dir with no checkpoints at all restarts from step 0.

`final_steps.txt` is what makes the finals usable: it is written last and
deleted before any re-save, so it vouches for one whole generation of
artifacts. Finals without it — a run dir predating the marker, or one whose
final save crashed partway through the rewrite — are never resumed from;
those runs fall back to the last checkpoint.

`env_config.yaml` is written once, on the first launch, and every env
worker is handed what it records: a resume trains on that file rather than
re-resolving `environments=from_dataset`, whose ref can move between legs.

`rl.total_timesteps` is the run's *absolute* budget, not a per-invocation
increment: resuming a run that already met its budget is a no-op that
leaves the final artifacts and hub upload untouched. A resume takes its
whole config from the run's own `config.yaml`, so raising the budget needs
`extend_timesteps=<N>` (`rl.total_timesteps=<N>` on a resume is ignored —
it always composes to the `conf/rl/*.yaml` default, which a resume cannot
tell apart from a deliberate request). Resuming a run that crashed before
its first checkpoint restarts training from step 0, but into the *same*
wandb run — wandb may warn about out-of-order steps; that's expected.

If a run fails *after* its finals were written — during the wandb artifact
log or the hub upload — a later resume sees the budget met with both finals
present and skips publishing entirely, so the wandb artifact stays missing;
`python -m owm.baselines.rl.hub <run_dir>` re-publishes the HF half.

## Rendering and video

Video capture (`video.enabled=true`) and eval video
(`eval.video_path=...`) both need owm-envs' `render` extra's 3D assets. A
`uv sync`-installed `owm-envs[render]` currently ships git-lfs pointer
files instead of the real `.glb` assets (an upstream owm-envs packaging
issue), which surfaces as the renderer failing to load an asset. Workaround:
in a separate `outofthisworldmodel-envs` checkout, run `git lfs pull`, then
copy the resolved assets into this venv's
`.venv/lib/*/site-packages/owm_envs/render/resources/`.

## Design

See `docs/superpowers/specs/2026-08-06-owm-repo-design.md` (local working
doc, not committed) for the full design rationale.
