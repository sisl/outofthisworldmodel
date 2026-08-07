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
```

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
