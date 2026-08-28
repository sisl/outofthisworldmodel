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
- **owm-envs** ships the ISS docking environments — `iss`, `iss-hcw` and
  `iss-numerical`, which differ in the dynamics they fly the same task under —
  and their 3D render assets, and is pinned as a git dependency in
  `pyproject.toml`.

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
just eval-matrix CKPT [ARGS...]  # evaluate it per port, under every dock definition
just compare A B [ARGS...]       # difference two eval-matrix result dirs
just promote RUN_DIR [ARGS...]   # keep and publish a run's best checkpoint
just sweep-init SWEEP        # create a wandb sweep, print its id
just sweep-agent ID SWEEP    # run one sweep agent (see Sweeps below)
just test                    # pytest, network tests deselected
just test-network            # pytest -m network only
```

All recipes are thin wrappers over hydra entry points; extra `key=value`
overrides pass straight through, e.g.
`just train-ppo rl.total_timesteps=1000000 seed=1`.

### Environments

`conf/environments/` holds one file per training environment, selected with
`environments=<name>`:

| group | env | observation | notes |
| --- | --- | --- | --- |
| `iss_coop_goal` | `iss` | 25 | the published dataset family's config; the default |
| `iss_coop_goal_ports` | `iss` | 25 | the above plus the five train-split dock ports |
| `iss_coop_goal_ports_heldout` | `iss` | 25 | the two val-only ports, for goal generalization |
| `iss_coop_goal_ports_render224` | `iss` | 25 | `iss_coop_goal_ports` rendered at the ResNet's input size |
| `iss_numerical_ports` | `iss-numerical` | 27 | full ECI propagation with J2–J6, third-body and drag |

A group file names its environment with the reserved `env_name` key, which is
a name in owm-envs' own env registry; absent, it means `iss`. Everything else
in the file is that environment's config class.

`iss_numerical_ports` flies the same task, gates, reward weights and five
train-split ports as `iss_coop_goal_ports`, on perturbed two-vehicle orbital
dynamics instead of rigid-body free-flyer ones. Its observation is the same
13-element relative view plus the same 12-element goal-error block, with a
`[jd, sec]` epoch prefix ahead of them — 27 values rather than 25 — so a
policy config carries over but its `VecNormalize` statistics do not.

To swap the training environment for one derived from a published dataset's
as-run config instead of the committed inline config, add
`environments=from_dataset` (optionally
`environments.from_dataset_repo=org/name`) to any command — training then
follows whatever horizon the data carries. Note: the `-trial` dataset
predates the 360 s horizon change (`max_steps` 12000 vs the current 7200),
so evaluating against it needs a matching env override.

## Evaluating a policy

`just eval` answers one question — mean return and success rate over N
episodes, port drawn at random. `just eval-matrix` answers the ones a run that
rarely docks is actually being asked: which port it can reach, and how close
it got when it missed.

```bash
just eval-matrix runs/best/ppo_70M_near/final_model.zip
just eval-matrix runs/best/ppo_70M_near/final_model.zip \
    eval_matrix.rate_hz=20 eval_matrix.action_repeat=20
```

It flies `eval_matrix.trials` deterministic episodes per port, with
`dock.ports` narrowed to **one port at a time** so that count is exact rather
than an expectation, over all eight ports owm-envs knows. Five of them are the
`iss_numerical_ports` train split; the other three the policy never saw, and
every result carries a `train`/`heldout` tag.

Each episode is scored under twenty-one success definitions — three criteria
(`position`, `position_velocity`, `full`) across seven position tolerances
(0.1, 0.2, 0.5, 1, 2, 5, 10 m), so `full` at 0.1 m is the environment's own
dock gate and the rest relax it by dropping tests or widening the position
bound. The ladder is dense because it costs nothing: scoring happens online
during the rollout, so a tolerance is a comparison against a number already in
hand, not another episode to fly.

One rollout per (port, trial) scores all twenty-one **exactly**. The gates reach an
episode only through termination — observation, dynamics and a deterministic
policy are the same whatever bounds are configured — and a looser definition is
satisfied at or before the armed gate fires, over a trajectory identical up to
that instant. Re-running the environment per definition would cost twenty-one
times as much and produce the same numbers.

Looser is the whole condition, and it is enforced rather than assumed: a
tolerance tighter than the environment's own `dock.max_distance_m` is one the
armed gate ends the approach before reaching, so the run is refused instead of
reporting failures that are an artifact of the gate. Each port is seeded from
its place in owm-envs' `PORTS` table, not from where it sits in the request, so
re-running one port alone reproduces that port's row exactly.

`collision_terminates` is false on these configs, so an episode can clip the
hull on its way to a port that sits on it. Every episode carries
`ever_collided` and every success rate is reported both raw and
collision-voided.

The environment is the run's own `env_config.yaml`, found beside the
checkpoint (name it with `eval_matrix.run_dir=` for a checkpoint fetched from
the hub). `eval_matrix.rate_hz` re-times `dt` and `max_steps` together, holding
the horizon in seconds — it exists because world-model policies will run at
20 Hz. `eval_matrix.action_repeat` is independent: a 1 Hz-trained policy at
`rate_hz=20 action_repeat=20` flies its trained cadence over finer
integration, while `action_repeat=1` asks it for twenty times the decisions.
Both default to the run's own rate and one decision per step.

Results land in `runs/evals/<run>_<timestamp>/`: `episodes.csv` (one row per
episode), `outcomes.csv` (one row per episode × criteria × tolerance),
`summary.csv` (one row per port × criteria × tolerance), `report.md` and
`meta.yaml`. Long form rather than one wide table, so a plot or a table is a
filter rather than a reshape.

### Comparing two policies

```bash
just eval-matrix runs/best/<a>/final_model.zip eval_matrix.out_dir=runs/evals/a
just eval-matrix runs/best/<b>/final_model.zip eval_matrix.out_dir=runs/evals/b
just compare runs/evals/a runs/evals/b --criteria position --tolerance 5
```

The comparison is **paired**. Ports are seeded from their place in owm-envs'
`PORTS` table, so run A's trial 7 on `pirs_nadir` and run B's trial 7 on
`pirs_nadir` are the same seed, the same initial state and the same target —
one episode flown by two policies, not two samples of a population.

That is what makes 50 trials a port enough to say anything. Two independent
proportions of 50 carry a standard error near 0.07 each, so a 0.10 gap between
them is noise. The paired form discards every episode the two policies agreed
on and tests only the disagreements (an exact McNemar test), which is where the
information about a difference actually lives. `p(Holm)` corrects across the
cells of each table, since a dozen uncorrected cells turn up a "significant"
one about once per comparison by construction.

**Across rates.** A world-model policy at 20 Hz with `action_repeat=1` and an
RL baseline at 1 Hz with `action_repeat=1` differ in `dt` and `max_steps` by
twenty, and are still *the same episodes*: `reset` draws its dispersions, its
port and its epoch offset from the seed alone and never from the timing, so the
same seed produces a bit-identical start at either rate. `just compare` allows
the timing fields to differ and says so in its header; `--strict-rate` refuses
them for the equal-cadence reading instead.

Allowed to differ is not assumed to be harmless. Every episode records a
`start_fingerprint` — a digest of its initial true state — and `compare`
refuses to report a difference unless those match episode for episode. What
holds today is a property of owm-envs rather than of this repo, so it is
checked rather than trusted.

#### The result format is a contract

`owm.baselines.rl.results` owns it, and neither the writer nor the reader owns
it. A **second harness** — a world-model policy that loads differently, decides
at its own rate and manages its own horizon — can produce these three files and
be compared against an RL baseline without sharing a line of rollout code:

| file | grain | fields a comparison reads |
| --- | --- | --- |
| `meta.yaml` | one document | `EPISODE_KEYS`, `CADENCE_KEYS`, `format_version`, `harness` |
| `episodes.csv` | one row per episode | `EPISODE_FIELDS` |
| `outcomes.csv` | one row per (episode, criteria, tolerance) | `OUTCOME_FIELDS` |

Ten fields in total. Extra columns and extra meta keys are ignored, and
`summary.csv` and `report.md` are conveniences nothing reads back. A second
harness can also import `dock_criteria` directly and get the twenty-one
definitions and their scoreboard for free — that module holds no env, no model
and no config group.

Three rules make results from two harnesses safe to difference:

- **`start_fingerprint` is not optional.** It is `results.start_fingerprint`
  over the episode's initial *true* state, and it is the only evidence that two
  directories describe the same episodes. A comparison refuses to report a
  difference without it, and refuses again if the digests disagree episode for
  episode. Use the shared function rather than reimplementing the digest.
- **`EPISODE_KEYS` must match**; `CADENCE_KEYS` may differ, and reporting a
  difference across them is the point.
- **The horizon must match.** `dt` and `max_steps` are each free to differ —
  that *is* the rate axis — but their product is how long the policy had to
  reach the port, and a policy given half the time is not a policy that did
  worse. This is checked separately from the cadence, because nothing else
  would catch it.

`format_version` lets a reader refuse a directory written by a newer harness
rather than interpret its columns hopefully.

**One caveat a stateful policy must handle.** `eval_matrix.rollout_port` never
resets the policy between episodes. SB3's `MlpPolicy` is stateless so there is
nothing to reset today, but a vec env auto-resets a finished slot mid-loop —
so a policy carrying recurrent state would begin that slot's next episode with
the previous episode's latent still in it. Nothing in the result format catches
that: the fingerprints would still match, because the *environment* restarted
correctly and only the policy did not, and the comparison would report a real
difference between a policy and a contaminated version of itself.

A harness reusing this rollout loop has to clear that state where
`live &= ~dones` already runs — the `dones` mask is exactly the set of slots to
clear. A harness writing its own loop has to do the same thing in its own. It
is described here rather than implemented because there is no stateful policy
to test it against yet, and it must be right before a recurrent policy's
numbers mean anything.

### Evaluating a world-model policy, or any other

Everything from the rollout outwards is already policy-agnostic. `dock_criteria`
reads `info["goal_error_true"]` and nothing else, and the scoring, the CSVs and
the report never learn what produced an action. What is SB3-specific is exactly
three lines in `eval_matrix.run_eval_matrix`:

```python
model   = ALGOS[algo].load(ckpt, device="cpu")     # SB3 PPO/SAC loader
vecnorm = load_normalizer(ckpt, ...)               # VecNormalize pickle sibling
...
actions, _ = model.predict(norm, deterministic=True)   # in rollout_port
```

To add a second policy family, give it a loader returning any object with a
`predict(obs, deterministic=...) -> (actions, state)` and select on something
recorded in the run — `rl.algo` today, a `policy.kind` key for a world model —
rather than widening `ALGOS`, which is training's table and means "which SB3
class trains this".

**One genuine gap to close first.** SB3's `MlpPolicy` is stateless, so
`rollout_port` never resets the policy between episodes. A world-model policy
carries recurrent latent state, and a vec env auto-resets a finished slot, so
that slot's next episode would begin with the previous one's latent unless the
loop clears it. The place to do that is where `live &= ~dones` already runs: the
`dones` mask is exactly the set of slots whose state must be dropped. Until a
stateful policy exists there is nothing to reset, which is why the hook is
described here rather than written.

**What makes two runs comparable.** Hold `seed`, `ports`, `trials`, `rate_hz`
and the environment record fixed; `meta.yaml` records all five, so two result
directories can be checked for agreement before their `summary.csv` files are
put side by side. Ports are seeded from owm-envs' `PORTS` table rather than
from the request, so two policies evaluated on the same seed fly *the same
episodes* — the comparison is paired, and a per-port difference is a difference
between policies rather than between draws.

**Rate is the trap.** A world-model policy running at 20 Hz and an RL baseline
trained at 1 Hz do not compare at one setting, they compare at two, and the
honest report gives both. `rate_hz=20 action_repeat=20` flies the 1 Hz policy at
its trained cadence over the same integration as the world model — equal
decisions, so the comparison isolates the policy. `rate_hz=20 action_repeat=1`
gives both policies the same 20 Hz control authority — equal authority, so it
measures what each is worth at the rate the system will actually run. The first
flatters the baseline, the second flatters whichever policy was trained at
20 Hz; neither is the comparison on its own.

## Filming episodes for the presentations

`just scout PORT` tags a port's evaluation seeds by the lighting the 360 s
approach flies through (`sunlit`, `eclipse`, `transition`) and prints the
outcome each eval drop recorded for the same seed, so a manifest row can be
picked for a known result. `just film MANIFEST OUT` then flies every row of
the presentation manifest that has an `rl` entry, from the same reset
`eval_matrix` used, writing `OUT/<name>/trajectory.npz` + `meta.json` and,
through owm-envs, `rl_fpv.mp4`, `rl_iso.mp4`, `rl_traj.png`, `rl_traj.mp4`.

    just scout poisk_zenith
    just film ../amos_2026_presentations/media/manifest.yaml \
        ../amos_2026_presentations/media/rollouts --gpu-index 3

Rows fly at the manifest's `rate_hz`/`action_repeat` (20 Hz integration with
the policy's 1 Hz decisions held for 20 steps), which is what makes the truth
video smooth; the 1 Hz evaluation outcome is printed beside each row for
comparison, and `meta.json.start_fingerprint` matches the evaluation's.

`meta.json.produced_by` names the checkpoint the row was flown with, so a
directory filmed over two sessions with two policies can be told apart after
the fact. The `eval=` column is the paper's evaluation drop for the same
`(port, seed)`, read through the drop's `env_docked`/`escaped`/`ever_collided`
flags so it lands in the same vocabulary the film's own outcome uses -- a drop
never says `collision`, it says `truncated` beside a collision flag. Rows
already on disk are skipped, but only after their stored port, seed, cadence
and checkpoint are checked against the manifest: change a row's seed or
`rate_hz`/`action_repeat`, or point `--checkpoint` at another policy, and the
run stops naming the mismatch until `--force` refilms it.

## Promoting a run's best checkpoint

A finished run's best policy is not its last one — PPO's entropy collapses
partway through a long run, and everything after that point is worse than what
came before.

```bash
just promote runs/ppo_70M_near                      # ranks, keeps, publishes
just promote runs/ppo_70M_near --criterion min_pos --no-upload
```

Ranking reads the run's own wandb history rather than flying fresh rollouts,
scoring every checkpoint (and the finals) over the window of history centred
on its own step — half the run's checkpoint spacing by default:

| criterion | series | direction |
| --- | --- | --- |
| `val_return` | `val/mean_return` | maximize (default) |
| `train_return` | `rollout/ep_rew_mean` | maximize |
| `min_pos` | `docking/ep_min_pos_m` | minimize |

All three are printed for every candidate, because they disagree and that
disagreement is informative: return is dominated by shaping cost on a run that
never docks, while closest approach reads only whether the policy closed.

The winner is copied to `runs/best/<name>/` as `final_model.zip` and
`vecnormalize.pkl`, beside the run's `config.yaml`, `env_config.yaml` and a
`promotion.yaml` recording where it came from. Under the finals' names
deliberately: `vecnormalize_name_for` recognises only `final_model.zip` and
`model_<N>_steps.zip`, so a file named for its step and score would lose its
statistics sibling and be refused by every entry point that loads it. Upload
goes to `rl/best/<name>/` on the Hub, clear of the run's own `rl/<run_dir>/`.

`--name` is the published identity and defaults to the run directory's, which
is a working name: `ppo_70M_near` says how long a run was and which shell it
flew, and nothing about the environment, observation mode or goal setup that
produced the policy. A checkpoint outlives the directory it came out of, so
name it for what it is —

```bash
just promote runs/ppo_70M_near --name owm-<env>-<goal>-<algo>-<obs> \
    --repo-id sislaboratory/owm-rl-baselines
```

Every upload rebuilds the repo's root `README.md`, which is what the Hub renders
as its model card — a repo whose files all sit under a path prefix otherwise
shows an empty card and a root listing of one folder, which reads as an empty
repo to anyone who did not upload it. The card is rebuilt from the
`promotion.yaml` records already in the repo rather than from an index kept
beside them, so it cannot fall out of step with what is actually published, and
each record carries the algorithm, observation mode, environment and horizon
that produced its checkpoint.

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
`CUDA_VISIBLE_DEVICES` narrowed to one of this machine's own GPUs (2 by
default; pass a trailing `3` for a second agent) and `rl.device=cuda:0`,
which then means that one visible device. The justfile also exports
`PYGFX_WGPU_ADAPTER_NAME` so val-episode rendering, which Vulkan would
otherwise put on GPU 0 regardless of `CUDA_VISIBLE_DEVICES`, lands on those
same GPUs.

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
  env_config.yaml                 concrete env config the run trained on
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

Val-episode video capture (`val.enabled=true` with `val.video_episodes > 0`,
the default) and eval video (`eval.video_path=...`) both need owm-envs'
`render` extra's 3D assets. A
`uv sync`-installed `owm-envs[render]` currently ships git-lfs pointer
files instead of the real `.glb` assets (an upstream owm-envs packaging
issue), which surfaces as the renderer failing to load an asset. Workaround:
in a separate `outofthisworldmodel-envs` checkout, run `git lfs pull`, then
copy the resolved assets into this venv's
`.venv/lib/*/site-packages/owm_envs/render/resources/`.

## Design

See `docs/superpowers/specs/2026-08-06-owm-repo-design.md` (local working
doc, not committed) for the full design rationale.
