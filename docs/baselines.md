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
just sweep-init ppo_vector               # prints the sweep id
just sweep-agent <sweep_id> ppo_vector   # one agent, CPU only
just sweep-init sac_vector
just sweep-agent <sweep_id> sac_vector   # one agent, on GPU 2
just sweep-agent <sweep_id> sac_vector 3 # a second agent, on GPU 3
```

**Running a fleet.** One agent runs one trial at a time, so covering a search
space overnight means several side by side. `just sweep-fleet <sweep_id>
<spec> <count> [gpus]` launches `count` detached agents, round-robin over the
GPU list for a `sac_*` spec and CPU-only for a `ppo_*` one, logging each to
`runs/logs/<spec>-<sweep_id>-<n>.log` and recording its pid in
`runs/logs/sweep-fleet.pids`. `just sweep-fleet-stop` SIGINTs every recorded
agent; see "Stopping agents at a deadline" below for what that does to the
trial in flight.

The binding resource is CPU, not GPU: every env worker is a process
integrating the dynamics, while a SAC learner asks little of a card and a PPO
learner never touches one. The recipe caps each learner's thread count for the
same reason — env workers pin themselves to one thread each, but left at
torch's default every learner in a fleet claims a thread per core, and they
spend the night contending rather than training.

`sweep-init` creates a wandb Bayesian sweep from `sweeps/<name>.yaml`.
`sweep-agent` pins devices off the spec's `ppo_*`/`sac_*` prefix:
`CUDA_VISIBLE_DEVICES=""` for PPO, and for SAC a single GPU out of 2/3, the
RTX PRO 6000s that are this project's to use — GPUs 0/1, the H100s, belong
to other tenants. It then runs `wandb agent`, which pulls and runs one trial
(`owm.baselines.rl.sweep_trial`) at a time until stopped.

**Objective.** Each spec's `metric.name` is `sweep/eval_mean_return`
(`goal: maximize`), a deterministic-policy eval return, reported
`EVAL_REPORTS` times over the trial plus once more at the end — not the SB3
training loss, because the loss's scale and meaning change with the
hyperparameters being swept (a different `clip_range` or `tau` changes what
the loss *is*, not just its value), while a deterministic eval return measures
the same thing — actual docking behavior — regardless of what produced the
policy.

Mean return rather than `sweep/eval_safe_min_pos_m`, the closure objective, on
two grounds. First, the one reward weight the vector specs search —
`progress` — is bounded in what it can add to a return, so trials stay
comparable. It telescopes over an episode to `progress * (start_range -
end_range) / position_scale_m`, which the 500 m start shell caps at under 9
against episodes scoring in the hundreds; the null-action baseline below
measures the gap between the term on and off at 0.3. The event weights
(`dock_success`, `collision`, `escape`) have no such bound — a trial handed a
bigger dock bonus scores higher for identical flying — and are deliberately
left out of the space for that reason.

Second, closure is weaker here than it looks. Under the soft keep-out zone
`collision_terminates` is false, so `info["collision"]` means "inside the hull
on this step" rather than "this episode ended by crashing" — and closure voids
an episode's credit on that flag, which now reads where an episode happened to
finish rather than whether it survived. Closure stays logged on every report,
so it can still be read after the fact; it is the right objective for a spec
that searches the event weights, which these do not.

**Hyperband pruning.** Both specs set `early_terminate: {type: hyperband,
min_iter: 2}` — a trial must have reported its objective twice before
hyperband will prune it against others at the same rung; that count is in
objective reports, not steps or epochs. `sweep_trial.EVAL_REPORTS` sets how
many reports a trial makes over its horizon (10), so at wandb's default eta
of 3 the rungs land at reports 2 and 6 — 20% and 60% of the horizon. The
first rung is what prices a bad draw: a trial killed there spent a fifth of a
full one. It also sits well past the longest `learning_starts` either spec can
draw, so no trial is banded before it has taken a gradient step.

Val rounds keep their own coarser cadence (`sweep_trial.VAL_ROUNDS`, 4): a
round flies episodes and draws matplotlib figures for each, so it costs
minutes where an eval report costs seconds, and it is read by a human looking
at a trajectory rather than by hyperband looking at a series.

**Trial horizon.** `sweeps/ppo_vector.yaml` pins `trial_timesteps: 4000000`
and `sweeps/sac_vector.yaml` `1500000`. Both are sized from measured
throughput on the training host — 376 env-steps/s for PPO at `n_envs` 8 and
119 for SAC at `n_envs` 4 and the densest update ratio its space allows — so
a full-length trial is roughly three hours either way. Re-measure before
assuming they transfer to another machine.

**Forced settings**, from `sweep_trial.py`'s `build_cfg` (verified against
that source, not paraphrased from memory):

- The environment comes from the spec's own `environments` parameter, falling
  back to `sweep_trial.DEFAULT_ENVIRONMENTS` (`iss_numerical_ports`) for a
  spec that names none. Both vector specs pin
  `iss_numerical_ports_progress_1hz`: the `iss-numerical` env on the random
  5-port goal distribution, with the dense-dominant reward and soft keep-out
  zone, flown at a 1 s control step. The dense terms carry the task, so a
  trial's score reflects how it flies rather than whether a terminal bonus
  survived the discount; and one decision per second over a 360 s episode is
  360 decisions rather than 7,200, the cadence both reference results trained
  at.
- Resources are fixed per algorithm and **not** tunable, overwritten after
  hydra composes the base `rl=<algo>` config: PPO gets `n_envs=8,
  vec=subproc, device=cpu`; SAC gets `n_envs=4, vec=subproc,
  device=cuda:0`.
- `rl.checkpoint.save_freq` is forced to `total_timesteps` — no periodic
  mid-training checkpoints, since a trial is disposable and never resumed
  (SAC's checkpoints each carry a replay buffer of hundreds of MB).
- `external_wandb=true` — the trial trains inside the run the wandb agent
  already opened, rather than opening its own.
- `hub.upload=false` and `val.enabled=false` — trials never publish to the
  HF Hub, and training's own val cadence is off. Instead each vector trial
  schedules plot-only val rounds of its own on the objective-report cadence
  (plus one at the end): `SWEEP_VAL_EPISODES` episodes (env var, default 5)
  at seeds every trial shares (`SWEEP_VAL_SEED`, `+1`, ...), logging the 3D
  trajectory/attitude and reward/force/torque plots under `val/*` — rollouts
  and some matplotlib, cheap enough for a handful of episodes per round.
  Video is opt-in: launch the agent with `SWEEP_VAL_VIDEO=1` to also render
  the first of those episodes to composite and FPV video at the trial's
  mid-point and end, the two like-for-like points to watch across trials —
  rendering draws six views per frame and costs minutes per round, which is
  why it is not the default and stays at one episode.
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

**Stopping agents at a deadline.** `just sweep-fleet-stop` SIGKILLs each
agent's own process group and nothing else. The trial in flight is orphaned
rather than signalled: it keeps running to its horizon and reports its
objective through its own wandb run, while no new trial is ever pulled. Budget
for the in-flight trial to finish, up to `SWEEP_TRIAL_MAX_SECONDS`.

**Do not signal an agent politely instead.** Two things defeat it. The pid the
fleet records is a `uv run` wrapper, which does not pass SIGINT down to the
wandb agent, so signalling that pid alone stops nothing at all. And signalling
wider actively harms the sweep: wandb's agent forwards whatever it receives to
the trial (`wandb_agent.AgentProcess._forward_signal`), where
`GracefulStopCallback` ends training early — and the agent then pulls a *new*
trial rather than exiting. The net effect is to truncate the run in flight and
carry on, which looks like a stop and is the opposite of one. SIGKILL cannot be
caught, so nothing is forwarded and the agent simply stops existing.

`GracefulStopCallback` still matters for every other way a trial is signalled
(`sweep-fleet-kill`, a stray `Ctrl-C`, a shutting-down host). It takes the
signal cooperatively, the way `TrialTimeoutCallback` bounds wall clock: set a
flag, return False from `_on_step`, and let `model.learn()` return as it does
at the end of a horizon so every `on_training_end` runs. Left to Python's
default handler the signal raises `KeyboardInterrupt` inside `model.learn()`,
and SB3 calls `callback.on_training_end()` as a plain statement after its
training loop rather than from a `finally` — so an interrupt skips it,
`EvalReportCallback` never fires its final report, and the trial drops out of
the ranking entirely.

`just sweep-fleet-kill` SIGKILLs the whole session — agent, trial and env
workers together — for when the machine is needed now and the in-flight trials
are expendable.

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
  # Not every run has the key. A trial that diverged -- SAC's actor going NaN
  # is the usual way -- still closes its wandb run through sweep_trial's own
  # finally, so it lands in this list as `finished` with no objective at all.
  # Indexing the summary directly raises KeyError on the first one.
  scored = [r for r in sweep.runs if "sweep/final_mean_return" in r.summary]
  winner = max(scored, key=lambda r: r.summary["sweep/final_mean_return"])
  print(f"{len(scored)}/{len(list(sweep.runs))} runs reported an objective")
  print(winner.id, winner.summary["sweep/final_mean_return"])
  print(dict(winner.config))
  ```

Each file's header comment records its provenance: the sweep id and its
wandb URL, the winning trial's run id and wandb URL, the objective value
(`sweep/eval_mean_return`) it achieved, and the date the sweep concluded.

Final baselines then launch from the frozen config against the production
environment:

```bash
just train-ppo rl=ppo_tuned environments=iss_numerical_ports_progress_1hz \
    environments.reward_weights.progress=-1.0
just train-sac rl=sac_tuned environments=iss_numerical_ports_progress_1hz
```

`progress` is an environment reward weight rather than an SB3 argument, so it
cannot live in a frozen `rl` config and has to be passed. PPO's winner used
−1.0; SAC's used −2.0, which is already the environment's own value, so only
the PPO line carries an override. Launching PPO without it trains a different
reward from the one its hyperparameters were tuned against.

Launch through `just`, not a bare `uv run`: the recipes export
`JAX_PLATFORMS=cpu`, and without it XLA initialises a backend on every visible
card and pre-claims ~75% of each — 73.7 GB on one of these GPUs, measured,
before torch has allocated anything.

`conf/rl/ppo_tuned.yaml` and `conf/rl/sac_tuned.yaml` are the paper-record
configs — any reported final baseline number should trace back to a run
launched from one of them, not from a sweep trial or the untuned defaults.

**Runs in flight (2026-08-14).** `ppo_100M_progress1hz` (wandb `bmet3j79`,
100M steps) and `sac_40M_progress1hz` (wandb `1rsnv2a2`, 40M). The budgets
differ because SAC is bound by gradient steps rather than env stepping —
measured 267 env-steps/s against PPO's 801 at their production widths — so the
two are **not comparable at these budgets**; compare them at a common step
count. Checkpoints land in `runs/<run_name>/checkpoints/` every 2.5M (PPO) and
1M (SAC) steps and are local to the training host, since `runs/` is gitignored
and only the finals upload to the Hub.

**Status.** `conf/rl/ppo_tuned.yaml` and `conf/rl/sac_tuned.yaml` hold winners
frozen on 2026-08-14 from the sweeps against
`iss_numerical_ports_progress_1hz` — the dense reward with the progress term,
at a 1 s control step: `ppo_vector` sweep `nh7k6mai` (winner
`lyric-sweep-21`, −87) and `sac_vector` sweep `e6q3zlrm` (winner
`electric-sweep-34`, −121).

These are the first configurations here to beat a zero-thrust policy, which
scores −405.7 on the same 20 eval episodes. They escape 0–1% of episodes and
close to 0.8% (PPO) and 8% (SAC) of the episode's own start range, against the
~87% the earlier 20M-step runs plateaued at. What separates every good trial
from every bad one is whether it stops leaving the domain: escape rate
correlates with the objective far more strongly than any hyperparameter does.

**They are approach controllers, not docking policies.** `sweep/final_success`
is 0.00 for every trial in both sweeps. The gate is 0.1 m with velocity,
attitude and body-rate constraints on top, and closing to metres is a
different problem from closing to centimetres.

**Seed spread is wider than the gaps between top trials.** The ablation below
measured 200–400 return units between seeds with every hyperparameter fixed,
so a sweep's single winner is not reliably its best configuration. Both frozen
configs therefore take the settings their top two trials independently agreed
on and fill in the rest from the winner; the headers mark which is which.

**A one-dimension ablation retired `target_entropy` as a tuned quantity.**
`sweeps/sac_entropy.yaml` (sweep `wc01nek0`) pinned every other hyperparameter
and swept it over −3, −1, 0 and 3. The spread *within* a setpoint (217 and 399
return units) exceeded the differences *between* setpoints, so the value in
`sac_tuned.yaml` is an admissible one both top trials used rather than a
result. An earlier reading of the broad sweep appeared to show `−3` dominating;
it did not survive more trials, which is what the ablation was built to check.

The pixel sweeps (`ppo_resnet`, `sac_resnet`) remain deferred: at measured
pixel throughput a trial buys ~650k steps in two hours, against the 5–30M
where both reference results see learning emerge.

**Read the objective against a null policy.** A return only means something
next to what doing nothing is worth on the same episodes.
`pocs/null_action_baseline.py` flies the final report's protocol — 20
episodes at seeds 10,000, 10,001, ... with a constant zero action — and
prints the three numbers a trial is scored on. On
`iss_numerical_ports_progress_1hz` it scores **−405.7** mean return, 0%
success, and 270 m closure against a 100–500 m start shell. Run it whenever
the reward or the control step changes; the figure is a property of both, and
a trial that does not beat it learned nothing.

**Match the episode count to the report you are reading.** The periodic
reports fly `EVAL_EPISODES` (5) and the final one `FINAL_EVAL_EPISODES` (20),
over different seed sets, and the 5-episode set is the easier of the two: the
same null policy scores **−484.8** there. The baseline takes an episode count
for exactly this reason (`pocs/null_action_baseline.py <env> <episodes>`).
Comparing a periodic `sweep/eval_mean_return` against the 20-episode figure
reads as a policy well clear of the baseline when it is not — rank trials by
`sweep/final_mean_return`, which is the 20-episode number.

It also measures what the swept `progress` weight costs the objective's
comparability. The same protocol on `iss_numerical_ports_dense_1hz`, which is
this config with the term off, scores −405.4 — a 0.3 gap, because a policy
that does not close the range telescopes the term to nearly zero. The
`progress` arm of a sweep is therefore ranked against the same scale as the
arm without it.

## Telemetry

Every run and every sweep trial logs `docking/*` metrics (via
`DockingMetricsCallback`, registered unconditionally in `run_training`) on
top of reward: per finished episode, a windowed rate of how it ended
(docked / collision / escaped / truncated) and that episode's closest true
approach to the goal (minimum position, velocity, attitude, and body-rate
error). Reward alone cannot tell a policy that is docking more often from
one that is just colliding less; `docking/*` can.

Training runs additionally log `val/*` (via `ValEpisodeCallback`,
`val.enabled=true` by default, every `val.every_steps` steps): a handful of
known-seed deterministic episodes on the dock task, with composite and FPV
videos of the first episode, 3D relative-frame trajectory plots (start
point, target dock port, and body-axis triads showing attitude — with and
without the triads), and per-step reward, force, and torque traces. The
seeds are fixed at launch, so successive rounds are the same episodes flown
by a progressively trained policy. Envs with `dock.enabled=false` skip these
rounds — there is no dock trajectory to measure.
