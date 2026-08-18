"""Evaluate one policy per docking port, under every success definition.

    python -m owm.baselines.rl.eval_matrix \\
        eval_matrix.checkpoint=runs/best/ppo_70M_near/final_model.zip
    python -m owm.baselines.rl.eval_matrix \\
        eval_matrix.checkpoint=runs/best/ppo_70M_near/final_model.zip \\
        eval_matrix.rate_hz=20 eval_matrix.action_repeat=20

`owm.baselines.rl.evaluate` answers one question -- mean return and success
rate on whatever `environments` group is composed, with the port drawn at
random per episode. That cannot say which port a policy can reach, nor how
close it got when it missed, and those are the two things a run that logged
essentially no docks at the shipped 0.1 m gate is actually being asked.

THE ENVIRONMENT is the run's own `env_config.yaml` -- the record of what it
trained on -- not a re-resolved `environments` group, for the same reason
`EvalReportCallback` reads that file: an `environments=from_dataset` ref can
move between two resolutions, and a policy scored on dynamics it never saw is
not being measured, it is being mismatched.

PORTS are overridden one at a time rather than left to the per-episode uniform
draw, so `trials` per port means that many and not that many in expectation. A
port the run's own config names keeps that config's pinned pose; one it does
not is resolved from owm-envs' `PORTS` table by name, which is what makes the
three ports outside the train split reachable at all.

RATE is `dt` and `max_steps` together: `rate_hz` holds the base config's
horizon in seconds and re-times the steps inside it. It exists because
world-model policies will run at 20 Hz. `action_repeat` is separate and
independent -- a policy trained at 1 Hz and flown at `rate_hz=20,
action_repeat=1` issues twenty times the decisions it was trained to, while
`rate_hz=20, action_repeat=20` reproduces its trained cadence over finer
integration. Both are real questions; neither is chosen silently.

One rollout per (port, trial) scores every definition, exactly -- see
`dock_criteria` for why that is sound rather than an approximation.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import hydra
import numpy as np
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from owm_envs.envs.common.docking_ports import PORT_NAMES
from owm_envs.envs.common.goal import GOAL_ERROR_NORM_LABELS
from stable_baselines3.common.vec_env import VecEnv

from owm.baselines.rl.dock_criteria import (
    CRITERIA,
    DEFAULT_TOLERANCES_M,
    DockDefinition,
    DockScoreboard,
    definitions,
)
from owm.baselines.rl.evaluate import load_normalizer, resolve_checkpoint
from owm.baselines.rl.results import FORMAT_VERSION, start_fingerprint
from owm.baselines.rl.run_state import load_run_config
from owm.baselines.rl.train import ALGOS
from owm.envs.factory import (
    DEFAULT_ENV_NAME,
    ENV_NAME_KEY,
    env_conf_dict,
    env_config,
    make_vec_env,
    task_config_from_yaml,
)

load_dotenv()

ENV_RECORD = "env_config.yaml"


@dataclass
class EpisodeRecord:
    """One finished episode: how it ended, and how close it came on the way."""

    port: str
    split: str
    trial: int
    seed: int
    steps: int
    outcome: str
    ep_return: float
    env_docked: bool
    ever_collided: bool
    escaped: bool
    start_pos_m: float
    # Identifies the episode's initial condition, so two result directories can
    # be shown to have flown the same episodes rather than assumed to have.
    start_fingerprint: str
    final: dict[str, float]
    minimum: dict[str, float]
    board: DockScoreboard


def run_dir_for(ckpt: Path, given: str | None) -> Path:
    """The run directory whose env record this checkpoint should be flown on.

    Named explicitly, or found by walking up from the checkpoint: a promoted
    policy sits beside its record, a periodic checkpoint one level under it.
    """
    if given is not None:
        return Path(given)
    for candidate in (ckpt.parent, ckpt.parent.parent):
        if (candidate / ENV_RECORD).exists():
            return candidate
    raise SystemExit(
        f"no {ENV_RECORD} beside {ckpt} or its parent, so there is no record of "
        "the environment this policy trained on. Name the run with "
        "eval_matrix.run_dir=<dir> (a checkpoint fetched from the hub always "
        "needs this)."
    )


def base_env_conf(run_dir: Path, run_cfg: DictConfig | None) -> dict:
    """The run's own task config, as the plain dict a worker is handed.

    The record holds the task config alone, so which env of the suite flies it
    is read off the run's saved hydra config beside it -- absent on a run
    started before the suite, which is exactly a run on `iss`.
    """
    record = run_dir / ENV_RECORD
    if not record.exists():
        raise SystemExit(f"{run_dir} has no {ENV_RECORD} to take the environment from")
    env_name = DEFAULT_ENV_NAME
    if run_cfg is not None:
        env_name = str(
            OmegaConf.select(run_cfg, f"environments.{ENV_NAME_KEY}") or DEFAULT_ENV_NAME
        )
    return env_conf_dict(task_config_from_yaml(env_name, record))


def at_rate(base: dict, rate_hz: float | None) -> dict:
    """`base` re-timed to `rate_hz`, holding its horizon in seconds.

    A policy is compared across rates only if the episodes it flies cover the
    same span of time; re-timing `dt` alone would shorten the horizon by the
    same factor it refines the integration.
    """
    if rate_hz is None:
        return base
    rate = float(rate_hz)
    if rate <= 0.0:
        raise SystemExit(f"eval_matrix.rate_hz must be positive, got {rate_hz}")
    horizon_s = float(base["dt"]) * int(base["max_steps"])
    return {**base, "dt": 1.0 / rate, "max_steps": int(round(horizon_s * rate))}


def for_port(base: dict, port: str) -> dict:
    """`base` with its port set narrowed to `port` alone.

    A port the config already names keeps the pose that config pinned; only a
    port it does not name is resolved from the `PORTS` table, which is the
    entry `resolve_port_entries` gives a bare name.
    """
    pinned = next((entry for entry in base["dock"]["ports"] if entry["name"] == port), None)
    return {**base, "dock": {**base["dock"], "ports": [pinned if pinned is not None else port]}}


def require_observable(task, tolerances_m: list[float]) -> None:
    """Refuse a tolerance the armed gate would end the episode before reaching.

    Scoring one rollout against many definitions is exact only for definitions
    the armed gate cannot hide. Every criteria here DROPS bounds rather than
    tightening them, so position is the only axis that can tighten, and a
    tolerance below `dock.max_distance_m` is exactly the case where the
    environment terminates the approach before the definition could have been
    satisfied -- reporting a failure that says nothing about the policy. A
    disabled gate ends nothing, so every tolerance is observable under it.
    """
    if not task.dock.enabled:
        return
    unobservable = sorted(t for t in tolerances_m if t < task.dock.max_distance_m)
    if unobservable:
        raise SystemExit(
            f"eval_matrix.tolerances_m {unobservable} are tighter than this "
            f"environment's own dock gate ({task.dock.max_distance_m} m), which "
            "ends the episode on contact with it. A rollout the gate stops can "
            "say nothing about a tolerance inside it, so those cells would "
            "report failures that are an artifact of the gate rather than a "
            "fact about the policy. Widen the tolerances, or evaluate against a "
            "config whose dock.max_distance_m is at least as tight."
        )


def goal_error_of(info: dict) -> dict[str, float] | None:
    error = info.get("goal_error_true")
    if error is None:
        return None
    return {key: float(error[key]) for key in GOAL_ERROR_NORM_LABELS}


def _fresh_minima() -> dict[str, float]:
    return {key: float("inf") for key in GOAL_ERROR_NORM_LABELS}


def rollout_port(
    model,
    vecnorm,
    venv: VecEnv,
    port: str,
    split: str,
    trials: int,
    seed: int,
    matrix: tuple[DockDefinition, ...],
    step_cap: int,
) -> list[EpisodeRecord]:
    """`trials` deterministic episodes on `port`, a vec-width at a time."""
    records: list[EpisodeRecord] = []
    width = venv.num_envs
    for first in range(0, trials, width):
        batch = min(width, trials - first)
        # VecEnv.seed hands env i seed+i at the next reset, so trial n is the
        # same episode however wide the pool happens to be.
        venv.seed(seed + first)
        obs = venv.reset()
        reset_infos = list(getattr(venv, "reset_infos", None) or [])

        boards = [DockScoreboard(matrix) for _ in range(width)]
        ep_return = np.zeros(width, dtype=np.float64)
        steps_taken = np.zeros(width, dtype=int)
        collided = np.zeros(width, dtype=bool)
        docked = np.zeros(width, dtype=bool)
        escaped = np.zeros(width, dtype=bool)
        # Only the slots this batch actually asked for. A final partial batch
        # leaves the rest of the pool running episodes nobody requested, and
        # waiting on them would cost a whole extra horizon for nothing.
        live = np.arange(width) < batch
        minima = [_fresh_minima() for _ in range(width)]
        finals: list[dict[str, float] | None] = [None] * width
        starts: list[float | None] = [None] * width
        fingerprints: list[str] = [
            start_fingerprint(reset_infos[index].get("state"))
            if index < len(reset_infos) else ""
            for index in range(width)
        ]

        # The reset state seeds the diagnostics -- where the episode began, and
        # the running minima it is the first sample of -- but is deliberately
        # NOT scored against any definition. The environment checks its dock
        # gate in `step` and never in `reset`, so an episode that begins inside
        # a loose tolerance and leaves it is not a dock at any tolerance, and
        # scoring step 0 would invent a success no armed gate could produce.
        for index in range(width):
            error = goal_error_of(reset_infos[index]) if index < len(reset_infos) else None
            if error is None:
                continue
            starts[index] = error["pos_m"]
            minima[index] = dict(error)
            finals[index] = dict(error)

        step = 0
        while live.any() and step < step_cap:
            norm = vecnorm.normalize_obs(obs) if vecnorm is not None else obs
            actions, _ = model.predict(norm, deterministic=True)
            obs, rewards, dones, infos = venv.step(actions)
            step += 1
            ep_return += rewards * live
            for index in np.flatnonzero(live):
                info = infos[index]
                steps_taken[index] = step
                collided[index] |= bool(info.get("collision"))
                error = goal_error_of(info)
                if error is None:
                    continue
                if starts[index] is None:
                    starts[index] = error["pos_m"]
                boards[index].update(step, error)
                finals[index] = error
                for key in GOAL_ERROR_NORM_LABELS:
                    if error[key] < minima[index][key]:
                        minima[index][key] = error[key]
            for index in np.flatnonzero(live & dones):
                docked[index] = bool(infos[index].get("success"))
                escaped[index] = bool(infos[index].get("escaped"))
            # A vec env auto-resets a finished env, so anything it reports
            # after this belongs to an episode nobody asked for.
            live &= ~dones

        for index in range(batch):
            if docked[index]:
                outcome = "docked"
            elif escaped[index]:
                outcome = "escaped"
            elif live[index]:
                outcome = "capped"
            else:
                outcome = "truncated"
            records.append(
                EpisodeRecord(
                    port=port,
                    split=split,
                    trial=first + index,
                    seed=seed + first + index,
                    steps=int(steps_taken[index]),
                    outcome=outcome,
                    ep_return=float(ep_return[index]),
                    env_docked=bool(docked[index]),
                    ever_collided=bool(collided[index]),
                    escaped=bool(escaped[index]),
                    start_pos_m=float(starts[index]) if starts[index] is not None else float("nan"),
                    start_fingerprint=fingerprints[index],
                    final=finals[index] or _fresh_minima(),
                    minimum=minima[index],
                    board=boards[index],
                )
            )
    return records


def summarize(
    records: list[EpisodeRecord], matrix: tuple[DockDefinition, ...]
) -> list[dict[str, object]]:
    """One row per (port, criteria, tolerance): rates and error statistics."""
    rows: list[dict[str, object]] = []
    ports = sorted({record.port for record in records}, key=PORT_NAMES.index)
    for port in ports:
        flown = [record for record in records if record.port == port]
        for item in matrix:
            fires = [record.board.fired(item) for record in flown]
            hit = [
                record for record, fire in zip(flown, fires) if fire is not None
            ]
            safe = [record for record in hit if not record.ever_collided]
            steps = [fire.step for fire in fires if fire is not None]
            finals = [record.final["pos_m"] for record in flown]
            rows.append({
                "port": port,
                "split": flown[0].split,
                "criteria": item.criteria,
                "tolerance_m": item.tolerance_m,
                "episodes": len(flown),
                "successes": len(hit),
                "success_rate": len(hit) / len(flown),
                "successes_no_collision": len(safe),
                "success_rate_no_collision": len(safe) / len(flown),
                "collision_rate": sum(r.ever_collided for r in flown) / len(flown),
                "mean_steps_to_success": float(np.mean(steps)) if steps else "",
                "mean_final_pos_m": float(np.mean(finals)),
                "median_final_pos_m": float(statistics.median(finals)),
                "mean_min_pos_m": float(np.mean([r.minimum["pos_m"] for r in flown])),
            })
    return rows


def write_csvs(out_dir: Path, records: list[EpisodeRecord], summary: list[dict[str, object]]) -> None:
    episode_fields = [
        "port", "split", "trial", "seed", "steps", "outcome", "ep_return",
        "env_docked", "ever_collided", "escaped", "start_pos_m", "start_fingerprint",
        *(f"final_{key}" for key in GOAL_ERROR_NORM_LABELS),
        *(f"min_{key}" for key in GOAL_ERROR_NORM_LABELS),
    ]
    with (out_dir / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=episode_fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "port": record.port, "split": record.split, "trial": record.trial,
                "seed": record.seed, "steps": record.steps, "outcome": record.outcome,
                "ep_return": record.ep_return, "env_docked": record.env_docked,
                "ever_collided": record.ever_collided, "escaped": record.escaped,
                "start_pos_m": record.start_pos_m,
                "start_fingerprint": record.start_fingerprint,
                **{f"final_{key}": record.final[key] for key in GOAL_ERROR_NORM_LABELS},
                **{f"min_{key}": record.minimum[key] for key in GOAL_ERROR_NORM_LABELS},
            })

    outcome_fields = [
        "port", "split", "trial", "seed", "criteria", "tolerance_m", "fired",
        "fire_step", *(f"fire_{key}" for key in GOAL_ERROR_NORM_LABELS),
    ]
    with (out_dir / "outcomes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=outcome_fields)
        writer.writeheader()
        for record in records:
            for item, fire in record.board.rows():
                writer.writerow({
                    "port": record.port, "split": record.split, "trial": record.trial,
                    "seed": record.seed, "criteria": item.criteria,
                    "tolerance_m": item.tolerance_m, "fired": fire is not None,
                    "fire_step": fire.step if fire else "",
                    **{
                        f"fire_{key}": fire.errors[key] if fire else ""
                        for key in GOAL_ERROR_NORM_LABELS
                    },
                })

    with (out_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def _rate_table(
    title: str,
    groups: list[tuple[str, list[EpisodeRecord]]],
    matrix: tuple[DockDefinition, ...],
    criteria: str,
    tolerances: list[float],
    voided: bool,
) -> list[str]:
    header = " | ".join(f"{tol:g} m" for tol in tolerances)
    lines = [f"### {title}", "", f"| group | {header} |", "|" + " --- |" * (len(tolerances) + 1)]
    for name, flown in groups:
        cells = []
        for tol in tolerances:
            item = next(
                d for d in matrix if d.criteria == criteria and d.tolerance_m == tol
            )
            hit = [
                record for record in flown
                if record.board.fired(item) is not None
                and not (voided and record.ever_collided)
            ]
            cells.append(f"{len(hit) / len(flown):.2f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def report(
    records: list[EpisodeRecord],
    matrix: tuple[DockDefinition, ...],
    criteria: list[str],
    tolerances: list[float],
    mismatches: int,
) -> str:
    ports = sorted({record.port for record in records}, key=PORT_NAMES.index)
    groups: list[tuple[str, list[EpisodeRecord]]] = [
        (port, [r for r in records if r.port == port]) for port in ports
    ]
    for split in ("train", "heldout"):
        flown = [r for r in records if r.split == split]
        if flown:
            groups.append((f"**{split}**", flown))
    groups.append(("**all**", records))

    lines = ["# Evaluation matrix", "", "## Success rate (collision-voided)", ""]
    for name in criteria:
        lines += _rate_table(name, groups, matrix, name, tolerances, voided=True)
    lines += ["## Success rate (raw)", ""]
    for name in criteria:
        lines += _rate_table(name, groups, matrix, name, tolerances, voided=False)

    lines += ["## Approach", "", "| group | n | mean final pos (m) | mean closest pos (m) | "
              "collision rate | escape rate | mean return |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, flown in groups:
        lines.append(
            f"| {name} | {len(flown)} | "
            f"{np.mean([r.final['pos_m'] for r in flown]):.2f} | "
            f"{np.mean([r.minimum['pos_m'] for r in flown]):.2f} | "
            f"{np.mean([r.ever_collided for r in flown]):.2f} | "
            f"{np.mean([r.escaped for r in flown]):.2f} | "
            f"{np.mean([r.ep_return for r in flown]):.1f} |"
        )
    lines += [
        "",
        f"`full` at the env's own tolerance disagreed with the environment's own "
        f"dock gate on {mismatches} of {len(records)} episodes.",
        "",
    ]
    return "\n".join(lines)


def run_eval_matrix(cfg: DictConfig) -> dict:
    settings = cfg.eval_matrix
    if settings.checkpoint is None:
        raise SystemExit("eval_matrix.checkpoint is not set; name a policy to evaluate")

    ckpt = resolve_checkpoint(str(settings.checkpoint))
    run_dir = run_dir_for(ckpt, settings.run_dir)
    run_cfg = load_run_config(run_dir)
    base = at_rate(base_env_conf(run_dir, run_cfg), settings.rate_hz)
    task = env_config(base)

    criteria = list(settings.criteria) if settings.criteria else list(CRITERIA)
    tolerances = (
        [float(t) for t in settings.tolerances_m]
        if settings.tolerances_m
        else list(DEFAULT_TOLERANCES_M)
    )
    matrix = definitions(task.dock, criteria, tolerances)
    require_observable(task, tolerances)

    trained = [entry["name"] for entry in base["dock"]["ports"]]
    ports = (
        list(PORT_NAMES)
        if str(settings.ports) == "all"
        else [str(name) for name in settings.ports]
    )
    if not ports:
        raise SystemExit("eval_matrix.ports names no port, so there is nothing to fly")
    unknown = [name for name in ports if name not in PORT_NAMES]
    if unknown:
        raise SystemExit(f"unknown port(s) {unknown}; owm-envs knows {list(PORT_NAMES)}")
    for name in ("trials", "n_envs", "action_repeat"):
        if int(settings[name]) < 1:
            raise SystemExit(f"eval_matrix.{name} must be >= 1, got {settings[name]}")

    # A run dir predating the saved hydra config still has its env record, and
    # `base_env_conf` already reads that case; the policy's own settings then
    # fall back to the composed config rather than crashing on a None lookup.
    algo = str((run_cfg and OmegaConf.select(run_cfg, "rl.algo")) or cfg.rl.algo)
    obs_mode = str((run_cfg and OmegaConf.select(run_cfg, "rl.obs")) or "vector")
    resnet = None
    if obs_mode == "vector_resnet":
        from owm.envs.resnet_obs import extractor_kwargs

        resnet = extractor_kwargs(run_cfg.rl)

    model = ALGOS[algo].load(ckpt, device="cpu")
    vecnorm = load_normalizer(ckpt, bool(settings.allow_unnormalized))

    # Named for the run and the checkpoint within it, not the checkpoint alone:
    # a periodic checkpoint's own parent directory is called `checkpoints` in
    # every run there is.
    out_dir = Path(
        settings.out_dir
        if settings.out_dir is not None
        else f"runs/evals/{run_dir.name}_{ckpt.stem}_"
             f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    trials = int(settings.trials)
    # Never wider than the work: a pool larger than the trial count would spend
    # a process per slot flying episodes nothing records.
    width = min(int(settings.n_envs), trials)
    step_cap = int(task.max_steps) + 1
    records: list[EpisodeRecord] = []
    for port in ports:
        venv = make_vec_env(
            for_port(base, port),
            n_envs=width,
            # Overwritten per batch by VecEnv.seed; this only seeds construction.
            seed=int(settings.seed),
            vec=str(settings.vec),
            obs_mode=obs_mode,
            resnet=resnet,
            action_repeat=int(settings.action_repeat),
        )
        try:
            # Keyed on the port's place in owm-envs' own PORTS table, never on
            # where it happens to sit in this request. Ports are seeded apart
            # so no two fly the same episode, and reading one port's row means
            # re-running that port alone -- which under a request-relative
            # offset would draw a different 50 episodes and disagree with the
            # table it was checking.
            port_seed = int(settings.seed) + PORT_NAMES.index(port) * 10_000
            flown = rollout_port(
                model, vecnorm, venv, port,
                "train" if port in trained else "heldout",
                trials, port_seed, matrix, step_cap,
            )
        finally:
            venv.close()
        records.extend(flown)
        docked = sum(record.env_docked for record in flown)
        print(
            f"{port:>20}  docked={docked:>3}/{trials}  "
            f"final_pos={np.mean([r.final['pos_m'] for r in flown]):8.2f} m  "
            f"closest={np.mean([r.minimum['pos_m'] for r in flown]):8.2f} m",
            flush=True,
        )

    # The env's gate is float32 and goal_error_true is float64, so a state
    # sitting exactly on a bound can be read either way: counted, not asserted.
    armed = next(
        (
            item for item in matrix
            if item.criteria == "full" and item.tolerance_m == task.dock.max_distance_m
        ),
        None,
    )
    mismatches = (
        sum(
            1 for record in records
            if record.env_docked != (record.board.fired(armed) is not None)
        )
        if armed is not None
        else 0
    )

    summary = summarize(records, matrix)
    write_csvs(out_dir, records, summary)
    text = report(records, matrix, criteria, tolerances, mismatches)
    (out_dir / "report.md").write_text(text)
    OmegaConf.save(
        OmegaConf.create({
            "format_version": FORMAT_VERSION,
            "harness": "owm.baselines.rl.eval_matrix",
            "checkpoint": str(settings.checkpoint),
            "resolved_checkpoint": str(ckpt),
            "run_dir": str(run_dir),
            "algo": algo,
            "obs": obs_mode,
            "ports": ports,
            "train_ports": trained,
            "trials": trials,
            "seed": int(settings.seed),
            "rate_hz": settings.rate_hz,
            "dt": float(base["dt"]),
            "max_steps": int(base["max_steps"]),
            "action_repeat": int(settings.action_repeat),
            "criteria": criteria,
            "tolerances_m": tolerances,
            "gate_mismatches": mismatches,
            "written": f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
            "env_config": base,
        }),
        out_dir / "meta.yaml",
    )
    print(f"\n{text}\nwrote {out_dir}")
    return {"out_dir": str(out_dir), "episodes": len(records), "summary": summary}


@hydra.main(config_path="../../../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_eval_matrix(cfg)


if __name__ == "__main__":
    main()
