"""Tag candidate seeds by the lighting their episode flies through.

    just scout poisk_zenith
    just scout harmony_fwd_pma2 --seeds 100000:100010

The epoch each seed draws sets where in the orbit the approach happens, and
so whether the station is in sunlight, in the Earth's shadow, or crosses the
terminator during the 360 s horizon. The chief's orbit does not depend on
what the chaser does, so a zero-action rollout traces it exactly, and the
conical-shadow illumination along that trace is what the tag summarises.
Beside the tag, the outcome each evaluation drop recorded for the same
`(port, seed)` is printed, so a row can be chosen for a known result -- a
world-model dock beside an RL escape -- rather than found by trial.
"""

from __future__ import annotations

# Ahead of the owm_envs imports below, and not sorted in with them: importing
# it pins JAX to CPU, and XLA reads the platform when its backend first comes
# up, which owm_envs triggers as it is imported.
from owm.envs.factory import (  # isort: skip
    env_config,
    env_name_of,
    env_spec,
    make_env,
)

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from owm_envs.envs.common.config import BaseTaskConfig
from owm_envs.envs.common.docking_ports import PORT_NAMES
from owm_envs.envs.common.epoch_state import epoch_from_prefix
from owm_envs.envs.common.layout import StateLayout
from owm_envs.envs.common.orbit import illumination

from owm.baselines.rl.eval_matrix import (
    at_rate,
    base_env_conf,
    for_port,
    goal_error_of,
    run_dir_for,
)
from owm.baselines.rl.evaluate import resolve_checkpoint
from owm.baselines.rl.film import (
    DEFAULT_CHECKPOINT,
    EVAL_DROPS,
    eval_outcome,
    resolve_evals,
)
from owm.baselines.rl.run_state import load_run_config

SUNLIT_MIN = 0.95
ECLIPSE_MAX = 0.05
EVAL_BASE_SEED = 100_000
EVAL_PORT_STRIDE = 10_000
EVAL_TRIALS = 50
SHADOW_THRESHOLD = 0.5


def lighting_tag(fraction: float) -> str:
    """The band `fraction` of the horizon spent lit falls in.

    The two extremes are held to a hair of full sun and full umbra so that a
    tag reads as a promise about the whole episode; everything between is a
    terminator crossing, whichever way it goes.
    """
    if fraction >= SUNLIT_MIN:
        return "sunlit"
    if fraction <= ECLIPSE_MAX:
        return "eclipse"
    return "transition"


def default_seeds(port: str, base_seed: int = EVAL_BASE_SEED,
                  count: int = EVAL_TRIALS) -> list[int]:
    """The seeds `eval_matrix` flies for `port`, in its own order.

    Each port owns a contiguous block a stride apart, so a seed scouted here
    is the very episode the evaluation drop recorded an outcome for.
    """
    start = base_seed + PORT_NAMES.index(port) * EVAL_PORT_STRIDE
    return list(range(start, start + count))


def _illumination_of(state: jnp.ndarray, layout: StateLayout) -> jnp.ndarray:
    """Lit fraction at one state; `layout.epoch` and `layout.chief` must be slices."""
    epoch = epoch_from_prefix(state[layout.epoch])
    chief = state[layout.chief]
    return illumination(epoch, chief[0:3])


def _require_orbit_slices(layout: StateLayout, env_name: str) -> None:
    """Refuse an env whose state carries no orbit to trace the lighting along.

    Both slices are optional in a `StateLayout`, and an env without them holds
    no epoch and no chief position, so there is no illumination to report.
    Named here rather than indexed with `None` inside the traced `vmap`, where
    the failure is a shape error about an env the user never chose.
    """
    missing = [name for name in ("epoch", "chief") if getattr(layout, name) is None]
    if missing:
        raise SystemExit(
            f"env '{env_name}' has no {' or '.join(missing)} in its state layout; "
            f"scout needs both to trace the chief's illumination")


def shadow_tag(profile: np.ndarray) -> str:
    """Which way `profile` crosses the terminator, if it crosses it at all.

    The manifest's `transition` tag says only that a crossing happens; this
    says which way, which is what picks the shot -- an approach flying into
    the dark reads very differently from one coming out of it.
    """
    first, last = float(profile[0]), float(profile[-1])
    if first > SHADOW_THRESHOLD and last < SHADOW_THRESHOLD:
        return "enters"
    if first < SHADOW_THRESHOLD and last > SHADOW_THRESHOLD:
        return "exits"
    return "-"


def illumination_profile(states: np.ndarray, layout: StateLayout) -> np.ndarray:
    """Lit fraction in [0, 1] at each row of `states`, from its own epoch."""
    fn = jax.jit(jax.vmap(lambda s: _illumination_of(s, layout)))
    return np.asarray(fn(jnp.asarray(states, jnp.float64)), dtype=np.float64)


def _trace_chief(cfg: BaseTaskConfig, seed: int) -> tuple[np.ndarray, float]:
    """Every state of `seed`'s zero-action episode, and its opening range.

    The chief propagates on its orbit whatever the chaser does, so the trace
    is the chief's true path over the horizon at whatever cadence `cfg` runs.
    """
    env = make_env(cfg, seed=seed)
    try:
        _, info = env.reset(seed=seed)
        error = goal_error_of(info)
        start_range = error["pos_m"] if error is not None else float("nan")
        states = [np.asarray(info["state"], dtype=np.float64)]
        done = False
        while not done:
            _, _, term, trunc, info = env.step(np.zeros(6, dtype=np.float32))
            states.append(np.asarray(info["state"], dtype=np.float64))
            done = bool(term or trunc)
    finally:
        env.close()
    return np.stack(states), start_range


def scout_seeds(cfg: BaseTaskConfig, seeds: Iterable[int], port: str,
                evals_dirs: dict[str, Path]) -> list[dict]:
    """One row per seed: its lighting, its opening range, its known outcomes."""
    spec = env_spec(env_name_of(cfg))
    layout = spec.layout
    _require_orbit_slices(layout, spec.name)
    rows = []
    for seed in seeds:
        states, start_range = _trace_chief(cfg, seed)
        profile = illumination_profile(states, layout)
        fraction = float(profile.mean())
        row = {
            "seed": int(seed),
            "lighting": lighting_tag(fraction),
            "illumination": fraction,
            "start_range_m": start_range,
            "shadow": shadow_tag(profile),
        }
        for label, directory in evals_dirs.items():
            row[f"{label}_outcome"] = eval_outcome(directory, port, seed)
        rows.append(row)
    return rows


def _parse_seeds(spec: str | None, port: str) -> list[int]:
    if spec is None:
        return default_seeds(port)
    if ":" in spec:
        lo, hi = spec.split(":", 1)
        return list(range(int(lo), int(hi)))
    return [int(s) for s in spec.split(",") if s]


def print_rows(rows: Sequence[dict], labels: Sequence[str]) -> None:
    header = (f"{'seed':>7} {'lighting':<11} {'illum':>6} {'shadow':>7} {'range_m':>8} "
              + " ".join(f"{label:>10}" for label in labels))
    print(header)
    for row in rows:
        outcomes = " ".join(f"{(row.get(f'{label}_outcome') or '-'):>10}" for label in labels)
        print(f"{row['seed']:>7} {row['lighting']:<11} {row['illumination']:6.2f} "
              f"{row['shadow']:>7} {row['start_range_m']:8.1f} {outcomes}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tag seeds by lighting and evaluation outcome.")
    parser.add_argument("--port", required=True, choices=PORT_NAMES)
    parser.add_argument("--seeds", default=None,
                        help="LO:HI range or comma list; default is the port's 50 eval_matrix seeds.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Run whose env_config.yaml defines the environment.")
    parser.add_argument("--rate-hz", type=float, default=1.0,
                        help="Trace cadence; 1 Hz is 360 steps and plenty for a lighting tag.")
    args = parser.parse_args(argv)
    ckpt = resolve_checkpoint(args.checkpoint)
    run_dir = run_dir_for(ckpt, None)
    base = at_rate(base_env_conf(run_dir, load_run_config(run_dir)), args.rate_hz)
    cfg = env_config(for_port(base, args.port))
    evals = resolve_evals(EVAL_DROPS)
    rows = scout_seeds(cfg, _parse_seeds(args.seeds, args.port), args.port, evals)
    print_rows(rows, list(evals))


if __name__ == "__main__":
    main()
