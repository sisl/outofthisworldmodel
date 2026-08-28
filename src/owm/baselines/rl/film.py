"""Film the rollout manifest with an RL checkpoint.

    just film media/manifest.yaml media/rollouts --gpu-index 3

Every manifest row naming an `rl` method is flown into its own directory
under the output root -- the trajectory file, one clip per view and the
overhead plot -- and printed beside the outcome `eval_matrix` recorded for
the same `(port, seed)`, which is the comparison the deck makes.

A row is flown once, deterministically, from the reset its seed draws, at the
row's `rate_hz` with its `action_repeat` held by this loop rather than by the
`ActionRepeat` wrapper -- so every integration step is recorded, not only the
decision steps -- and handed back as the `Trajectory` owm-envs' render and
plot commands read.

Rows whose outputs are all present are left alone unless `--force`, so a run
interrupted partway through resumes rather than re-rendering what it has.
"""

from __future__ import annotations

# Ahead of the owm_envs imports below, and not sorted in with them: importing
# it pins JAX to CPU, and XLA reads the platform when its backend first comes
# up, which owm_envs triggers as it is imported.
from owm.envs.factory import (  # isort: skip
    env_conf_dict,
    env_config,
    env_name_of,
    env_spec,
    make_env,
)

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from owm_envs.datasets.trajectory import (
    ARRAY_FILE,
    META_FILE,
    SHORT_VIEW_NAMES,
    Trajectory,
    save_trajectory,
    start_fingerprint,
)
from owm_envs.datasets.trajectory_plot import plot_trajectory_png, plot_trajectory_video
from owm_envs.datasets.trajectory_render import render_trajectory_clips
from owm_envs.datasets.video import parse_view_names
from owm_envs.envs.common.config import BaseTaskConfig
from owm_envs.render.device import select_gpu

from owm.baselines.rl.eval_matrix import at_rate, base_env_conf, for_port, run_dir_for
from owm.baselines.rl.evaluate import load_normalizer, resolve_checkpoint
from owm.baselines.rl.manifest import RolloutRow, load_manifest
from owm.baselines.rl.run_state import load_run_config
from owm.baselines.rl.train import ALGOS

METHOD = "rl"
DEFAULT_CHECKPOINT = "runs/best/owm-iss-numerical-v1-coop-terminal-ppo-vector/final_model.zip"
DEFAULT_VIEWS = "fpv,dragon_iso"
PAPER_EVALS = Path.home() / "repos/amos_2026/figures/data/evals"
# One label may name several drops: the first that exists is read, so the same
# command works whether a drop sits under the run tree or only in the paper's
# figure data. The paper's own drop leads for `ppo` -- it is the evaluation
# the deck reports, so it is the one a film is compared against.
EVAL_DROPS: dict[str, tuple[Path, ...]] = {
    "ppo": (PAPER_EVALS / "ppo_coop", Path("runs/evals/paper/ppo_coop")),
    "owm": (Path("runs/evals/paper/owm_coop"), PAPER_EVALS / "owm_coop"),
}
# The drops record how an episode ended as flags beside a coarser `outcome`,
# and in this order.
EVAL_FLAG_COLUMNS = ("env_docked", "escaped", "ever_collided")
# What makes a filmed row on disk the row a request is asking for.
IDENTITY_KEYS = ("port", "seed", "rate_hz", "action_repeat", "produced_by")
PLOT_PNG = f"{METHOD}_traj.png"
PLOT_MP4 = f"{METHOD}_traj.mp4"
# The overhead plot is a slow read of the approach rather than a replay of it,
# so it is written at a fixed rate instead of the episode's own.
PLOT_FPS = 10


def classify_outcome(docked: bool, escaped: bool, ever_collided: bool) -> str:
    """How the episode ended, as the trajectory file's `outcome`.

    Ordered the way the environment ends an episode: a dock and an escape are
    terminal, a collision only when the keep-out zone is a hard constraint, so
    an approach that grazed the zone and went on to dock is a dock.
    """
    if docked:
        return "docked"
    if escaped:
        return "escaped"
    if ever_collided:
        return "collision"
    return "truncated"


def fly_episode(
    model,
    vecnorm,
    cfg: BaseTaskConfig,
    seed: int,
    action_repeat: int,
    rate_hz: float,
    port: str,
    lighting: str,
    produced_by: str,
) -> Trajectory:
    """One deterministic episode of `model` on `cfg`, recorded step by step.

    The decision is taken on the observation the policy would have seen and
    then held for `action_repeat` environment steps by this loop, so the rows
    come back at the environment's own integration step whatever cadence the
    policy decided at -- which is what the trajectory file is for.
    """
    env = make_env(cfg, seed=seed)
    spec = env_spec(env_name_of(cfg))
    limits = np.array([cfg.control.limit_force_n] * 3 + [cfg.control.limit_torque_nm] * 3,
                      dtype=np.float32)
    try:
        obs, info = env.reset(seed=seed)
        fingerprint = start_fingerprint(info["state"])
        goal = np.asarray(info["goal_pose"], dtype=np.float64)
        states = [np.asarray(info["state"], dtype=np.float64)]
        measured = [np.asarray(info["measured_state"], dtype=np.float64)]
        observations = [np.asarray(obs, dtype=np.float32)]
        actions, rewards, collisions = [], [], []
        docked = escaped = False
        done = False
        while not done:
            norm = vecnorm.normalize_obs(obs) if vecnorm is not None else obs
            action, _ = model.predict(norm, deterministic=True)
            action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            for _ in range(action_repeat):
                obs, reward, term, trunc, info = env.step(action)
                states.append(np.asarray(info["state"], dtype=np.float64))
                measured.append(np.asarray(info["measured_state"], dtype=np.float64))
                observations.append(np.asarray(obs, dtype=np.float32))
                actions.append(action)
                rewards.append(float(reward))
                collisions.append(bool(info.get("collision", False)))
                docked |= bool(info.get("success", False))
                escaped |= bool(info.get("escaped", False))
                done = bool(term or trunc)
                if done:
                    break
    finally:
        env.close()

    state = np.stack(states)
    rel_view = np.asarray(jax.vmap(spec.view)(jnp.asarray(state, jnp.float64)),
                          dtype=np.float64)
    action_norm = np.stack(actions)
    collision = np.asarray(collisions, dtype=bool)
    # An env whose layout carries no epoch is flown at a fixed time; the file
    # keeps the column so every reader can light a clip the same way.
    epoch_slice = spec.layout.epoch
    epoch = (state[:, epoch_slice] if epoch_slice is not None
             else np.zeros((state.shape[0], 2), dtype=np.float64))
    meta = {
        "method": METHOD,
        "port": port,
        "seed": int(seed),
        "env": spec.name,
        "env_config": {k: v for k, v in env_conf_dict(cfg).items() if k != "env_name"},
        "dt": float(cfg.dt),
        "rate_hz": float(rate_hz),
        "action_repeat": int(action_repeat),
        "steps": int(len(actions)),
        "outcome": classify_outcome(docked, escaped, bool(collision.any())),
        "ever_collided": bool(collision.any()),
        "min_range_m": float(np.min(np.linalg.norm(rel_view[:, 0:3] - goal[0:3], axis=1))),
        "start_fingerprint": fingerprint,
        "lighting": lighting,
        "produced_by": produced_by,
    }
    return Trajectory(
        epoch=epoch,
        state=state,
        rel_view=rel_view,
        measured_state=np.stack(measured),
        observation=np.stack(observations),
        action_norm=action_norm,
        action_phys=action_norm * limits,
        reward=np.asarray(rewards, dtype=np.float64),
        collision=collision,
        dock_target=goal,
        meta=meta,
    )


def _csv_bool(column: str, value: str) -> bool:
    """One flag column of a drop as a bool, refusing anything it cannot read.

    A silent `False` for an unrecognised spelling would turn a collision into
    a truncation in the printed comparison, which is exactly the distinction
    the column exists to make.
    """
    parsed = {"true": True, "false": False}.get(value.strip().casefold())
    if parsed is None:
        raise ValueError(
            f"eval drop column {column!r} holds {value!r}, expected True or False")
    return parsed


def eval_outcome(evals_dir: str | Path | None, port: str, seed: int) -> str | None:
    """The outcome `eval_matrix` recorded for `(port, seed)`, if a drop holds one.

    Read from the drop's flag columns wherever it carries them, so the printed
    column speaks film's vocabulary rather than the drop's: a drop never says
    "collision", it records `ever_collided` beside a `truncated` outcome, and
    the two columns are only comparable once both are classified the same way.

    Absent whenever the drop is: the comparison column is a convenience beside
    the film, and a missing one is worth a dash in the table rather than a
    stopped render.
    """
    if evals_dir is None:
        return None
    path = Path(evals_dir) / "episodes.csv"
    if not path.is_file():
        return None
    with path.open() as handle:
        for record in csv.DictReader(handle):
            if record["port"] == port and int(record["seed"]) == seed:
                flags = {column: record.get(column) for column in EVAL_FLAG_COLUMNS}
                # A row cut short leaves its trailing columns unset; the drop
                # still knows its own coarse outcome, so fall back to that.
                if all(flags.values()):
                    return classify_outcome(
                        *(_csv_bool(column, value) for column, value in flags.items()))
                return record["outcome"]
    return None


def resolve_evals(candidates: dict[str, tuple[Path, ...]]) -> dict[str, Path]:
    """The first drop that exists per label; labels with none are dropped."""
    found = {}
    for label, paths in candidates.items():
        existing = next((path for path in paths if path.exists()), None)
        if existing is not None:
            found[label] = existing
    return found


def row_outputs(directory: Path, views: str, render: bool) -> list[Path]:
    """Every file filming one row writes, so a finished row can be recognised.

    Named from the view selection rather than assumed, so a run asking for one
    view is skipped on that view's clip and not on one it never renders.
    """
    files = [directory / ARRAY_FILE, directory / META_FILE]
    if render:
        files += [directory / f"{METHOD}_{SHORT_VIEW_NAMES[name]}.mp4"
                  for name in parse_view_names(views)]
        files += [directory / PLOT_PNG, directory / PLOT_MP4]
    return files


def _requested_identity(row: RolloutRow, checkpoint: str) -> dict:
    """What a finished row's `meta.json` must say for it to be this row.

    The episode, the cadence it was flown at and the policy that flew it: two
    directories agreeing on their name and disagreeing on any of these are two
    different rollouts, and only one of them is the one being asked for.
    """
    return {
        "port": row.port,
        "seed": int(row.seed),
        "rate_hz": float(row.rl.rate_hz),
        "action_repeat": int(row.rl.action_repeat),
        "produced_by": str(Path(checkpoint).resolve()),
    }


def _stored_identity(stored: dict) -> dict:
    """The same fields out of a stored `meta.json`, spelled comparably.

    `produced_by` is resolved against the working directory because the same
    checkpoint reached as `runs/best/...` and as an absolute path is one
    policy, and a row filmed under either spelling answers a request made
    under the other.
    """
    identity = {key: stored.get(key) for key in IDENTITY_KEYS}
    if identity["produced_by"]:
        identity["produced_by"] = str(Path(identity["produced_by"]).resolve())
    return identity


def film_row(
    row: RolloutRow,
    model,
    vecnorm,
    base: dict,
    out_root: Path,
    checkpoint: str,
    views: str = DEFAULT_VIEWS,
    stride: int = 1,
    force: bool = False,
    render: bool = True,
    max_steps: int | None = None,
) -> dict:
    """Fly and draw one row into `out_root/<name>`, or report it already done.

    The row is re-timed to its own `rate_hz` before its port is narrowed, the
    order `run_eval_matrix` composes the environment in, so the episode the
    seed draws here is the one that evaluation flew.

    A finished row is only left alone when what is on disk is this row: its
    stored episode, cadence and checkpoint are checked against the request and
    a disagreement stops the run, because a directory holding some other
    rollout under this name is worse than an hour of re-rendering. `lighting`
    is a label rather than an input, so a changed one is written through.
    """
    directory = out_root / row.name
    if not force and all(path.exists() for path in row_outputs(directory, views, render)):
        stored = json.loads((directory / META_FILE).read_text())
        wanted = _requested_identity(row, checkpoint)
        on_disk = _stored_identity(stored)
        differing = [key for key, value in wanted.items() if on_disk[key] != value]
        if differing:
            detail = "; ".join(f"{key}: on disk {on_disk[key]!r}, requested {wanted[key]!r}"
                               for key in differing)
            raise SystemExit(
                f"row '{row.name}' in {directory} was flown with a different "
                f"{', '.join(differing)} ({detail}); pass --force to refilm it")
        relit = stored.get("lighting") != row.lighting
        if relit:
            stored["lighting"] = row.lighting
            (directory / META_FILE).write_text(
                json.dumps(stored, indent=2, sort_keys=True) + "\n")
        return {"name": row.name, "skipped": True, "relit": relit, **stored}
    timed = at_rate(base, row.rl.rate_hz)
    if max_steps is not None:
        timed = {**timed, "max_steps": max_steps}
    cfg = env_config(for_port(timed, row.port))
    traj = fly_episode(model, vecnorm, cfg, row.seed, row.rl.action_repeat, row.rl.rate_hz,
                       row.port, row.lighting, produced_by=checkpoint)
    save_trajectory(traj, directory)
    if render:
        render_trajectory_clips(traj, directory, views, stride=stride)
        plot_trajectory_png(traj, directory / PLOT_PNG)
        plot_trajectory_video(traj, directory / PLOT_MP4, fps=PLOT_FPS)
    return {"name": row.name, "skipped": False, "relit": False, **traj.meta}


def run_film(
    manifest: str | Path,
    out_root: str | Path,
    checkpoint: str = DEFAULT_CHECKPOINT,
    views: str = DEFAULT_VIEWS,
    stride: int = 1,
    force: bool = False,
    only: Sequence[str] = (),
    evals_dir: str | Path | None = None,
    render: bool = True,
    max_steps: int | None = None,
) -> list[dict]:
    """Film every `rl` row of `manifest`, one directory per row under `out_root`.

    One checkpoint is loaded for the whole manifest -- the rows differ in their
    episode, not in their policy -- and each row is printed as it finishes,
    beside the outcome evaluation recorded for the same episode.
    """
    rows = [row for row in load_manifest(manifest) if row.rl is not None]
    if only:
        unknown = sorted(set(only) - {row.name for row in rows})
        if unknown:
            raise SystemExit(
                f"--only names rows with no rl entry or not in the manifest: {unknown}")
        rows = [row for row in rows if row.name in only]
    # Resolved before it is recorded, so `produced_by` is one spelling of the
    # policy whatever spelling the caller used to name it.
    ckpt = resolve_checkpoint(str(checkpoint)).resolve()
    run_dir = run_dir_for(ckpt, None)
    run_cfg = load_run_config(run_dir)
    base = base_env_conf(run_dir, run_cfg)
    # The run's own record says which algorithm wrote the checkpoint; a run dir
    # predating the saved hydra config still has its env record, and `ppo` is
    # what those runs are.
    algo = str((run_cfg and OmegaConf.select(run_cfg, "rl.algo")) or "ppo")
    model = ALGOS[algo].load(ckpt, device="cpu")
    vecnorm = load_normalizer(ckpt, allow_unnormalized=False)
    out_root = Path(out_root)
    results = []
    for row in rows:
        result = film_row(row, model, vecnorm, base, out_root, str(ckpt), views, stride,
                          force, render, max_steps=max_steps)
        result["eval_outcome"] = eval_outcome(evals_dir, row.port, row.seed)
        results.append(result)
        flag = "skipped" if result["skipped"] else "filmed"
        note = "  lighting updated" if result["relit"] else ""
        print(f"[film] {row.name:>28} {row.port:>18} seed={row.seed} {flag}: "
              f"{result['outcome']:<10} collided={'Y' if result['ever_collided'] else 'N'} "
              f"steps={result['steps']:<5} "
              f"min_range={result['min_range_m']:6.2f} m  "
              f"eval={result['eval_outcome'] or '-'}{note}", flush=True)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Film manifest rows with an RL checkpoint.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="Root of the rollouts directory.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--views", default=DEFAULT_VIEWS)
    parser.add_argument("--stride", type=int, default=1,
                        help="Keep every STRIDEth frame of the clips.")
    parser.add_argument("--force", action="store_true", help="Refilm rows that already have clips.")
    parser.add_argument("--only", nargs="*", default=[],
                        help="Row names to film; default all rl rows.")
    parser.add_argument("--evals", type=Path, default=None,
                        help="eval_matrix drop whose episodes.csv outcome is printed beside "
                             "each row; the first EVAL_DROPS['ppo'] drop that exists otherwise.")
    parser.add_argument("--no-render", action="store_true", help="Write the trajectory file only.")
    parser.add_argument("--gpu-index", type=int, default=None,
                        help="Which GPU to render on; OWM_ENVS_GPU_INDEX otherwise.")
    args = parser.parse_args(argv)
    if args.stride < 1:
        parser.error(f"--stride must be >= 1, got {args.stride}")
    if not args.no_render:
        select_gpu(args.gpu_index)
    evals = args.evals if args.evals is not None else resolve_evals(EVAL_DROPS).get("ppo")
    run_film(args.manifest, args.out, args.checkpoint, args.views, args.stride, args.force,
             list(args.only), evals, not args.no_render)


if __name__ == "__main__":
    main()
