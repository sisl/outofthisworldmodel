"""Keep a finished run's best policy, not its last one.

    python -m owm.baselines.rl.promote runs/ppo_70M_near
    python -m owm.baselines.rl.promote runs/ppo_70M_near --criterion min_pos

PPO's entropy collapses partway through a long run and everything after that
point is worse than what came before, so `final_model.zip` can sit well past
the peak. Ranking reads the run's own wandb history rather than flying fresh
rollouts: the history already holds three independent readings of every point
in training, at a resolution no affordable re-evaluation would match.

    val_return     val/mean_return       deterministic val episodes    maximize
    train_return   rollout/ep_rew_mean   SB3's rolling training mean   maximize
    min_pos        docking/ep_min_pos_m  closest true approach         minimize

All three are printed for every candidate, because they can disagree and that
disagreement is the interesting part: return is dominated by shaping cost on a
run that never docks, while closest approach reads only whether the policy
closed. One of them decides.

Each candidate is scored over the window of history centred on its own step,
rather than at the single sample nearest it — a per-episode series is far too
noisy to pick a checkpoint off one point.

`docking/*` logs against its own `docking/episodes` step metric and carries no
`global_step`, so its rows reach the training-step axis by interpolating
wandb's own monotonic `_step` against the `(_step, global_step)` pairs the
`rollout/*` series carries.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from owm.baselines.rl.hub import upload_run
from owm.baselines.rl.run_state import (
    FINAL_MODEL,
    FINAL_VECNORM,
    checkpoints,
    load_final_steps,
    load_wandb_id,
    vecnormalize_for,
)

load_dotenv()

BEST_ROOT = Path("runs/best")
PROMOTION_RECORD = "promotion.yaml"
ENV_RECORD = "env_config.yaml"
RUN_CONFIG = "config.yaml"

# How many history rows to pull per series. The per-episode docking series is
# the long one -- a 70M-step run logs it hundreds of thousands of times -- and
# wandb downsamples to this many, which still leaves hundreds of samples inside
# a default window.
SAMPLES = 20_000


@dataclass(frozen=True)
class Criterion:
    """One ranking series: where to read it, and which way is better."""

    value_key: str
    step_key: str | None  # None means the series is placed by interpolation
    maximize: bool

    @property
    def better(self):
        return max if self.maximize else min


CRITERIA: dict[str, Criterion] = {
    "val_return": Criterion("val/mean_return", "val/global_step", maximize=True),
    "train_return": Criterion("rollout/ep_rew_mean", "global_step", maximize=True),
    "min_pos": Criterion("docking/ep_min_pos_m", None, maximize=False),
}

# The series carrying both wandb's row counter and the training step, and so
# the only one that can place a series logged against a different step metric.
_ANCHOR = CRITERIA["train_return"]


@dataclass(frozen=True)
class Series:
    """One criterion's history, on the training-step axis."""

    steps: np.ndarray
    values: np.ndarray

    def score(self, step: int, window: int) -> float:
        """The mean over `step` +- `window`, or the nearest sample if none land there."""
        inside = np.abs(self.steps - step) <= window
        if inside.any():
            return float(self.values[inside].mean())
        return float(self.values[np.abs(self.steps - step).argmin()])


def _rows(run, keys: list[str]) -> list[dict]:
    return run.history(keys=keys, samples=SAMPLES, pandas=False)


def fetch_series(run) -> dict[str, Series]:
    """Every criterion's history, each mapped onto the training-step axis."""
    anchor = _rows(run, [_ANCHOR.value_key, _ANCHOR.step_key])
    if not anchor:
        raise SystemExit(
            f"run {run.id} logged no {_ANCHOR.value_key}; without it there is no "
            "map from wandb's row counter to the training step, and no series "
            "can be placed against a checkpoint"
        )
    row_steps = np.array([row["_step"] for row in anchor], dtype=np.float64)
    train_steps = np.array([row[_ANCHOR.step_key] for row in anchor], dtype=np.float64)

    series: dict[str, Series] = {}
    for name, criterion in CRITERIA.items():
        keys = [criterion.value_key]
        if criterion.step_key is not None:
            keys.append(criterion.step_key)
        rows = _rows(run, keys)
        if not rows:
            continue
        values = np.array([row[criterion.value_key] for row in rows], dtype=np.float64)
        if criterion.step_key is not None:
            steps = np.array([row[criterion.step_key] for row in rows], dtype=np.float64)
        else:
            # Placed by wandb's own monotonic row counter: both series are
            # rows of one history, so a docking row between two rollout rows
            # was logged between the training steps they report.
            steps = np.interp(
                [row["_step"] for row in rows], row_steps, train_steps
            )
        series[name] = Series(steps=steps, values=values)
    return series


@dataclass(frozen=True)
class Candidate:
    """A policy on disk that could be promoted, and what it scores."""

    step: int
    model: Path
    stats: Path
    scores: dict[str, float]

    @property
    def is_final(self) -> bool:
        return self.model.name == FINAL_MODEL


def candidates(run_dir: Path) -> list[tuple[int, Path, Path]]:
    """Every (step, model, stats) triple the run can be promoted from.

    The finals are a candidate in their own right: a finished run's finals sit
    past its last periodic checkpoint, and on a run whose best policy is its
    last one, excluding them would promote a strictly earlier model.
    """
    found: list[tuple[int, Path, Path]] = []
    for step, model in checkpoints(run_dir):
        stats = vecnormalize_for(model)
        if stats is not None:
            found.append((step, model, stats))
    final_steps = load_final_steps(run_dir)
    final_model = run_dir / FINAL_MODEL
    final_stats = run_dir / FINAL_VECNORM
    if final_steps is not None and final_model.exists() and final_stats.exists():
        found.append((final_steps, final_model, final_stats))
    if not found:
        raise SystemExit(
            f"{run_dir} holds no checkpoint with its VecNormalize sibling, and no "
            "finals vouched for by final_steps.txt: there is nothing to promote"
        )
    return sorted(found, key=lambda item: item[0])


def default_window(steps: list[int]) -> int:
    """Half the run's own checkpoint spacing.

    Wide enough to average out a per-episode series, narrow enough that two
    neighbouring checkpoints are scored over disjoint history.
    """
    if len(steps) < 2:
        return max(1, steps[0] // 2)
    return int(np.median(np.diff(steps)) // 2)


def rank(run_dir: Path, run, window: int | None) -> tuple[list[Candidate], int]:
    found = candidates(run_dir)
    steps = [step for step, _, _ in found]
    span = default_window(steps) if window is None else window
    series = fetch_series(run)
    scored = [
        Candidate(
            step=step,
            model=model,
            stats=stats,
            scores={
                name: history.score(step, span) for name, history in series.items()
            },
        )
        for step, model, stats in found
    ]
    return scored, span


def best(scored: list[Candidate], criterion: str) -> Candidate:
    if criterion not in CRITERIA:
        raise click.UsageError(
            f"unknown criterion {criterion!r}; expected one of {list(CRITERIA)}"
        )
    usable = [item for item in scored if criterion in item.scores]
    if not usable:
        raise SystemExit(
            f"no candidate could be scored on {criterion!r}: the run logged no "
            f"{CRITERIA[criterion].value_key}"
        )
    return CRITERIA[criterion].better(usable, key=lambda item: item.scores[criterion])


def table(scored: list[Candidate], chosen: Candidate, criterion: str) -> str:
    names = [name for name in CRITERIA if any(name in item.scores for item in scored)]
    header = "  ".join(f"{name:>14}" for name in names)
    lines = [f"{'step':>12}  {header}"]
    for item in scored:
        cells = "  ".join(
            f"{item.scores[name]:>14.3f}" if name in item.scores else f"{'-':>14}"
            for name in names
        )
        mark = " <- best" if item is chosen else ""
        label = f"{item.step}{'*' if item.is_final else ''}"
        lines.append(f"{label:>12}  {cells}{mark}")
    lines.append(f"\nchosen by {criterion}; * marks the run's finals")
    return "\n".join(lines)


def promote(
    run_dir: Path,
    chosen: Candidate,
    criterion: str,
    window: int,
    wandb_url: str,
    out_root: Path,
) -> Path:
    """Copy the chosen policy into `out_root/<run name>/` under the finals' names.

    Under the finals' names deliberately: `vecnormalize_name_for` recognises
    only `final_model.zip` and `model_<N>_steps.zip`, so a promoted file named
    for its step and score would lose its statistics sibling and be refused by
    every evaluation entry point that loads it.
    """
    destination = out_root / run_dir.name
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(chosen.model, destination / FINAL_MODEL)
    shutil.copy2(chosen.stats, destination / FINAL_VECNORM)
    for name in (RUN_CONFIG, ENV_RECORD):
        source = run_dir / name
        if source.exists():
            shutil.copy2(source, destination / name)
    OmegaConf.save(
        OmegaConf.create({
            "run": run_dir.name,
            "source": str(chosen.model),
            "step": chosen.step,
            "from_finals": chosen.is_final,
            "criterion": criterion,
            "window": window,
            "scores": {name: float(value) for name, value in chosen.scores.items()},
            "wandb": wandb_url,
        }),
        destination / PROMOTION_RECORD,
    )
    return destination


def open_run(run_dir: Path):
    """The wandb run this directory recorded, through its own saved config."""
    run_id = load_wandb_id(run_dir)
    if run_id is None:
        raise SystemExit(f"{run_dir} has no wandb_run_id.txt, so its history cannot be read")
    config = run_dir / RUN_CONFIG
    if not config.exists():
        raise SystemExit(f"{run_dir} has no {RUN_CONFIG}, so its wandb project is unknown")
    saved: DictConfig = OmegaConf.load(config)
    entity = OmegaConf.select(saved, "logging.entity")
    project = OmegaConf.select(saved, "logging.project")
    if not entity or not project:
        raise SystemExit(f"{config} records no logging.entity/logging.project")
    return wandb.Api().run(f"{entity}/{project}/{run_id}")


@click.command()
@click.argument("run_dir", type=click.Path(path_type=Path, exists=True, file_okay=False))
@click.option(
    "--criterion", default="val_return", show_default=True,
    type=click.Choice(list(CRITERIA)), help="Which series decides.",
)
@click.option(
    "--window", type=int, default=None,
    help="Half-width in training steps of the history each candidate is scored "
         "over [default: half the run's checkpoint spacing].",
)
@click.option("--out", "out_root", type=click.Path(path_type=Path), default=BEST_ROOT,
              show_default=True, help="Where the promoted directory is written.")
@click.option("--upload/--no-upload", default=True, show_default=True,
              help="Publish the promoted directory to the HF Hub.")
@click.option("--repo-id", default=None, help="HF model repo [default: $OWM_HF_MODEL_REPO].")
def main(
    run_dir: Path, criterion: str, window: int | None, out_root: Path,
    upload: bool, repo_id: str | None,
) -> None:
    """Rank RUN_DIR's checkpoints on its wandb history and keep the best one."""
    run = open_run(run_dir)
    scored, span = rank(run_dir, run, window)
    chosen = best(scored, criterion)
    print(table(scored, chosen, criterion))
    print(f"window: +-{span} steps")

    destination = promote(run_dir, chosen, criterion, span, run.url, out_root)
    print(f"\npromoted {chosen.model} -> {destination}")

    if not upload:
        return
    if not repo_id:
        repo_id = os.environ.get("OWM_HF_MODEL_REPO")
    if not repo_id:
        raise click.UsageError("--repo-id not given and OWM_HF_MODEL_REPO is not set")
    print(upload_run(
        destination,
        repo_id,
        path_in_repo=f"rl/best/{destination.name}",
        extra_files=(ENV_RECORD, PROMOTION_RECORD),
    ))


if __name__ == "__main__":
    main()
