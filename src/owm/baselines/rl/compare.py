"""Compare two eval-matrix result directories on collision-voided success.

    python -m owm.baselines.rl.compare runs/evals/a runs/evals/b
    python -m owm.baselines.rl.compare runs/evals/a runs/evals/b --tolerance 1.0

THE COMPARISON IS PAIRED. `eval_matrix` seeds each port from its place in
owm-envs' `PORTS` table, so run A's trial 7 on `pirs_nadir` and run B's trial 7
on `pirs_nadir` are the same seed, the same initial state and the same target:
one episode flown by two policies, not two samples of a population.

That is what makes 50 trials a port enough to say anything. Two independent
proportions of 50 carry a standard error near 0.07 each, so a 0.10 gap between
them is noise. The paired form discards every episode the two policies agreed
on and tests only the ones they disagreed on, which is where the information
about a difference actually lives.

ACROSS RATES. The intended comparison is a world-model policy at 20 Hz with
`action_repeat=1` against an RL baseline at 1 Hz with `action_repeat=1` -- two
runs whose `dt` and `max_steps` differ by twenty. Those are still the same
episodes: `reset` draws its dispersions, its port and its epoch offset from the
seed alone and never from the timing, so the same seed produces a bit-identical
start at either rate. Timing fields are therefore allowed to differ here, and
`--strict-rate` refuses them for the equal-cadence reading instead.

Allowed to differ is not assumed to be harmless, though. Every episode carries
a `start_fingerprint` -- a digest of its initial true state -- and this refuses
to report a difference unless those match episode for episode. What holds today
is a property of owm-envs, not of this file, so it is checked rather than
trusted.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import click
from omegaconf import OmegaConf

# Settings that decide WHICH episodes were flown. Two runs disagreeing on any
# of these were not scored on the same work and cannot be differenced.
EPISODE_KEYS = ("seed", "ports", "trials", "criteria", "tolerances_m")

# Settings that decide how those episodes were FLOWN. Differing here is the
# point of a cross-rate comparison, so it is reported rather than refused --
# unless --strict-rate asks for the equal-cadence reading.
CADENCE_KEYS = ("rate_hz", "action_repeat", "dt", "max_steps")


@dataclass
class Results:
    """One eval-matrix output directory, indexed for pairing."""

    path: Path
    meta: dict
    collided: dict[tuple[str, int], bool]
    fingerprint: dict[tuple[str, int], str]
    fired: dict[tuple[str, int, str, float], bool]

    @property
    def label(self) -> str:
        return self.path.name

    def succeeded(self, port: str, trial: int, criteria: str, tolerance: float) -> bool:
        """Met the definition, and never touched the hull on the way."""
        if self.collided[(port, trial)]:
            return False
        return self.fired.get((port, trial, criteria, tolerance), False)


def load(run_dir: Path) -> Results:
    meta = OmegaConf.to_container(OmegaConf.load(run_dir / "meta.yaml"), resolve=True)
    collided: dict[tuple[str, int], bool] = {}
    fingerprint: dict[tuple[str, int], str] = {}
    with (run_dir / "episodes.csv").open() as handle:
        for row in csv.DictReader(handle):
            key = (row["port"], int(row["trial"]))
            collided[key] = row["ever_collided"] == "True"
            fingerprint[key] = row.get("start_fingerprint", "")
    fired: dict[tuple[str, int, str, float], bool] = {}
    with (run_dir / "outcomes.csv").open() as handle:
        for row in csv.DictReader(handle):
            fired[
                (row["port"], int(row["trial"]), row["criteria"], float(row["tolerance_m"]))
            ] = row["fired"] == "True"
    return Results(run_dir, meta, collided, fingerprint, fired)


def require_pairable(a: Results, b: Results, strict_rate: bool) -> list[str]:
    """Refuse two runs whose difference could not be read, and report the rest.

    Returns the cadence keys that differ, which the caller prints: a cross-rate
    comparison is legitimate and should say out loud that it is one.
    """
    keys = EPISODE_KEYS + (CADENCE_KEYS if strict_rate else ())
    mismatched = [key for key in keys if a.meta.get(key) != b.meta.get(key)]
    if mismatched:
        detail = "\n".join(
            f"    {key}: {a.meta.get(key)!r}  vs  {b.meta.get(key)!r}" for key in mismatched
        )
        raise SystemExit(
            f"{a.path} and {b.path} cannot be differenced; these disagree:\n{detail}\n"
            "Re-run both with the same eval_matrix settings."
            + ("" if strict_rate else "\nTiming fields are allowed to differ; these are not.")
        )

    # The start fingerprints are the actual evidence that the two runs flew the
    # same episodes. Config equality only says they were ASKED to.
    missing = [run.path for run in (a, b) if not any(run.fingerprint.values())]
    if missing:
        raise SystemExit(
            f"{missing[0]} records no start_fingerprint, so its episodes cannot be "
            "matched against another run's. Re-run it on a build that records one."
        )
    differing = [
        key for key in a.fingerprint
        if key in b.fingerprint and a.fingerprint[key] != b.fingerprint[key]
    ]
    if differing:
        example = differing[0]
        raise SystemExit(
            f"{len(differing)} of {len(a.fingerprint)} episodes began from different "
            f"initial conditions in the two runs -- e.g. {example[0]} trial "
            f"{example[1]}. The runs are not paired, so a per-episode difference "
            "between them would be a difference between start states."
        )
    return [key for key in CADENCE_KEYS if a.meta.get(key) != b.meta.get(key)]


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value over the discordant pairs.

    Under "the two policies are the same", each episode they disagreed on is a
    fair coin, so the count falling one way is Binomial(discordant, 1/2). Exact
    rather than the chi-square approximation because the discordant counts here
    are routinely single digits.
    """
    total = only_a + only_b
    if total == 0:
        return 1.0
    smaller = min(only_a, only_b)
    tail = sum(math.comb(total, i) for i in range(smaller + 1)) / 2**total
    return min(1.0, 2 * tail)


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the order given.

    The matrix is a dozen cells and the per-port table eight more, so an
    uncorrected 0.05 would turn up a "significant" cell about once per
    comparison by construction. Holm rather than plain Bonferroni: it is
    uniformly more powerful and needs no more assumptions.
    """
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        scaled = (len(p_values) - rank) * p_values[index]
        running = max(running, min(1.0, scaled))
        adjusted[index] = running
    return adjusted


@dataclass
class Cell:
    """One (criteria, tolerance) comparison over a set of ports."""

    rate_a: float
    rate_b: float
    only_a: int
    only_b: int
    episodes: int
    p: float


def compare(a: Results, b: Results, criteria: str, tolerance: float, ports: list[str]) -> Cell:
    both = only_a = only_b = neither = 0
    for port in ports:
        for trial in range(int(a.meta["trials"])):
            hit_a = a.succeeded(port, trial, criteria, tolerance)
            hit_b = b.succeeded(port, trial, criteria, tolerance)
            both += hit_a and hit_b
            only_a += hit_a and not hit_b
            only_b += hit_b and not hit_a
            neither += not hit_a and not hit_b
    episodes = both + only_a + only_b + neither
    return Cell(
        rate_a=(both + only_a) / episodes,
        rate_b=(both + only_b) / episodes,
        only_a=only_a,
        only_b=only_b,
        episodes=episodes,
        p=mcnemar_exact(only_a, only_b),
    )


def _table(rows: list[tuple[str, Cell]], head: str, label_a: str, label_b: str) -> str:
    adjusted = holm([cell.p for _, cell in rows])
    lines = [
        f"{head:>26} {label_a[:16]:>17} {label_b[:16]:>17} "
        f"{'diff':>8} {'A-only':>7} {'B-only':>7} {'p':>7} {'p(Holm)':>8}",
    ]
    for (name, cell), p_adj in zip(rows, adjusted):
        lines.append(
            f"{name:>26} {cell.rate_a:>17.3f} {cell.rate_b:>17.3f} "
            f"{cell.rate_b - cell.rate_a:>+8.3f} {cell.only_a:>7} {cell.only_b:>7} "
            f"{cell.p:>7.3f} {p_adj:>8.3f}"
        )
    return "\n".join(lines)


@click.command()
@click.argument("run_a", type=click.Path(path_type=Path, exists=True, file_okay=False))
@click.argument("run_b", type=click.Path(path_type=Path, exists=True, file_okay=False))
@click.option("--criteria", default="position", show_default=True,
              help="Criteria for the per-port breakdown.")
@click.option("--tolerance", type=float, default=5.0, show_default=True,
              help="Position tolerance in metres for the per-port breakdown.")
@click.option("--strict-rate", is_flag=True,
              help="Refuse runs whose rate or action_repeat differ, instead of "
                   "reporting the difference.")
def main(
    run_a: Path, run_b: Path, criteria: str, tolerance: float, strict_rate: bool
) -> None:
    """Difference two eval-matrix result directories, episode by episode.

    RUN_A and RUN_B are eval_matrix output directories.
    """
    a, b = load(run_a), load(run_b)
    cadence = require_pairable(a, b, strict_rate)
    ports = list(a.meta["ports"])

    print(f"A = {a.label}\nB = {b.label}")
    print(f"paired on seed={a.meta['seed']}, {a.meta['trials']} trials x "
          f"{len(ports)} ports, {len(a.fingerprint)} episodes, "
          "start fingerprints verified identical")
    if cadence:
        detail = ", ".join(
            f"{key} {a.meta.get(key)} vs {b.meta.get(key)}" for key in cadence
        )
        print(f"flown at different cadences ({detail}); the episodes are the same, "
              "the control rate is not")
    print()

    print("collision-voided success rate, all ports")
    rows = [
        (f"{name} @ {tol:g} m", compare(a, b, name, float(tol), ports))
        for name in a.meta["criteria"]
        for tol in a.meta["tolerances_m"]
    ]
    print(_table(rows, "criteria", a.label, b.label))

    print(f"\nper port, {criteria} @ {tolerance:g} m")
    per_port = [(port, compare(a, b, criteria, tolerance, [port])) for port in ports]
    print(_table(per_port, "port", a.label, b.label))

    print("\np is a two-sided exact McNemar test over the episodes the two runs "
          "disagreed on.\np(Holm) is corrected across the cells of its own table.")


if __name__ == "__main__":
    main()
