"""What counts as a docked approach, as a table rather than a single gate.

`EventChecker.docked` tests a conjunction of four bounds -- position,
velocity, attitude and body rate -- and an environment carries exactly one
set of them. Reading a policy needs several: a run that reaches 4 m of its
port and one that never leaves the start shell both score zero against a
0.1 m gate, and only the second is a policy that failed to learn the task.

The four magnitudes `EventChecker.docked` tests are the four
`info["goal_error_true"]` reports (`goal.GOAL_ERROR_NORM_LABELS`), measured
from the same true state against the same per-episode port. So a definition
is a set of bounds on that dict and nothing more, which is what makes this
module pure: no env, no model, no config group.

WHY THIS SCORES OFFLINE. The gates reach an episode only through
termination -- observation, dynamics and a deterministic policy are identical
whatever bounds are configured. A looser definition is a superset of a
stricter one, so it is satisfied at or before the step the armed gate fires,
and up to that instant the trajectory is the one a loose-gate env would have
flown. One rollout with the env's own gate armed therefore scores every
looser definition exactly, rather than approximately.

LOOSER is the whole condition. The criteria below only ever DROP bounds, so
position is the one axis a definition can tighten along, and a tolerance
inside `dock.max_distance_m` is one the armed gate ends the approach before
reaching -- unobservable from such a rollout, and reported as a failure that
is an artifact of the gate. `eval_matrix.require_observable` refuses that
matrix rather than scoring it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

from owm_envs.envs.common.config import DockConfig
from owm_envs.envs.common.goal import GOAL_ERROR_NORM_LABELS

# Which of the four bounds a criteria applies. "full" is the environment's own
# definition; the other two relax it by dropping tests, not by widening them.
CRITERIA: tuple[str, ...] = ("position", "position_velocity", "full")

DEFAULT_TOLERANCES_M: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)


@dataclass(frozen=True)
class DockDefinition:
    """One success definition: a position bound plus whichever others apply.

    A bound of None is not tested at all, matching `DockConfig`'s own meaning
    for its two optional gates. `position` needs that for velocity too, which
    is why this carries its own bounds rather than a `DockConfig`:
    `DockConfig.max_velocity_m_s` is a plain float upstream, so there is no
    env config expressing "position and nothing else".
    """

    criteria: str
    tolerance_m: float
    max_velocity_m_s: float | None
    max_attitude_error_rad: float | None
    max_body_rate_rad_s: float | None

    @property
    def label(self) -> str:
        return f"{self.criteria}@{self.tolerance_m:g}m"

    def satisfied(self, error: Mapping[str, float]) -> bool:
        """Whether `error` -- one `info["goal_error_true"]` -- meets every bound."""
        if error["pos_m"] > self.tolerance_m:
            return False
        if self.max_velocity_m_s is not None and error["vel_mps"] > self.max_velocity_m_s:
            return False
        if (
            self.max_attitude_error_rad is not None
            and error["att_rad"] > self.max_attitude_error_rad
        ):
            return False
        if (
            self.max_body_rate_rad_s is not None
            and error["rate_radps"] > self.max_body_rate_rad_s
        ):
            return False
        return True


def definition(dock: DockConfig, criteria: str, tolerance_m: float) -> DockDefinition:
    """One definition, taking every bound but the position one from `dock`.

    So `full` at `dock.max_distance_m` is the environment's own gate, and the
    other cells differ from it only where they say they do.
    """
    if criteria not in CRITERIA:
        raise ValueError(f"unknown criteria {criteria!r}; expected one of {list(CRITERIA)}")
    velocity = dock.max_velocity_m_s if criteria in ("position_velocity", "full") else None
    attitude = rate = None
    if criteria == "full":
        if dock.max_attitude_error_deg is not None:
            attitude = math.radians(dock.max_attitude_error_deg)
        rate = dock.max_body_rate_rad_s
    return DockDefinition(
        criteria=criteria,
        tolerance_m=float(tolerance_m),
        max_velocity_m_s=velocity,
        max_attitude_error_rad=attitude,
        max_body_rate_rad_s=rate,
    )


def definitions(
    dock: DockConfig,
    criteria: Iterable[str] = CRITERIA,
    tolerances_m: Iterable[float] = DEFAULT_TOLERANCES_M,
) -> tuple[DockDefinition, ...]:
    """The full criteria x tolerance matrix, in the order given."""
    return tuple(
        definition(dock, name, tolerance)
        for name in criteria
        for tolerance in tolerances_m
    )


@dataclass(frozen=True)
class FirstFire:
    """The first step a definition was satisfied, and the errors there."""

    step: int
    errors: dict[str, float]


class DockScoreboard:
    """First satisfaction of each definition over one episode.

    First rather than last: a definition is a claim about whether the approach
    ever arrived, and the armed gate ends the episode the moment the strictest
    one does. Recording the step as well as the fact is what lets a loose
    definition report how long the approach took to reach it.
    """

    def __init__(self, definitions: Iterable[DockDefinition]):
        self.definitions: tuple[DockDefinition, ...] = tuple(definitions)
        self._fired: list[FirstFire | None] = [None] * len(self.definitions)

    def update(self, step: int, error: Mapping[str, float]) -> None:
        for index, item in enumerate(self.definitions):
            if self._fired[index] is None and item.satisfied(error):
                self._fired[index] = FirstFire(
                    step=step, errors={key: float(error[key]) for key in GOAL_ERROR_NORM_LABELS}
                )

    def rows(self) -> Iterator[tuple[DockDefinition, FirstFire | None]]:
        yield from zip(self.definitions, self._fired)

    def fired(self, item: DockDefinition) -> FirstFire | None:
        return self._fired[self.definitions.index(item)]
