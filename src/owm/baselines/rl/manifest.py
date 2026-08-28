"""The presentation's rollout manifest: which episodes to film, by whom.

One row names an episode by `(port, seed)` and the cadence each method flies
it at. The same file is the hand-off to the world-model harness, so it is
validated here exactly as written rather than normalised on the way in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from owm_envs.envs.common.docking_ports import PORT_NAMES

LIGHTINGS = ("sunlit", "eclipse", "transition", "unknown")
DISTRIBUTIONS = ("train", "heldout")
METHODS = ("rl", "wm")


@dataclass(frozen=True)
class MethodSettings:
    rate_hz: float
    action_repeat: int


@dataclass(frozen=True)
class RolloutRow:
    name: str
    port: str
    seed: int
    lighting: str
    distribution: str
    rl: MethodSettings | None
    wm: MethodSettings | None


def _settings(raw: dict | None, row: str, method: str) -> MethodSettings | None:
    if raw is None:
        return None
    try:
        settings = MethodSettings(rate_hz=float(raw["rate_hz"]),
                                  action_repeat=int(raw["action_repeat"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"rollout '{row}' methods.{method} needs rate_hz and action_repeat: {exc}"
        ) from exc
    if settings.rate_hz <= 0 or settings.action_repeat < 1:
        raise ValueError(f"rollout '{row}' methods.{method}: rate_hz must be > 0 and "
                         f"action_repeat >= 1, got {settings}")
    return settings


def load_manifest(path: str | Path) -> list[RolloutRow]:
    payload = yaml.safe_load(Path(path).read_text()) or {}
    rows: list[RolloutRow] = []
    seen: set[str] = set()
    for raw in payload.get("rollouts") or []:
        name = str(raw.get("name", ""))
        if not name:
            raise ValueError("every rollout needs a name")
        if name in seen:
            raise ValueError(f"rollout name '{name}' appears more than once")
        seen.add(name)
        port = str(raw.get("port", ""))
        if port not in PORT_NAMES:
            raise ValueError(f"rollout '{name}' names unknown port '{port}'; one of {PORT_NAMES}")
        lighting = str(raw.get("lighting", "unknown"))
        if lighting not in LIGHTINGS:
            raise ValueError(f"rollout '{name}' lighting '{lighting}' is not one of {LIGHTINGS}")
        distribution = str(raw.get("distribution", ""))
        if distribution not in DISTRIBUTIONS:
            raise ValueError(f"rollout '{name}' distribution '{distribution}' is not one of {DISTRIBUTIONS}")
        methods = raw.get("methods") or {}
        unknown = sorted(set(methods) - set(METHODS))
        if unknown:
            raise ValueError(f"rollout '{name}' names unknown methods {unknown}")
        rows.append(RolloutRow(
            name=name,
            port=port,
            seed=int(raw["seed"]),
            lighting=lighting,
            distribution=distribution,
            rl=_settings(methods.get("rl"), name, "rl"),
            wm=_settings(methods.get("wm"), name, "wm"),
        ))
    return rows
