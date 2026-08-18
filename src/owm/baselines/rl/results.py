"""The evaluation result format, as a contract rather than a convention.

`eval_matrix` writes this and `compare` reads it, but neither owns it. A second
harness -- a world-model policy that loads differently, decides at its own rate
and manages its own horizon -- can produce these three files and be compared
against an RL baseline without sharing a line of rollout code.

WHAT A HARNESS MUST WRITE. Three files in one directory; `summary.csv` and
`report.md` are conveniences that nothing reads back.

    meta.yaml      the settings below, one document
    episodes.csv   one row per episode, EPISODE_FIELDS at minimum
    outcomes.csv   one row per (episode, criteria, tolerance), OUTCOME_FIELDS

Extra columns and extra meta keys are fine and ignored. What is NOT optional is
`start_fingerprint`: it is the only evidence that two directories describe the
same episodes, and a comparison refuses to report a difference without it.

WHAT MUST MATCH between two comparable directories, and what may not:

    EPISODE_KEYS   which episodes were flown -- must match exactly
    horizon_s      how long each episode had to succeed -- must match
    CADENCE_KEYS   how those episodes were flown -- may differ, and reporting
                   a difference across them is the point

The horizon is separate from the cadence deliberately. `dt` and `max_steps`
are each free to differ, because a 20 Hz policy and a 1 Hz one are compared at
different step counts on purpose -- but their PRODUCT is how much time the
policy had to reach the port, and a policy given half as long is not a policy
that did worse.
"""

from __future__ import annotations

import hashlib

import numpy as np

# Bumped when a field changes meaning. A reader that finds a version it does
# not know should say so rather than interpret the columns hopefully.
FORMAT_VERSION = 1

META = "meta.yaml"
EPISODES = "episodes.csv"
OUTCOMES = "outcomes.csv"

# The columns a comparison reads. A harness may write more.
EPISODE_FIELDS = ("port", "trial", "ever_collided", "start_fingerprint")
OUTCOME_FIELDS = ("port", "trial", "criteria", "tolerance_m", "fired")

# Which episodes were flown. Two directories disagreeing on any of these were
# not scored on the same work, and their difference cannot be read.
EPISODE_KEYS = ("seed", "ports", "trials", "criteria", "tolerances_m")

# How those episodes were flown. Differing here is legitimate and is what a
# cross-rate comparison exists to report.
CADENCE_KEYS = ("rate_hz", "action_repeat", "dt", "max_steps")


def start_fingerprint(state) -> str:
    """A digest of an episode's initial TRUE state.

    The cross-harness handshake. Two harnesses that seed the same environment
    the same way produce the same digest, which is what lets results they
    computed independently be paired episode for episode -- and what catches it
    when they do not.

    Over the raw float64 bytes, which is exact rather than tolerant: the same
    seed at 1 Hz and at 20 Hz produces a bit-identical start, because the
    environment draws its dispersions, its port and its epoch offset from the
    seed alone and never from the timing. A tolerance here would only hide a
    harness that had broken that property.
    """
    if state is None:
        return ""
    return hashlib.sha256(np.asarray(state, dtype=np.float64).tobytes()).hexdigest()[:16]


def horizon_s(meta: dict) -> float | None:
    """How long each episode had to succeed, in seconds.

    Recorded as `dt` and `max_steps` rather than directly, because those are
    what an environment is configured with; this is the quantity that actually
    has to agree between two comparable runs.
    """
    dt, steps = meta.get("dt"), meta.get("max_steps")
    if dt is None or steps is None:
        return None
    return float(dt) * int(steps)
