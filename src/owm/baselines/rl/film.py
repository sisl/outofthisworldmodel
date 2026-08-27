"""Fly a checkpoint on one rollout row, into a trajectory file.

The row is flown once, deterministically, from the reset its seed draws, at
the row's `rate_hz` with its `action_repeat` held by this loop rather than by
the `ActionRepeat` wrapper -- so every integration step is recorded, not only
the decision steps -- and handed back as the `Trajectory` owm-envs' render
and plot commands read.
"""

from __future__ import annotations

# Ahead of the owm_envs imports below, and not sorted in with them: importing
# it pins JAX to CPU, and XLA reads the platform when its backend first comes
# up, which owm_envs triggers as it is imported.
from owm.envs.factory import env_conf_dict, env_name_of, env_spec, make_env  # isort: skip

import jax
import jax.numpy as jnp
import numpy as np
from owm_envs.datasets.trajectory import Trajectory, start_fingerprint
from owm_envs.envs.common.config import BaseTaskConfig


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
        "method": "rl",
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
