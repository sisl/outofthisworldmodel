"""Seed an off-policy replay buffer with pre-flown trajectories from the hub.

Nothing has docked. Across 65 sweep trials, a 3M-step run and four 10M runs,
`docking/docked_count` never left zero, so `dock_success` was never once
collected and the policy has never seen the outcome it is being trained
toward. Demonstrations are the direct answer: put successful episodes in the
buffer before the first gradient step, and SAC's critic has something to
regress the goal against from the start.

The dataset (`sislaboratory/owm-*`) is written by owm-envs' own rollout
driver, so it carries what a transition needs -- a 27-wide
`observation_vector` matching this env's observation exactly, the `action`
taken, and `state_vector`, the env's 21-wide TRUE state.

REWARDS ARE RECOMPUTED, NOT READ. The published datasets were flown under
their own reward weights (the -1e6 collision term, since revised), so their
stored `reward` column does not describe the reward a run here trains under.
It does not have to: `docking_reward` is a pure function of (state, action,
events, cfg), and both inputs are stored, so the reward is recoverable under
whatever weights the caller is training with -- which is also what lets one
download serve both the baseline and approach reward variants. Recomputation
was checked against the dataset's own column under the dataset's own weights:
over 6,000 train rows it agrees to a mean absolute error of 4.8e-4, with the
only disagreements being the four collision steps, whose event term this
reproduces separately.

OBSERVATIONS ARE NORMALIZED ON THE WAY IN. Training wraps the env in
VecNormalize, so the policy reads normalized observations and the buffer must
hold them in the same units. The demo observations are used to seed
`obs_rms` before they are normalized with it, so the statistics describe the
demonstrations rather than starting from nothing. Training then keeps
updating those statistics from its own rollouts, so the demos drift slowly
out of calibration as it runs -- inherent to VecNormalize with a replay
buffer, and the reason `demo.fraction` is worth keeping below the buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from owm_envs.envs.common.config import BaseTaskConfig
from owm_envs.envs.common.events import EventChecker, Events
from owm_envs.envs.common.reward import docking_reward

from owm.envs.factory import env_spec

# The dataset's own splits; train is the one with the flown episodes.
DEFAULT_SPLIT = "train"

# One jitted vmap of docking_reward per task config. The config is static to
# the trace, so it cannot be an argument; caching on it keeps a run that reads
# several shards from recompiling for each.
_SHAPED_CACHE: dict[int, object] = {}


def _shaped_reward_fn(cfg: BaseTaskConfig):
    key = id(cfg)
    if key not in _SHAPED_CACHE:
        _SHAPED_CACHE[key] = jax.jit(jax.vmap(
            lambda s, a, e, t, prev: docking_reward(s, a, e, cfg, t, prev)
        ))
    return _SHAPED_CACHE[key]


_DOCKED_CACHE: dict[int, object] = {}


def _docked_fn(cfg: BaseTaskConfig):
    """Vmapped dock-gate test, for finding where an episode first succeeds."""
    key = id(cfg)
    if key not in _DOCKED_CACHE:
        checker = EventChecker(cfg)
        _DOCKED_CACHE[key] = jax.jit(jax.vmap(
            lambda v, t: checker.docked(v[0:3], v[3:6], v[6:10], v[10:13], t)
        ))
    return _DOCKED_CACHE[key]


@dataclass(frozen=True)
class DemoTransitions:
    """Flat transitions, already in the reward the caller trains under."""

    obs: np.ndarray          # (n, obs_dim) raw, unnormalized
    next_obs: np.ndarray     # (n, obs_dim) raw, unnormalized
    action: np.ndarray       # (n, act_dim)
    reward: np.ndarray       # (n,) recomputed under the caller's weights
    done: np.ndarray         # (n,) True on a terminating final step
    timeout: np.ndarray      # (n,) True on a truncating final step
    episode: np.ndarray      # (n,) source episode index
    policy_id: np.ndarray    # (n,) which generating policy flew it
    docked: np.ndarray       # (n,) True on a final step the dock gate passed
    collided: np.ndarray     # (n,) True on a final step the hull test fired

    def __len__(self) -> int:
        return int(self.obs.shape[0])


def _column(table, name: str, dtype=None) -> np.ndarray:
    values = table.column(name).to_pylist()
    return np.asarray(values, dtype=dtype) if dtype else np.asarray(values)


def _scalar_column(table, name: str, dtype) -> np.ndarray:
    """A column the writer stored nested, e.g. reward as [[x]]."""
    return np.array(
        [np.asarray(v, dtype=dtype).ravel()[0] for v in table.column(name).to_pylist()],
        dtype=dtype,
    )


def _data_files(repo_id: str, split: str, revision: str | None) -> list[str]:
    siblings = HfApi().repo_info(repo_id, repo_type="dataset", revision=revision).siblings
    return sorted(
        s.rfilename for s in siblings if s.rfilename.startswith(f"{split}/data/")
    )


def _classify_terminal(
    checker: EventChecker, prev_rel: np.ndarray, last_rel: np.ndarray, target: np.ndarray
) -> Events:
    """Which absorbing event, if any, the episode's last step raised.

    Delegated to the env's own EventChecker rather than reimplemented from the
    gate values. Two of its three tests do not fall out of the terminal state
    the obvious way: `escaped` measures range from the CHIEF ORIGIN while the
    reward is shaped toward a port ~24.6 m off it, and `collision` is a
    swept-segment test against 313 boxes over the whole step rather than a
    property of the endpoint. Hand-rolling those disagreed with the env by a
    factor of twenty on how many episodes docked.

    Read off the true state rather than taken from the dataset, which records
    `terminated` per episode and so says that something absorbing happened
    without saying which. The gates are the caller's, so an episode that
    docked under the dataset's gate but not under a tighter one is scored the
    way the run would score it.
    """
    return checker.events(
        jnp.asarray(prev_rel, dtype=jnp.float32),
        jnp.asarray(last_rel, dtype=jnp.float32),
        jnp.asarray(target, dtype=jnp.float32),
    )


def load_demo_transitions(
    repo_id: str,
    cfg: BaseTaskConfig,
    env_name: str,
    revision: str | None = None,
    split: str = DEFAULT_SPLIT,
    policies: tuple[int, ...] | None = None,
    successful_only: bool = False,
    max_transitions: int | None = None,
) -> DemoTransitions:
    """Read a dataset split and return transitions rewarded under `cfg`."""
    view = env_spec(env_name).view
    checker = EventChecker(cfg)
    # Whether a collision is absorbing belongs to the run being seeded, not to
    # the run that flew the dataset.
    collision_terminates = bool(getattr(cfg, "collision_terminates", True))
    files = _data_files(repo_id, split, revision)
    if not files:
        raise SystemExit(f"dataset {repo_id!r} has no {split}/data/ files to read")

    chunks: list[DemoTransitions] = []
    total = 0
    for name in files:
        table = pq.read_table(hf_hub_download(
            repo_id, name, repo_type="dataset", revision=revision))
        obs = _column(table, "observation_vector", np.float32)
        action = _column(table, "action", np.float32)
        state = _column(table, "state_vector", np.float64)
        episode = _column(table, "episode_index").ravel().astype(np.int64)
        policy = _scalar_column(table, "policy_id", np.int64)
        is_last = np.asarray(table.column("is_last").to_pylist(), dtype=bool)
        target = np.stack([
            np.asarray(v, dtype=np.float32).reshape(-1)[:7]
            for v in table.column("dock_target").to_pylist()
        ])

        rel = np.asarray(jax.vmap(view)(jnp.asarray(state, dtype=jnp.float32)))
        # Shaped reward for every row of the shard in one vmapped pass. Done
        # per shard rather than per episode so the jit compiles once for a
        # fixed length instead of once per distinct episode length -- the
        # difference between ~0.1 s and several minutes over a 500k split.
        rows_n = len(rel)
        quiet = Events(
            collision=jnp.zeros(rows_n, dtype=bool),
            docked=jnp.zeros(rows_n, dtype=bool),
            escaped=jnp.zeros(rows_n, dtype=bool),
        )
        # The previous row, for the optional progress term. Episodes are
        # contiguous in the shard, so a row's predecessor is the one before it
        # -- except the first of each episode, which is its own predecessor and
        # so scores zero progress, exactly as the env's first step does.
        first_of_episode = np.r_[True, episode[1:] != episode[:-1]]
        prev_index = np.where(first_of_episode, np.arange(rows_n), np.arange(rows_n) - 1)
        prev_rel_all = rel[prev_index]
        shaped_all = np.asarray(_shaped_reward_fn(cfg)(
            jnp.asarray(rel), jnp.asarray(action), quiet, jnp.asarray(target),
            jnp.asarray(prev_rel_all),
        ), dtype=np.float32)
        # Where the dock gate passes, step by step. The generating rollouts did
        # not stop at the gate -- 48 of 94 episodes reach inside 0.1 m and only
        # 2 END there -- so replayed whole, an episode's success would be a
        # moment in its middle that no transition is ever rewarded for. Cutting
        # each episode at its first passing step is what the env would have
        # produced, since dock success is absorbing.
        docked_all = np.asarray(_docked_fn(cfg)(
            jnp.asarray(rel), jnp.asarray(target)
        ), dtype=bool)

        for ep in np.unique(episode):
            rows = np.flatnonzero(episode == ep)
            pid = int(policy[rows[0]])
            if policies is not None and pid not in policies:
                continue
            hit = np.flatnonzero(docked_all[rows])
            if len(hit):
                rows = rows[: hit[0] + 1]
            ep_rel, ep_target = rel[rows], target[rows]
            prev_rel = ep_rel[-2] if len(rows) > 1 else ep_rel[-1]
            terminal = _classify_terminal(checker, prev_rel, ep_rel[-1], ep_target[-1])
            if successful_only and not bool(terminal.docked):
                continue

            reward = shaped_all[rows].copy()
            # The absorbing term lands on the step that raised it.
            reward[-1] = float(docking_reward(
                jnp.asarray(ep_rel[-1]), jnp.asarray(action[rows][-1]),
                terminal, cfg, jnp.asarray(ep_target[-1]),
                jnp.asarray(prev_rel_all[rows[-1]])))

            ep_obs = obs[rows]
            # Whether the dataset still holds the row after this episode's
            # last one. Cutting at the gate leaves the rest of the rollout on
            # disk, so a cut episode has a real successor state; an episode
            # that runs to the end of its rollout does not.
            last = int(rows[-1])
            has_successor = last + 1 < len(obs) and int(episode[last + 1]) == int(ep)
            next_obs = np.vstack([
                ep_obs[1:], obs[last + 1][None] if has_successor else ep_obs[-1:],
            ])
            done = np.zeros(len(rows), dtype=bool)
            timeout = np.zeros(len(rows), dtype=bool)
            # Read from the events, not from the dataset's terminated flag: a
            # cut episode ends because the gate passed, whatever the rollout it
            # was cut out of recorded, and -- more importantly -- whether a
            # collision is absorbing is a property of THIS run's config, not of
            # the run that flew the data. Seeding a soft-keep-out run with
            # collisions marked terminal would teach the critic that hitting
            # the hull ends the world when the env it is training against says
            # otherwise.
            absorbing = (
                bool(terminal.docked)
                or bool(terminal.escaped)
                or (bool(terminal.collision) and collision_terminates)
            )
            done[-1] = absorbing
            timeout[-1] = not absorbing
            was_truncated = not absorbing
            # SB3 zeroes a timeout's done and bootstraps the critic off
            # next_obs, so a timeout step needs a REAL successor state. Where
            # the episode ran out of dataset there is none, and repeating the
            # final observation would make the target a self-loop whose fixed
            # point is reward/(1-gamma) -- ~-1,000 at gamma 0.999, against a
            # shaped step worth ~-1. Drop the step rather than teach that.
            dropped = was_truncated and not has_successor
            if dropped:
                rows = rows[:-1]
                if not len(rows):
                    continue
                ep_obs, next_obs = ep_obs[:-1], next_obs[:-1]
                reward, done, timeout = reward[:-1], done[:-1], timeout[:-1]

            docked_row = np.zeros(len(rows), dtype=bool)
            collided_row = np.zeros(len(rows), dtype=bool)
            # Only when the terminal step is still here to carry them: the
            # step above drops it, and the row that becomes last is an
            # ordinary mid-episode one that ended nothing.
            if not dropped:
                docked_row[-1] = bool(terminal.docked)
                collided_row[-1] = bool(terminal.collision)

            chunks.append(DemoTransitions(
                obs=ep_obs, next_obs=next_obs, action=action[rows], reward=reward,
                done=done, timeout=timeout,
                episode=np.full(len(rows), int(ep), dtype=np.int64),
                policy_id=np.full(len(rows), pid, dtype=np.int64),
                docked=docked_row, collided=collided_row,
            ))
            total += len(rows)
            if max_transitions is not None and total >= max_transitions:
                break
        if max_transitions is not None and total >= max_transitions:
            break

    if not chunks:
        raise SystemExit(
            f"no episodes in {repo_id}/{split} matched policies={policies} "
            f"successful_only={successful_only}"
        )
    merged = DemoTransitions(
        obs=np.concatenate([c.obs for c in chunks]),
        next_obs=np.concatenate([c.next_obs for c in chunks]),
        action=np.concatenate([c.action for c in chunks]),
        reward=np.concatenate([c.reward for c in chunks]),
        done=np.concatenate([c.done for c in chunks]),
        timeout=np.concatenate([c.timeout for c in chunks]),
        episode=np.concatenate([c.episode for c in chunks]),
        policy_id=np.concatenate([c.policy_id for c in chunks]),
        docked=np.concatenate([c.docked for c in chunks]),
        collided=np.concatenate([c.collided for c in chunks]),
    )
    if max_transitions is not None and len(merged) > max_transitions:
        keep = slice(0, max_transitions)
        merged = DemoTransitions(*[getattr(merged, f)[keep] for f in
                                   ("obs", "next_obs", "action", "reward", "done",
                                    "timeout", "episode", "policy_id", "docked",
                                    "collided")])
    return merged


def aggregate_for_action_repeat(demos: DemoTransitions, repeat: int) -> DemoTransitions:
    """Collapse each episode's transitions into `repeat`-step holds.

    The dataset was flown one decision per env step. A run with
    rl.action_repeat > 1 collects transitions that span `repeat` steps and
    carry the summed reward of all of them, so seeding it with per-step rows
    would put two different time deltas -- and two different reward scales,
    off by `repeat` -- into the same buffer, and the critic would be regressing
    a mixture of two MDPs.

    THE HELD ACTION IS THE CHUNK'S MEAN, which is an approximation and the one
    place this is not exact. The demonstrator varied its action over the steps
    being collapsed, so no single held action reproduces the recorded
    `next_obs`. The mean is the right choice for the translational half: total
    impulse is sum(F_i * dt), so holding the mean force over the chunk delivers
    exactly the impulse the varying sequence did. The rotational half is only
    approximately so, since body rate enters the attitude update nonlinearly.
    Over 20 steps of 50 ms the error is small, and it is the honest way to
    reuse trajectories flown at a different cadence.

    A trailing partial chunk is kept rather than dropped: it carries the
    episode's terminal, which for the docked episodes is the entire reason the
    demonstrations are here.
    """
    if repeat <= 1:
        return demos

    starts, actions, rewards, ends = [], [], [], []
    for ep in np.unique(demos.episode):
        rows = np.flatnonzero(demos.episode == ep)
        for begin in range(0, len(rows), repeat):
            chunk = rows[begin : begin + repeat]
            starts.append(chunk[0])
            ends.append(chunk[-1])
            actions.append(demos.action[chunk].mean(axis=0))
            rewards.append(demos.reward[chunk].sum())
    starts, ends = np.asarray(starts), np.asarray(ends)
    return DemoTransitions(
        obs=demos.obs[starts],
        next_obs=demos.next_obs[ends],
        action=np.asarray(actions, dtype=demos.action.dtype),
        reward=np.asarray(rewards, dtype=demos.reward.dtype),
        done=demos.done[ends],
        timeout=demos.timeout[ends],
        episode=demos.episode[starts],
        policy_id=demos.policy_id[starts],
        docked=demos.docked[ends],
        collided=demos.collided[ends],
    )


def seed_replay_buffer(model, venv, demos: DemoTransitions) -> dict[str, float]:
    """Add `demos` to `model`'s replay buffer, normalized as training sees it.

    Returns a summary the caller can log. `venv` is the VecNormalize wrapper
    when there is one; its `obs_rms` is updated from the demonstrations first,
    so the demos and the run's own early rollouts share a calibration.
    """
    buffer = model.replay_buffer
    # SB3 stores a replay buffer as (buffer_size // n_envs, n_envs, ...) and
    # `add` writes one row across every env slot, so transitions go in
    # n_envs at a time and the capacity is the product.
    width = buffer.n_envs
    capacity = buffer.buffer_size * width
    if len(demos) > capacity:
        raise SystemExit(
            f"{len(demos)} demo transitions do not fit a buffer holding "
            f"{capacity}; raise rl.hyperparams.buffer_size or lower "
            f"rl.demo.max_transitions"
        )

    obs, next_obs = demos.obs, demos.next_obs
    if venv is not None and hasattr(venv, "obs_rms") and venv.obs_rms is not None:
        venv.obs_rms.update(obs)
        obs = venv.normalize_obs(obs).astype(np.float32)
        next_obs = venv.normalize_obs(next_obs).astype(np.float32)

    # Whole rows only; the remainder is dropped rather than padded, since a
    # padded slot would be a transition the env never produced.
    usable = (len(demos) // width) * width
    for start in range(0, usable, width):
        sl = slice(start, start + width)
        buffer.add(
            obs[sl],
            next_obs[sl],
            demos.action[sl],
            demos.reward[sl],
            demos.done[sl],
            [{"TimeLimit.truncated": bool(t)} for t in demos.timeout[sl]],
        )

    return {
        "demo/transitions": float(usable),
        "demo/transitions_read": float(len(demos)),
        "demo/episodes": float(len(np.unique(demos.episode))),
        "demo/terminal_docked": float(np.sum(demos.docked)),
        "demo/terminal_collided": float(np.sum(demos.collided)),
        "demo/terminal_docked_and_collided": float(np.sum(demos.docked & demos.collided)),
        "demo/mean_reward": float(np.mean(demos.reward)),
        "demo/buffer_fraction": float(usable / capacity),
    }
