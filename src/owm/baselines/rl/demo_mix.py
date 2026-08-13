"""Two ways to keep demonstrations influencing training after they are loaded.

Seeding the buffer once (demo_buffer.py) puts expert episodes in front of the
critic, and then training dilutes them: SAC keeps adding its own transitions,
so the demo share of each sampled batch falls monotonically from the moment
learning starts. The first warm-started run showed exactly that shape --
best-episode closure reached 0.19 early and drifted back toward 1.0 as its own
data took over. These are the two standard answers, kept separate so their
contributions can be read apart.

PROTECTED FRACTION (DemoMixReplayBuffer). Demonstrations live in their own
array that training never writes over, and every sampled batch is drawn
`fraction` from it. The demo share stops decaying because it is set rather
than emergent.

BEHAVIOUR CLONING (behaviour_clone). The buffer only tells the CRITIC what
good states are worth; the actor still starts at random and has to rediscover
the actions. Regressing the actor on the demonstrators' actions first starts
it near the expert instead.
"""

from __future__ import annotations

import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from owm.baselines.rl.demo_buffer import DemoTransitions


class DemoMixReplayBuffer(ReplayBuffer):
    """A replay buffer that reserves part of every batch for demonstrations.

    The demonstrations are held apart from the circular buffer rather than
    inside it, so training's own transitions can never overwrite them and the
    mixing ratio is exactly `demo_fraction` for the whole run.
    """

    def __init__(self, *args, demo_fraction: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 <= demo_fraction < 1.0:
            raise ValueError(f"demo_fraction must be in [0, 1), got {demo_fraction}")
        self._demo_fraction = float(demo_fraction)
        self._demos: ReplayBufferSamples | None = None
        self._n_demos = 0

    def load_demos(self, demos: DemoTransitions, obs: np.ndarray, next_obs: np.ndarray) -> int:
        """Store `demos` apart from the circular buffer. `obs`/`next_obs` are
        the already-normalized observations, since that is what the policy
        reads and what the rest of the buffer holds."""
        self._n_demos = len(demos)
        to_t = lambda a, dtype: torch.as_tensor(np.asarray(a), dtype=dtype, device=self.device)
        self._demos = ReplayBufferSamples(
            observations=to_t(obs, torch.float32),
            actions=to_t(demos.action, torch.float32),
            next_observations=to_t(next_obs, torch.float32),
            # SB3's critic target multiplies by (1 - dones), and reads dones as
            # "terminated, not truncated" -- a timeout must still bootstrap.
            dones=to_t(demos.done & ~demos.timeout, torch.float32).reshape(-1, 1),
            rewards=to_t(demos.reward, torch.float32).reshape(-1, 1),
        )
        return self._n_demos

    def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
        if self._demos is None or self._demo_fraction == 0.0 or self._n_demos == 0:
            return super().sample(batch_size, env=env)
        n_demo = int(round(batch_size * self._demo_fraction))
        n_demo = min(n_demo, self._n_demos)
        own = super().sample(batch_size - n_demo, env=env)
        idx = torch.randint(0, self._n_demos, (n_demo,), device=self.device)
        return ReplayBufferSamples(*[
            torch.cat([getattr(own, f), getattr(self._demos, f)[idx]], dim=0)
            for f in ("observations", "actions", "next_observations", "dones", "rewards")
        ])


def behaviour_clone(
    model,
    obs: np.ndarray,
    actions: np.ndarray,
    steps: int,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
) -> dict[str, float]:
    """Regress the actor's mean action onto the demonstrators' actions.

    Trains the deterministic action rather than the sampled one: SAC's actor is
    a squashed Gaussian, and it is its mean that should sit on the expert. The
    entropy temperature and the critics are left alone -- this only moves the
    policy's starting point, and RL takes over unchanged afterwards.
    """
    if steps <= 0 or len(obs) == 0:
        return {}
    device = model.device
    obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device)
    act_t = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=device)
    # Actions are stored in the env's units; the actor emits squashed [-1, 1].
    low = torch.as_tensor(model.action_space.low, dtype=torch.float32, device=device)
    high = torch.as_tensor(model.action_space.high, dtype=torch.float32, device=device)
    act_t = torch.clamp(2.0 * (act_t - low) / (high - low) - 1.0, -0.999, 0.999)

    optimizer = torch.optim.Adam(model.actor.parameters(), lr=learning_rate)
    losses: list[float] = []
    model.actor.train()
    for step in range(steps):
        idx = torch.randint(0, len(obs_t), (min(batch_size, len(obs_t)),), device=device)
        mean_actions, _ = model.actor.get_action_dist_params(obs_t[idx])[:2]
        loss = torch.nn.functional.mse_loss(torch.tanh(mean_actions), act_t[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return {
        "demo/bc_steps": float(steps),
        "demo/bc_loss_first": float(np.mean(losses[: max(1, steps // 20)])),
        "demo/bc_loss_last": float(np.mean(losses[-max(1, steps // 20):])),
    }
