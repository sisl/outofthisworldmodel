"""Periodic validation episodes on the dock task, logged to wandb.

Runs known-seed deterministic episodes on a cadence and/or at fixed step
marks, and logs what a scalar return cannot say about a docking policy:

- composite (all six cameras tiled) and first-person videos of the approach,
- the 3D relative trajectory against its start point and the episode's
  target dock port, with and without body-axis triads showing attitude,
- the triads alone, and per-step reward, force, and torque traces.

The episodes are validation in the literal sense: seeds fixed at
construction, disjoint from the training seeds, so successive rounds (and
sweep trials sharing a seed) are the same episodes flown by different
policies. Everything is logged under ``val/*`` against ``val/global_step``.

The rollout env observes the way training does (vector observations) and is
held for the run; the renderer is not. Each round builds an `ISSRenderer`
and closes it again, because a live renderer holds a Vulkan device worth
~1.9 GB on the GPU for as long as the process lives -- and Vulkan does not
honour CUDA_VISIBLE_DEVICES, so that memory lands on a device the run was
never given. Rebuilding costs seconds per round, the cheaper side of that
trade on a shared machine.

This callback measures the dock task, so an env with `dock.enabled=false`
has nothing for it to measure: it announces that once and disables itself.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")  # figures are logged to wandb, never shown

import matplotlib.pyplot as plt
import numpy as np
import wandb
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection="3d")
from owm_envs.datasets.video import COMPOSITE_VIEWS, FPV_VIEW, render_adapter_for, tile_views
from stable_baselines3.common.callbacks import BaseCallback

from owm.envs.factory import env_spec, make_env, task_config_from_yaml

STEP_METRIC = "val/global_step"

# Okabe-Ito hues, CVD-validated as adjacent sets. The component triple keeps
# the aerospace x/y/z reading (warm/green/blue) on the triad, force, and
# torque plots; the episode set colors whole trajectories and never appears
# on the same figure as the component triple.
COMPONENT_COLORS = ("#D55E00", "#009E73", "#0072B2")
EPISODE_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9")

# How many body-axis triads to draw along a trajectory. Enough to read the
# attitude history, few enough that the arrows stay arrows rather than fur.
TRIAD_SAMPLES = 12


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """[w, x, y, z] body->world quaternion to its rotation matrix.

    Columns are the body axes expressed in the world frame -- owm-envs'
    q_bw convention (see owm_envs.core.quaternion). Numpy rather than the
    upstream jax helper because the plots run in the learner process on a
    handful of samples, not inside a jit.
    """
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class ValEpisode:
    """One finished validation episode's record, in plot-ready arrays."""

    def __init__(
        self,
        rel: np.ndarray,
        rewards: np.ndarray,
        actions: np.ndarray,
        goal_pose: np.ndarray,
        outcome: str,
        fpv_frames: list[np.ndarray],
        composite_frames: list[np.ndarray],
    ):
        # (T+1, 13) canonical relative view rows [pos, vel, q_bw, omega],
        # initial state included so the plotted trajectory starts where the
        # episode did.
        self.rel = rel
        self.rewards = rewards          # (T,)
        self.actions = actions          # (T, 6) physical [force N, torque N*m]
        self.goal_pose = goal_pose      # (7,) [position, quaternion]
        self.outcome = outcome
        self.fpv_frames = fpv_frames
        self.composite_frames = composite_frames

    @property
    def ep_return(self) -> float:
        return float(self.rewards.sum())


class ValEpisodeCallback(BaseCallback):
    """Run validation episodes on a schedule and log video + diagnostics.

    Scheduling composes three triggers so one class serves both callers:
    `every_steps` is training's cadence; `at_steps` are absolute step marks
    (a sweep trial's mid-point); `final=True` runs once more when training
    ends (a sweep trial's finished policy). At least one must be set.

    The env is taken from the run's own env_config.yaml record rather than
    re-resolved from the hydra config, for the same reason EvalReportCallback
    reads it: training writes that file as the record of what it actually
    trained on, and an `environments=from_dataset` ref can move between two
    resolutions.
    """

    def __init__(
        self,
        run_dir: Path,
        env_name: str,
        seed: int,
        episodes: int = 3,
        video_episodes: int = 1,
        every_steps: int | None = None,
        at_steps: tuple[int, ...] = (),
        final: bool = False,
        max_frames: int = 1200,
        action_repeat: int = 1,
    ):
        super().__init__()
        # All caught at registration, i.e. at launch: a bad budget or an
        # empty schedule only surfaces hours later otherwise.
        if every_steps is None and not at_steps and not final:
            raise ValueError(
                "ValEpisodeCallback needs a schedule: every_steps, at_steps, "
                "or final=True"
            )
        if every_steps is not None and every_steps < 1:
            raise ValueError(f"every_steps must be >= 1, got {every_steps}")
        if episodes < 1:
            raise ValueError(f"episodes must be >= 1, got {episodes}")
        # Must match training's: a val episode flown at a different decision
        # cadence measures a different policy than the one being trained.
        self._action_repeat = action_repeat
        if not 0 <= video_episodes <= episodes:
            raise ValueError(
                f"video_episodes must be in [0, episodes], got {video_episodes} "
                f"with episodes={episodes}"
            )
        if video_episodes and max_frames < 1:
            raise ValueError(f"max_frames must be >= 1, got {max_frames}")
        self._env_record = Path(run_dir) / "env_config.yaml"
        self._env_name = env_name
        self._seed = seed
        self._episodes = episodes
        self._video_episodes = video_episodes
        self._every = every_steps
        self._at = sorted(int(s) for s in at_steps)
        self._final = final
        self._max_frames = max_frames
        self._next_cadence: int | None = None
        self._env: gym.Env | None = None
        self._task_cfg = None
        self._disabled = False
        self._failures = 0

    def _on_training_start(self) -> None:
        # A resumed run boots at num_timesteps far past every_steps, which the
        # first _on_step would read as an overdue round; the cadence starts
        # from wherever training actually resumes, and at-marks the run is
        # already past are not overdue work either. A mark exactly at the
        # resumed step stays: whether the previous leg fired it before dying
        # cannot be known from here, and re-running a round costs minutes
        # while silently dropping one loses the mid-point video for good.
        if self._every is not None:
            self._next_cadence = self.num_timesteps + self._every
        self._at = [s for s in self._at if s >= self.num_timesteps]
        # train.py's wandb.init(sync_tensorboard=True) already owns the
        # implicit step axis, which silently drops any step= we pass to
        # wandb.log; val/* gets its own step metric instead.
        wandb.define_metric(STEP_METRIC)
        wandb.define_metric("val/*", step_metric=STEP_METRIC)

    def _on_step(self) -> bool:
        due = False
        if self._next_cadence is not None and self.num_timesteps >= self._next_cadence:
            # Schedule from the step actually reached, not just += every: on
            # a resumed run num_timesteps starts far past every_steps, and
            # += would fire on every subsequent step until it caught up.
            self._next_cadence = self.num_timesteps + self._every
            due = True
        if self._at and self.num_timesteps >= self._at[0]:
            while self._at and self.num_timesteps >= self._at[0]:
                self._at.pop(0)
            due = True
        if due:
            self._run_round()
        return True

    def _on_training_end(self) -> None:
        # Runs even when another callback ended training early, so a timed-out
        # sweep trial still shows the policy it ended with. The env has to be
        # closed even if that last round raises.
        try:
            if self._final:
                self._run_round()
        finally:
            if self._env is not None:
                self._env.close()
                self._env = None

    def _run_round(self) -> None:
        """One val round, contained: diagnostics must never kill the run.

        The renderer draws on a Vulkan device this run does not control
        (Vulkan ignores CUDA_VISIBLE_DEVICES), so a round can fail for
        reasons — another tenant's memory spike, a lost device — that say
        nothing about the training being diagnosed. One failure is retried
        at the next trigger; two in a row reads as persistent, and the
        callback stands down rather than failing every cadence to the end
        of the run.
        """
        if self._disabled:
            return
        try:
            self._round()
            self._failures = 0
        except Exception:
            traceback.print_exc()
            self._failures += 1
            if self._failures >= 2:
                print("[val] two rounds failed in a row; disabling val episodes")
                self._disabled = True
            else:
                print("[val] round failed; will retry at the next trigger")

    def _round(self) -> None:
        cfg = self._cfg()
        if not cfg.dock.enabled:
            print(
                "[val] env has dock.enabled=false; validation episodes "
                "measure the dock task, so none will run"
            )
            self._disabled = True
            return

        renderer = self._open_renderer(cfg) if self._video_episodes else None
        # One adapter serves every episode of the round; it is a pure function
        # of (env, cfg), both fixed for the run.
        adapter = (
            render_adapter_for(self._env_name, cfg) if renderer is not None else None
        )
        try:
            episodes = [
                self._rollout(cfg, index, renderer, adapter)
                for index in range(self._episodes)
            ]
        finally:
            # Closed before the figures are drawn, not after: the renderer's
            # Vulkan device is the expensive thing here, and matplotlib does
            # not need it.
            if renderer is not None:
                renderer.close()

        payload = {
            STEP_METRIC: self.num_timesteps,
            "val/mean_return": float(np.mean([ep.ep_return for ep in episodes])),
            "val/success_rate": sum(ep.outcome == "docked" for ep in episodes)
            / len(episodes),
        }
        fps = self._env.metadata.get("render_fps", 20)
        for name, frames in (
            ("val/video_fpv", [f for ep in episodes for f in ep.fpv_frames]),
            ("val/video_composite", [f for ep in episodes for f in ep.composite_frames]),
        ):
            if frames:
                payload[name] = wandb.Video(
                    np.stack(frames).transpose(0, 3, 1, 2), fps=fps, format="mp4"
                )

        figures = {
            "val/traj_3d": self._fig_trajectories(episodes, triads=False),
            "val/traj_3d_attitude": self._fig_trajectories(episodes[:1], triads=True),
            "val/attitude_triads": self._fig_triads_only(episodes[0]),
            "val/reward": self._fig_reward(episodes),
            "val/control": self._fig_control(cfg, episodes[0]),
        }
        try:
            payload.update({name: wandb.Image(fig) for name, fig in figures.items()})
            wandb.log(payload)
        finally:
            for fig in figures.values():
                plt.close(fig)
        outcomes = ", ".join(ep.outcome for ep in episodes)
        print(
            f"[val] step {self.num_timesteps}: mean_return "
            f"{payload['val/mean_return']:.1f}, outcomes: {outcomes}"
        )

    # ------------------------------------------------------------------ env

    def _cfg(self):
        if self._task_cfg is None:
            self._task_cfg = task_config_from_yaml(self._env_name, self._env_record)
        return self._task_cfg

    def _rollout(self, cfg, index: int, renderer, adapter):
        """One deterministic episode at seed+index, rendered if asked to.

        Rendering stops at max_frames; the rollout itself always runs to the
        episode's end, so the trajectory plots and the return cover the whole
        episode even when the video does not.
        """
        if self._env is None:
            # Held for the run: a vector-observation env costs nothing to
            # keep, unlike the renderer.
            self._env = make_env(cfg, seed=self._seed, action_repeat=self._action_repeat)
        env = self._env
        record_video = renderer is not None and index < self._video_episodes
        limits = np.array(
            [cfg.control.limit_force_n] * 3 + [cfg.control.limit_torque_nm] * 3,
            dtype=np.float32,
        )
        vecnorm = self.model.get_vec_normalize_env()

        states: list[np.ndarray] = []
        rewards: list[float] = []
        actions: list[np.ndarray] = []
        fpv: list[np.ndarray] = []
        composite: list[np.ndarray] = []

        obs, info = env.reset(seed=self._seed + index)
        goal_pose = np.asarray(info["goal_pose"], dtype=np.float64)
        states.append(np.asarray(info["state"], dtype=np.float64))
        if record_video and len(fpv) < self._max_frames:
            self._render_frame(renderer, adapter, states[-1], None, fpv, composite)
        outcome = "truncated"
        done = False
        while not done:
            norm = vecnorm.normalize_obs(obs) if vecnorm is not None else obs
            action, _ = self.model.predict(norm, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            physical = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0) * limits
            states.append(np.asarray(info["state"], dtype=np.float64))
            rewards.append(float(reward))
            actions.append(physical)
            if record_video and len(fpv) < self._max_frames:
                self._render_frame(
                    renderer, adapter, states[-1], physical, fpv, composite
                )
            done = term or trunc
        if info.get("success"):
            outcome = "docked"
        elif info.get("collision"):
            outcome = "collision"
        elif info.get("escaped"):
            outcome = "escaped"

        # The canonical 13D [pos, vel, q_bw, omega] relative view, whatever
        # width the env's raw state is -- the registry's view is the one
        # derivation the task layer itself reads.
        view = env_spec(self._env_name).view
        rel = np.asarray(
            jax.vmap(view)(jnp.asarray(np.stack(states), dtype=jnp.float32))
        )
        return ValEpisode(
            rel=rel,
            rewards=np.asarray(rewards),
            actions=np.stack(actions) if actions else np.zeros((0, 6)),
            goal_pose=goal_pose,
            outcome=outcome,
            fpv_frames=fpv,
            composite_frames=composite,
        )

    def _render_frame(self, renderer, adapter, state, action, fpv, composite) -> None:
        rendered = renderer.render_views(
            adapter(np.asarray(state, dtype=np.float32), action),
            views=COMPOSITE_VIEWS,
        )
        fpv.append(rendered[FPV_VIEW])
        composite.append(
            tile_views(rendered, renderer.cfg.image_height, renderer.cfg.image_width)
        )

    @staticmethod
    def _open_renderer(cfg):
        # Imported here, not at the top: pygfx and the GL stack are the
        # render extra's, and rounds with video_episodes=0 never need them.
        # The same sanctioned-lazy-import reasoning as owm.envs.factory's
        # resnet branch.
        from owm_envs.render.iss_scene import RenderConfig
        from owm_envs.render.renderer import ISSRenderer

        render_cfg = RenderConfig(**cfg.render) if cfg.render else RenderConfig()
        return ISSRenderer(render_cfg)

    # -------------------------------------------------------------- figures

    def _fig_trajectories(self, episodes: list[ValEpisode], triads: bool):
        """3D relative trajectories, start points and target dock ports.

        With `triads=True`, body-axis triads are drawn along the (single)
        trajectory: x/y/z body axes in the component colors, sampled evenly,
        so tumbling or a held attitude is visible at a glance.
        """
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(projection="3d")
        points = [np.zeros((1, 3))]  # the chief is the frame's origin
        for index, ep in enumerate(episodes):
            color = EPISODE_COLORS[index % len(EPISODE_COLORS)]
            pos = ep.rel[:, 0:3]
            points.extend([pos, ep.goal_pose[None, 0:3]])
            ax.plot(
                pos[:, 0], pos[:, 1], pos[:, 2],
                color="0.45" if triads else color,
                linewidth=1.4,
                label=f"ep{index} ({ep.outcome}, R={ep.ep_return:.0f})",
            )
            ax.scatter(*pos[0], color=color, marker="o", s=40, depthshade=False)
            ax.scatter(
                *ep.goal_pose[0:3], color=color, marker="*", s=140, depthshade=False,
                label=f"ep{index} dock port" if not triads else "dock port",
            )
        ax.scatter(0, 0, 0, color="0.2", marker="s", s=30, depthshade=False,
                   label="ISS origin")
        if triads:
            self._draw_triads(ax, episodes[0], self._triad_length(points))
        self._equalize_axes(ax, np.concatenate(points))
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title(
            "Val trajectory with body axes (relative frame)"
            if triads else "Val trajectories (relative frame)"
        )
        ax.legend(loc="upper left", fontsize=8)
        return fig

    def _fig_triads_only(self, ep: ValEpisode):
        """Attitude alone: the triads at their trajectory points, no path."""
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(projection="3d")
        points = [ep.rel[:, 0:3], ep.goal_pose[None, 0:3], np.zeros((1, 3))]
        ax.scatter(*ep.goal_pose[0:3], color="0.2", marker="*", s=140,
                   depthshade=False, label="dock port")
        self._draw_triads(ax, ep, self._triad_length(points))
        self._equalize_axes(ax, np.concatenate(points))
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title(f"Body-axis attitude along trajectory (ep0, {ep.outcome})")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper left", fontsize=8)
        return fig

    def _draw_triads(self, ax, ep: ValEpisode, length: float) -> None:
        samples = np.linspace(0, len(ep.rel) - 1, num=min(TRIAD_SAMPLES, len(ep.rel)))
        for label_axes, t in zip([True] + [False] * TRIAD_SAMPLES, samples.astype(int)):
            origin = ep.rel[t, 0:3]
            axes_world = quat_to_rotmat(ep.rel[t, 6:10])
            for k, (name, color) in enumerate(
                zip(("body x", "body y", "body z"), COMPONENT_COLORS)
            ):
                ax.quiver(
                    *origin, *axes_world[:, k],
                    length=length, normalize=True, color=color, linewidth=1.2,
                    label=name if label_axes else None,
                )

    @staticmethod
    def _triad_length(points: list[np.ndarray]) -> float:
        stacked = np.concatenate(points)
        extent = float((stacked.max(axis=0) - stacked.min(axis=0)).max())
        return max(extent, 1.0) * 0.06

    @staticmethod
    def _equalize_axes(ax, points: np.ndarray) -> None:
        # matplotlib 3D axes autoscale per axis, which shears a trajectory;
        # equal ranges around the data's center keep geometry readable.
        center = (points.max(axis=0) + points.min(axis=0)) / 2
        radius = float((points.max(axis=0) - points.min(axis=0)).max()) / 2
        radius = max(radius, 1.0) * 1.1
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))

    @staticmethod
    def _fig_reward(episodes: list[ValEpisode]):
        fig, ax = plt.subplots(figsize=(8, 4))
        for index, ep in enumerate(episodes):
            ax.plot(
                ep.rewards,
                color=EPISODE_COLORS[index % len(EPISODE_COLORS)],
                linewidth=1.4,
                label=f"ep{index} ({ep.outcome}, R={ep.ep_return:.0f})",
            )
        ax.set_xlabel("step")
        ax.set_ylabel("reward")
        ax.set_title("Per-step reward over val episodes")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        return fig

    @staticmethod
    def _fig_control(cfg, ep: ValEpisode):
        fig, (ax_f, ax_t) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        for k, (axis_name, color) in enumerate(zip("xyz", COMPONENT_COLORS)):
            ax_f.plot(ep.actions[:, k], color=color, linewidth=1.2, label=axis_name)
            ax_t.plot(ep.actions[:, 3 + k], color=color, linewidth=1.2, label=axis_name)
        for ax, limit, ylabel in (
            (ax_f, cfg.control.limit_force_n, "force [N]"),
            (ax_t, cfg.control.limit_torque_nm, "torque [N·m]"),
        ):
            ax.axhline(limit, color="0.6", linewidth=0.8, linestyle="--")
            ax.axhline(-limit, color="0.6", linewidth=0.8, linestyle="--")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.25)
        ax_t.set_xlabel("step")
        ax_f.set_title(f"Control effort (ep0, {ep.outcome})")
        fig.tight_layout()
        return fig
