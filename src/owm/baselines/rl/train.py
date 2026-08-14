"""Train an SB3 PPO/SAC baseline on the ISS docking env.

    python -m owm.baselines.rl.train rl=ppo
    python -m owm.baselines.rl.train rl=sac run_dir=runs/sac_a seed=1
    python -m owm.baselines.rl.train run_dir=runs/sac_a resume=true
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import hydra
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

from owm.baselines.rl.hub import upload_run
from owm.baselines.rl.metrics import DockingMetricsCallback
from owm.baselines.rl.run_state import (
    CHECKPOINT_DIR,
    FINAL_MODEL,
    FINAL_REPLAY_BUFFER,
    FINAL_STEPS,
    FINAL_VECNORM,
    NAME_PREFIX,
    checkpoint_steps,
    clear_final_steps,
    latest_checkpoint,
    latest_complete_checkpoint,
    load_final_steps,
    load_wandb_id,
    missing_siblings,
    replay_buffer_for,
    save_final_steps,
    save_wandb_id,
    vecnormalize_for,
)
from owm.baselines.rl.val_episodes import ValEpisodeCallback
from owm.envs.factory import (
    DEFAULT_ENV_NAME,
    ENV_NAME_KEY,
    env_conf_dict,
    env_config,
    env_name_of,
    make_vec_env,
    preflight_render,
    require_renderable,
    task_config_from_yaml,
)

load_dotenv()

ALGOS = {"ppo": PPO, "sac": SAC}


def run_training(cfg: DictConfig, extra_callbacks: Sequence[BaseCallback] = ()) -> Path:
    run_dir = Path(cfg.run_dir)
    resume = bool(cfg.resume)
    # A sweep trial's run belongs to the wandb agent, which opened it before
    # this call and closes it after: starting a second run here would split the
    # trial's history in two and hide the objective from the sweep controller.
    # Read from the caller's config rather than the resumed run's, and so
    # deliberately not restored by the resume below: who owns the wandb run is a
    # property of this invocation, like extend_timesteps, not of the artifacts
    # on disk. A run first trained under an external run has no id saved, so it
    # can only ever be continued externally.
    external_wandb = bool(cfg.get("external_wandb", False))

    if resume:
        # The run's own saved config is authoritative: hyperparameters cannot
        # silently diverge from the ones the checkpoint was trained under. Only
        # extend_timesteps carries over from the command line — rl.total_timesteps
        # always composes to its group default, so honouring it here would grow a
        # bare resume's budget to 5M without anyone asking.
        saved = OmegaConf.load(run_dir / "config.yaml")
        saved.resume = True
        if cfg.get("extend_timesteps") is not None:
            saved.rl.total_timesteps = int(cfg.extend_timesteps)
        cfg = saved
    elif run_dir.exists() and any(run_dir.iterdir()):
        # Refuse to write a second run's config and id over an existing run's,
        # which would leave the dir describing one run and holding another's
        # checkpoints.
        raise SystemExit(
            f"run_dir {run_dir} already contains a run; pass resume=true to "
            "continue it or choose a new run_dir"
        )

    # Checked before anything is written or launched: a run that trains for
    # hours and then finds it has nowhere to publish has wasted the run.
    if cfg.hub.upload and not cfg.hub.repo_id:
        raise SystemExit(
            "hub.upload=true but hub.repo_id is empty (set OWM_HF_MODEL_REPO in "
            ".env or pass hub.repo_id=... / hub.upload=false)"
        )

    # .get, not attribute access: a run started before rl.obs existed resumes
    # from a saved config that has no such key, and it trained on vectors.
    obs_mode = str(cfg.rl.get("obs", "vector"))
    # .get for the same reason: a run started before val existed resumes from
    # a saved config with no such block, and it scheduled no val rounds.
    val_conf = cfg.get("val")
    val_enabled = bool(val_conf and val_conf.enabled)
    if obs_mode == "vector_resnet" and val_enabled:
        # Also checked at launch rather than at the first round, hours in.
        raise SystemExit(
            "rl.obs=vector_resnet with val.enabled=true: the val env is built "
            "with vector observations, so its rollout would hand the policy an "
            "observation of the wrong width to read (set val.enabled=false)"
        )
    if val_enabled and int(val_conf.video_episodes) > 0:
        # The renderer is built lazily at the first round, so an env that
        # ships no render adapter would otherwise surface only once the
        # cadence first came due. A dock-less env schedules no rounds at all
        # (see the registration below), so it owes no renderer either; a
        # from_dataset group carries no dock key at this point and is checked
        # as if docking, matching the default its config resolves to.
        dock_conf = cfg.environments.get("dock")
        if dock_conf is None or bool(dock_conf.get("enabled", True)):
            require_renderable(str(cfg.environments.get(ENV_NAME_KEY, DEFAULT_ENV_NAME)))

    # Only meaningful for the run this function owns; an external run's id is
    # the agent's business, and recording it here would invite a resume to
    # reattach to a run nothing is holding open.
    run_id = None
    if resume:
        if not external_wandb:
            run_id = load_wandb_id(run_dir)
            if run_id is None:
                raise SystemExit(f"resume=true but {run_dir} has no wandb_run_id.txt")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        # Save resolved, not raw: ${now:...} and ${oc.env:...} would re-resolve to
        # different values when a resume reloads this file.
        OmegaConf.save(
            OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)), run_dir / "config.yaml"
        )

        if not external_wandb:
            run_id = wandb.util.generate_id()
            save_wandb_id(run_dir, run_id)

    # No checkpoint at all means the run crashed before its first one: there is
    # no training state to lose, so it restarts from scratch under the same
    # wandb id and keeps its history attached. Checkpoints that exist but are
    # unreadable are a different case, handled below.
    needs_buffer = cfg.rl.algo == "sac"
    ckpt = latest_complete_checkpoint(run_dir, needs_buffer) if resume else None
    newest = latest_checkpoint(run_dir) if resume else None

    # A finished run's finals sit past its last periodic checkpoint — the budget
    # is met at a rollout boundary, not a checkpoint one — so extending one has
    # to restart from the finals or it silently discards that tail. Decided
    # before the checkpoints are judged: usable finals are enough to resume
    # from even when every checkpoint on disk is unreadable.
    finals_on_disk = (run_dir / FINAL_MODEL).exists() and (run_dir / FINAL_VECNORM).exists()
    final_steps = load_final_steps(run_dir) if resume and finals_on_disk else None
    from_final = final_steps is not None and (
        ckpt is None or final_steps >= checkpoint_steps(ckpt)
    )
    if from_final and needs_buffer and not (run_dir / FINAL_REPLAY_BUFFER).exists():
        raise SystemExit(
            f"{run_dir / FINAL_MODEL} has no {FINAL_REPLAY_BUFFER} beside it; "
            "resuming SAC from it would restart from an empty buffer"
        )
    if resume and finals_on_disk and final_steps is None:
        print(
            f"[resume] WARNING finals in {run_dir} have no {FINAL_STEPS} (a crashed "
            "final save, or a run dir predating the marker); they cannot be trusted "
            "to be one generation, so the last checkpoint is used instead"
        )

    if newest is not None and newest != ckpt:
        # CheckpointCallback writes the siblings after the zip, so a run killed
        # mid-save leaves a newest checkpoint that cannot be resumed from.
        # Substituting fresh statistics or an empty buffer would corrupt the
        # resumed run; an older complete checkpoint (or the finals) only costs
        # re-training.
        gaps = " and ".join(missing_siblings(newest, needs_buffer))
        if ckpt is None and not from_final:
            raise SystemExit(
                f"{newest} has no {gaps} and no older checkpoint in {run_dir} is "
                "complete; nothing here can be resumed from"
            )
        print(
            f"[resume] WARNING {newest.name} has no {gaps} (killed mid-save?); "
            f"resuming from {FINAL_MODEL if from_final else ckpt.name} and "
            "re-training the steps in between"
        )

    if from_final:
        source = run_dir / FINAL_MODEL
        source_vecnorm = run_dir / FINAL_VECNORM
        source_buffer = run_dir / FINAL_REPLAY_BUFFER
    elif ckpt is not None:
        source = ckpt
        source_vecnorm = vecnormalize_for(ckpt)
        source_buffer = replay_buffer_for(ckpt)
    else:
        source = None

    if external_wandb:
        # sync_tensorboard is a wandb.init argument, so the caller's init is
        # what decides whether SB3's TB scalars reach wandb at all.
        if wandb.run is None:
            raise SystemExit(
                "external_wandb=true but no wandb run is active; the caller must "
                "wandb.init(sync_tensorboard=True) before run_training"
            )
    else:
        wandb.init(
            id=run_id,
            resume="must" if resume else None,
            entity=cfg.logging.entity,
            project=cfg.logging.project,
            mode=cfg.logging.mode,
            name=run_dir.name,
            dir=str(run_dir),
            # .get, not attribute access: a run started before logging.tags
            # existed resumes from a saved config with no such key.
            tags=list(cfg.logging.get("tags") or ()),
            config=OmegaConf.to_container(cfg, resolve=True),
            sync_tensorboard=True,  # SB3 writes losses to TB; wandb mirrors them
        )

    # environments=from_dataset names a repo, not an env, so the hydra config
    # alone does not say what was trained on. Resolve it exactly once — on the
    # first launch — and record it the way published datasets ship theirs. A
    # resume reads that record back instead of re-resolving: an unpinned
    # dataset ref can move, which would leave the run's legs training under
    # different dynamics than the file claims.
    #
    # The record holds the task config alone, the shape a published dataset
    # ships and nothing more, so which env of the suite it belongs to is read
    # back off the run's own saved hydra config beside it -- absent on a run
    # started before the suite, which is exactly a run on `iss`.
    env_record = run_dir / "env_config.yaml"
    if resume and env_record.exists():
        env_name = str(cfg.environments.get(ENV_NAME_KEY, DEFAULT_ENV_NAME))
        task_cfg = task_config_from_yaml(env_name, env_record)
    else:
        task_cfg = env_config(cfg.environments)
        task_cfg.to_yaml(env_record)
    # Workers and the video env get the concrete config, never the repo name:
    # each would otherwise re-download it, and could disagree about the result.
    env_conf = env_conf_dict(task_cfg)

    resnet = None
    if obs_mode == "vector_resnet":
        # One frame before any worker is spawned: a GPU that cannot serve a
        # render should say so here, not from inside a worker an hour in.
        preflight_render(task_cfg)
        # Imported here, not at the top: owm.envs.resnet_obs pulls in
        # torchvision, and a vector run has no use for it. See the note in
        # owm/envs/factory.py.
        from owm.envs.resnet_obs import extractor_kwargs

        resnet = extractor_kwargs(cfg.rl)

    # .get, not attribute access: a run started before rl.action_repeat existed
    # resumes from a saved config with no such key, and 1 is what it flew.
    action_repeat = int(cfg.rl.get("action_repeat", 1) or 1)
    venv = make_vec_env(
        env_conf,
        cfg.rl.n_envs,
        cfg.seed,
        vec=cfg.rl.vec,
        obs_mode=obs_mode,
        resnet=resnet,
        action_repeat=action_repeat,
    )
    if source is not None:
        venv = VecNormalize.load(str(source_vecnorm), venv)
    else:
        # Position obs are O(100 m) while rates are O(1e-3); observation
        # normalization is load-bearing. Under vector_resnet the embedding
        # channels are normalized by the same running statistics, which is
        # what a frozen feature wants: their scale is whatever ImageNet
        # happened to give them, and no gradient reaches back to fix it.
        #
        # Reward normalization is rl.norm_reward's call and defaults off:
        # the shaped reward already costs ~1 per step at the envelope by
        # construction, and normalizing it lets one collision's -1e6 into
        # the running return variance, where it crushes every shaped reward
        # that follows. .get, not attribute access, so an rl group that does
        # not spell the key means "off"; a resumed run never reaches this
        # branch at all -- VecNormalize.load above restores the normalizer
        # it actually trained with, flag included.
        venv = VecNormalize(
            venv,
            norm_obs=True,
            norm_reward=bool(cfg.rl.get("norm_reward", False)),
            clip_obs=10.0,
            gamma=cfg.rl.hyperparams.gamma,
        )

    # A protected demo fraction needs its own buffer class, chosen before the
    # model is built; the demonstrations are loaded into it further down.
    demo_conf = cfg.rl.get("demo")
    demo_fraction = float(demo_conf.get("protected_fraction", 0.0)) if demo_conf else 0.0
    # Checked rather than clamped: the guard below is `> 0`, so a negative
    # value would train a plain buffer while the config, the sweep and the run
    # summary all say the demos are protected.
    if not 0.0 <= demo_fraction < 1.0:
        raise SystemExit(
            f"rl.demo.protected_fraction is {demo_fraction}; it is a share of "
            "each batch and must be in [0, 1)"
        )
    extra_algo_kwargs: dict = {}
    if demo_fraction > 0.0 and demo_conf and demo_conf.get("repo_id"):
        from owm.baselines.rl.demo_mix import DemoMixReplayBuffer

        extra_algo_kwargs["replay_buffer_class"] = DemoMixReplayBuffer
        extra_algo_kwargs["replay_buffer_kwargs"] = {"demo_fraction": demo_fraction}

    algo_cls = ALGOS[cfg.rl.algo]
    if source is not None:
        model = algo_cls.load(
            source, env=venv, device=cfg.rl.device, tensorboard_log=str(run_dir / "tb")
        )
        if needs_buffer:
            model.load_replay_buffer(source_buffer)
    else:
        model = algo_cls(
            "MlpPolicy",
            venv,
            seed=cfg.seed,
            device=cfg.rl.device,
            tensorboard_log=str(run_dir / "tb"),
            **extra_algo_kwargs,
            **OmegaConf.to_container(cfg.rl.hyperparams, resolve=True),
        )

    callbacks = [
        DockingMetricsCallback(),
        CheckpointCallback(
            # SB3 counts save_freq in per-env steps; divide to get total steps.
            # action_repeat divides too: every "steps" knob in this config is in
            # env steps, so a cadence does not silently stretch by k.
            save_freq=max(
                cfg.rl.checkpoint.save_freq // (cfg.rl.n_envs * action_repeat), 1
            ),
            save_path=str(run_dir / CHECKPOINT_DIR),
            name_prefix=NAME_PREFIX,
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
    ]
    if val_enabled:
        if task_cfg.dock.enabled:
            callbacks.append(ValEpisodeCallback(
                run_dir=run_dir,
                env_name=env_name_of(task_cfg),
                seed=cfg.seed + 10_000,  # never the training seeds
                episodes=int(val_conf.episodes),
                video_episodes=int(val_conf.video_episodes),
                # In env steps like every other cadence here; the callback
                # compares against SB3's decision counter.
                every_steps=max(int(val_conf.every_steps) // action_repeat, 1),
                max_frames=int(val_conf.max_frames),
                action_repeat=action_repeat,
            ))
        else:
            # Not an error: val.enabled defaults on, and a deliberately
            # dock-less env is a legitimate thing to train on -- it just has
            # no dock trajectory for these episodes to measure.
            print(
                "[val] env has dock.enabled=false; validation episodes "
                "measure the dock task, so none are scheduled"
            )
    callbacks.extend(extra_callbacks)

    # Demonstrations, if this run asked for them. Only on a fresh launch: a
    # resume already carries the buffer its earlier legs filled, and adding
    # the same episodes again would count them twice.
    if demo_conf and demo_conf.get("repo_id") and source is None:
        if not hasattr(model, "replay_buffer"):
            raise SystemExit(
                f"rl.demo.repo_id is set but rl={cfg.rl.algo} is on-policy and has "
                "no replay buffer to seed; demonstrations need an off-policy "
                "algorithm (rl=sac)"
            )
        # Imported here, not at the top: reading the hub's parquet shards pulls
        # in pyarrow and downloads a dataset, and a run without rl.demo set
        # should do neither.
        from owm.baselines.rl.demo_buffer import (
            aggregate_for_action_repeat,
            load_demo_transitions,
            seed_replay_buffer,
        )

        demos = load_demo_transitions(
            repo_id=str(demo_conf["repo_id"]),
            cfg=task_cfg,
            env_name=str(cfg.environments.get(ENV_NAME_KEY, DEFAULT_ENV_NAME)),
            revision=demo_conf.get("revision"),
            split=str(demo_conf.get("split", "train")),
            policies=(tuple(demo_conf["policies"]) if demo_conf.get("policies") else None),
            successful_only=bool(demo_conf.get("successful_only", False)),
            max_transitions=demo_conf.get("max_transitions"),
        )
        # The dataset was flown one decision per env step; this run may not be.
        # Seeding per-step rows into a buffer of k-step holds would mix two
        # time deltas and two reward scales in the same critic target.
        # Two factors, both from the same cause: the dataset was flown one
        # decision per 50 ms env step, and this run's decisions may span more
        # wall-clock than that -- because its env integrates coarsely
        # (task_cfg.dt), because it holds actions (action_repeat), or both.
        # Seeding without collapsing would mix time deltas and reward scales.
        source_dt = float(demo_conf.get("source_dt", 0.05) or 0.05)
        demo_stride = action_repeat * max(int(round(float(task_cfg.dt) / source_dt)), 1)
        demos = aggregate_for_action_repeat(demos, demo_stride)
        summary = seed_replay_buffer(model, venv, demos)
        summary["demo/action_repeat"] = float(action_repeat)
        summary["demo/stride"] = float(demo_stride)
        # The same observations the buffer holds: normalized once, reused by
        # the protected store and by behaviour cloning so all three agree.
        demo_obs, demo_next = demos.obs, demos.next_obs
        if venv is not None and getattr(venv, "obs_rms", None) is not None:
            demo_obs = venv.normalize_obs(demos.obs).astype("float32")
            demo_next = venv.normalize_obs(demos.next_obs).astype("float32")
        if demo_fraction > 0.0:
            n = model.replay_buffer.load_demos(demos, demo_obs, demo_next)
            summary["demo/protected_fraction"] = demo_fraction
            summary["demo/protected_transitions"] = float(n)
        bc_steps = int(demo_conf.get("bc_steps", 0) or 0)
        if bc_steps < 0:
            raise SystemExit(f"rl.demo.bc_steps is {bc_steps}; it is a step count")
        if bc_steps > 0:
            from owm.baselines.rl.demo_mix import behaviour_clone

            summary.update(behaviour_clone(model, demo_obs, demos.action, bc_steps))
        print(f"[demo] seeded replay buffer: {summary}")
        wandb.log(summary)

    # rl.total_timesteps is in ENVIRONMENT steps, and SB3 counts DECISIONS --
    # the two differ by action_repeat. Budgeting in env steps is what keeps a
    # run comparable across repeat settings: they are the same six minutes of
    # flight and the same simulator work whatever k is, where a decision budget
    # would silently multiply the wall clock by k.
    #
    # SB3 also adds the restored counter to whatever it is given when
    # reset_num_timesteps=False, so a resumed leg asks only for what is
    # outstanding -- in decisions, the units its counter is in.
    remaining = int(cfg.rl.total_timesteps) // action_repeat
    if source is not None:
        remaining = max(remaining - model.num_timesteps, 0)

    if remaining > 0:
        model.learn(
            total_timesteps=remaining,
            callback=callbacks,
            reset_num_timesteps=source is None,
        )

    # Nothing was trained, so re-saving would at best rewrite the finals with
    # themselves and at worst — when this model came from a checkpoint that
    # trails them — roll them back to that checkpoint's boundary.
    # Absent finals still have to be written — a crash between the last
    # checkpoint and the final save leaves the budget met but nothing final.
    # venv/wandb cleanup must run even if artifact logging or the hub upload
    # raises (e.g. a network error), or the run's env leaks and its wandb
    # history is never flushed; the exception still propagates after cleanup.
    try:
        finals_exist = (run_dir / FINAL_MODEL).exists() and (run_dir / FINAL_VECNORM).exists()
        if remaining == 0 and finals_exist:
            print(
                f"budget already met; final artifacts in {run_dir} left untouched, "
                "wandb artifact log and hub upload skipped"
            )
        else:
            # Withdrawn first: from here until the new marker is written the
            # finals are mid-replacement, and the old count would vouch for a
            # set that is half this leg's and half the previous one's.
            clear_final_steps(run_dir)
            model.save(run_dir / FINAL_MODEL)
            venv.save(str(run_dir / FINAL_VECNORM))
            if needs_buffer:
                model.save_replay_buffer(run_dir / FINAL_REPLAY_BUFFER)
            # Written last: until every final artifact is on disk there is no
            # step count to claim, and a resume that read one early would skip
            # a complete checkpoint for a half-written final.
            save_final_steps(run_dir, model.num_timesteps)

            artifact = wandb.Artifact(name=f"{run_dir.name}-model", type="model")
            artifact.add_file(str(run_dir / FINAL_MODEL))
            artifact.add_file(str(run_dir / FINAL_VECNORM))
            wandb.log_artifact(artifact)
            if cfg.hub.upload and cfg.hub.repo_id:
                url = upload_run(run_dir, cfg.hub.repo_id)
                print(f"[hub] uploaded final model: {url}")
    finally:
        venv.close()
        if not external_wandb:
            wandb.finish()

    return run_dir


@hydra.main(config_path="../../../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
