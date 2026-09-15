# World-model runbook

The published ISS-docking world models, and the exact recipe that produced
them. World models are trained in **quickdraw** (the world-model library), not
in this repo — this repo trains the RL policies and ships the `owm-envs`
docking environments. See `docs/baselines.md` for the RL side.

## Published models

Two latent world models on the Hub, one per target regime, symmetric in every
respect but the data they were trained on:

| model | target | Hub |
| --- | --- | --- |
| `owm-wm-coop` | cooperative (stable) | https://huggingface.co/sislaboratory/owm-wm-coop |
| `owm-wm-noncoop` | non-cooperative (tumbling) | https://huggingface.co/sislaboratory/owm-wm-noncoop |

Each takes a short window of context (proprioceptive state + a first-person
camera) and an action sequence, and predicts the frames and states that follow.
Both cards show 1-step-ahead prediction (ground truth beside predicted) on a
docking approach and an orbit, the open-loop proprioceptive rollout in 3D and
per-axis, and the autoencoder floor. Load either with
`quickdraw.load_pretrained` — snippet at the bottom.

## Reproduce

Trained with quickdraw at commit
[`67c4158`](https://github.com/isaac-ward/quickdraw/commit/67c4158c14fbc3db79dd39ff899c3d46c0cc4a45),
`model=vl128_phys`, on the `owm-iss-numerical-v1` recorded datasets. The two
runs differ **only** in `data.repo_id`:

```bash
# coop
python -m quickdraw.train_world_model \
    model=vl128_phys \
    environments=recorded \
    data.repo_id=owm-iss-numerical-v1-coop-goal-dt50ms-500k \
    data.subsample=5 data.P=8 data.F=64 \
    data.autobatch=false data.batch=26 \
    model.action_dim=6 model.relative_position=true

# noncoop — identical but for the dataset
python -m quickdraw.train_world_model \
    model=vl128_phys \
    environments=recorded \
    data.repo_id=owm-iss-numerical-v1-noncoop-goal-dt50ms-500k \
    data.subsample=5 data.P=8 data.F=64 \
    data.autobatch=false data.batch=26 \
    model.action_dim=6 model.relative_position=true
```

The proprioceptive arm is the ego-13 physics subset (`environments=recorded`
supplies the dynamics prior and the observation layout); no `obs_keep` override
is needed.

### The two load-bearing levers

Everything else is a default, but these two decide whether the run collapses:

- **`model=vl128_phys` bakes `flow_hidden=128`.** Every L1+LPIPS run with the
  wide flow head (`flow_hidden=512`) self-destructed between evals 5 and 12 —
  sharp but spatially wrong, then gone. Setting the flow head to the model
  width (128) is the fix. Do **not** pass `model.diffusion.flow_hidden` on the
  CLI; 128 comes from the config, and 512 is the collapse.
- **`model.relative_position=true`** re-expresses proprioceptive position
  relative to each window's first step. It is what stabilises the shared latent
  at `subsample=5`; the otherwise-identical run without it collapses (latent
  cosine falls to ~0.07). With it, the same run holds (~0.997).

`data.subsample=5` (≈4 Hz) is deliberate: image motion in this data is far
below the codec's temporal floor, so a finer subsample gives a shorter
horizon-span per rollout and visibly better open-loop frames. Do not raise
`window_stride` above 1 or use `accumulate_grad_batches` — both are forbidden
levers in quickdraw and change the effective objective.

### Which checkpoint to publish

**Not the last epoch, and not the monitored-metric best.** These runs peak and
then degrade, and the perceptual metric alone does not flag it — you have to
read the frames.

- **coop → epoch 15.** Stable through the run; epoch 15 is its best clean
  open-loop.
- **noncoop → epoch 7.** This is the last clean epoch: it degrades past epoch 7,
  and its per-epoch metrics are close enough that an automatic best-of picker
  can land on a later, worse checkpoint. Epoch 7 was pinned explicitly and its
  1-step frames were checked to confirm the image head still tracks the station
  (it does — softer and blockier only at the closest range, which is the codec
  floor, not a collapse).

## Publishing

quickdraw's `python -m quickdraw.push_model` stages a self-contained model repo
— `weights.safetensors` (loss net excluded), `training_state.ckpt` for
resuming, `config.resolved.yaml`, the dataset's `normalization_stats.json`,
`example_context.npz`, and `metrics.json`. Pin the checkpoint with
`+hub.ckpt=<path>` (quote it — the `epoch=…` filename confuses hydra's parser)
and the target with `+hub.name=sislaboratory/owm-wm-<coop|noncoop>`; a
`+hub.dry_run=true +hub.out=<dir>` first stages without uploading.

The cards here replace the default open-loop products with the 1-step-ahead
prediction videos, the 3D + per-axis proprioceptive rollouts (cut at 200
steps), and the autoencoder-floor filmstrips, then upload the folder with
`delete_patterns="*"` so a re-push rebuilds the repo rather than unioning with
it.

## Loading

```python
from quickdraw import load_pretrained, load_example_context
import torch

model, norm, cfg = load_pretrained("sislaboratory/owm-wm-coop", device="cuda")
ex = load_example_context("sislaboratory/owm-wm-coop")   # real context windows, shipped with the model

P, H, head = int(cfg.data.P), 64, "image"
ctx = {"proprio": norm.norm_obs(torch.from_numpy(ex["obs"][:, :P])).float().cuda(),
       head:      torch.from_numpy(ex["frames__image"][:, :P]).float().div(255).cuda()}
acts = norm.norm_act(torch.from_numpy(ex["act"][:, :P + H - 1])).float().cuda()

out = model.imagine_eval(ctx, acts, horizon=H, decode_chunk=16)
frames  = out[head].clamp(0, 1)             # imagined frames
proprio = norm.denorm_obs(out["proprio"])   # states, physical units
```

Vectors are normalised (`norm_obs`/`denorm_obs`), frames are plain `[0, 1]`
floats, and `acts` needs `P + H - 1` steps — the context steps consume actions
too. `N missing keys` on load (names containing `visual._net`) is the excluded
LPIPS loss network, and is expected.
