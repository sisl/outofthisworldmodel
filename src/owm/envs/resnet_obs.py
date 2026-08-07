"""State vector plus a frozen pretrained ResNet's embedding of the frame.

The question this mode exists to answer is whether a policy that also gets an
interpretation of what the docking approach looks like beats one on state
alone. The ResNet is never trained, so its output is a fixed feature of the
frame rather than a representation the policy shapes: the observation stays a
single flat Box and the policy stays an MLP over it.
"""

from __future__ import annotations

from functools import cache

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.functional import interpolate
from torchvision.models import get_model

# The statistics the ImageNet weights were trained under; feeding raw [0, 1]
# pixels to them measures a preprocessing mismatch as much as the image.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@cache
def _backbone(variant: str, device: str) -> tuple[nn.Module, int]:
    """The pretrained network, headless and frozen, once per process.

    Cached because every env in a DummyVecEnv builds its own extractor, and
    loading and moving 45 MB of weights per env is pure startup cost — the
    module is in eval mode with no gradients, so sharing it is read-only.
    """
    model = get_model(variant, weights="IMAGENET1K_V1")
    if not isinstance(getattr(model, "fc", None), nn.Linear):
        raise ValueError(
            f"obs_resnet.variant={variant!r} is not a torchvision ResNet "
            "(no .fc head to read the pooled feature width from)"
        )
    embed_dim = model.fc.in_features
    # Dropping the classifier makes forward() return the avgpool output the
    # classifier would have read, which is the embedding we want.
    model.fc = nn.Identity()
    model.eval().requires_grad_(False).to(device)
    return model, embed_dim


class FrozenResnetExtractor:
    """One rendered frame to one fixed embedding vector."""

    def __init__(
        self,
        variant: str = "resnet18",
        layer: str = "avgpool",
        image_size: int = 224,
        device: str = "cpu",
    ):
        if layer != "avgpool":
            raise ValueError(
                f"obs_resnet.layer={layer!r}; only 'avgpool' is implemented — the "
                "pooled vector the classifier head reads, which is the only "
                "output of a ResNet that is already flat"
            )
        self._device = torch.device(device)
        self._model, self.embed_dim = _backbone(variant, str(self._device))
        self._image_size = int(image_size)
        self._mean = torch.tensor(IMAGENET_MEAN, device=self._device).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self._device).view(3, 1, 1)

    @torch.inference_mode()
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError(
                f"expected an HxWx3 uint8 frame, got shape {frame.shape} "
                f"dtype {frame.dtype}"
            )
        x = torch.from_numpy(np.ascontiguousarray(frame)).to(self._device)
        x = x.permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
        if x.shape[-2:] != (self._image_size, self._image_size):
            x = interpolate(
                x,
                size=(self._image_size, self._image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        x = (x - self._mean) / self._std
        return self._model(x)[0].cpu().numpy()


class ResnetObservationWrapper(gym.ObservationWrapper):
    """Append the embedding of the current frame to the state observation."""

    def __init__(self, env: gym.Env, extractor: FrozenResnetExtractor):
        super().__init__(env)
        if env.render_mode != "rgb_array":
            raise ValueError(
                "ResnetObservationWrapper renders on every reset and step, so its "
                f"env must be built with render_mode='rgb_array' (got "
                f"{env.render_mode!r})"
            )
        self._extractor = extractor
        base = env.observation_space
        # Nothing bounds a ResNet feature, so the embedding half of the space is
        # infinite; VecNormalize clips what actually shows up.
        bound = np.full(extractor.embed_dim, np.inf, dtype=np.float32)
        self.observation_space = Box(
            low=np.concatenate([base.low, -bound]).astype(np.float32),
            high=np.concatenate([base.high, bound]).astype(np.float32),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        # Rendered here rather than cached from the previous step so the frame
        # is the one the state vector beside it describes.
        embedding = self._extractor(self.env.render())
        return np.concatenate(
            [np.asarray(observation, dtype=np.float32), embedding]
        ).astype(np.float32)


def extractor_kwargs(rl_cfg: DictConfig | dict) -> dict:
    """FrozenResnetExtractor arguments for an rl config node, device resolved."""
    if isinstance(rl_cfg, DictConfig):
        rl_cfg = OmegaConf.to_container(rl_cfg, resolve=True)
    kwargs = dict(rl_cfg["obs_resnet"])
    if kwargs.get("device") is None:
        # Every subproc worker builds its own extractor, and a CUDA context per
        # worker costs both seconds of startup and hundreds of MB of device
        # memory for a 45 MB network, so workers embed on the CPU. A dummy vec
        # embeds inside the learner's own process and can share its device.
        kwargs["device"] = "cpu" if rl_cfg["vec"] == "subproc" else str(rl_cfg["device"])
    return kwargs
