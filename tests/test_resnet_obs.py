import gymnasium as gym
import numpy as np
import pytest
from gymnasium.spaces import Box

from owm.envs.factory import make_iss_env
from owm.envs.resnet_obs import (
    FrozenResnetExtractor,
    ResnetObservationWrapper,
    extractor_kwargs,
)

STATE_DIM = 25
STUB_EMBED_DIM = 4


class StubExtractor:
    """Stands in for the real network: no weights, no GL, no download."""

    embed_dim = STUB_EMBED_DIM

    def __init__(self):
        self.frames: list[tuple[int, ...]] = []

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        self.frames.append(frame.shape)
        return np.full(STUB_EMBED_DIM, 7.0, dtype=np.float32)


class FakeRenderEnv(gym.Env):
    """A renderable env with a recognisable state vector, without the simulator."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: str | None = "rgb_array"):
        self.render_mode = render_mode
        self.observation_space = Box(-np.inf, np.inf, (STATE_DIM,), dtype=np.float32)
        self.action_space = Box(-1.0, 1.0, (6,), dtype=np.float32)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return np.zeros(STATE_DIM, dtype=np.float32), {}

    def step(self, action):
        self._t += 1
        return np.full(STATE_DIM, self._t, dtype=np.float32), 0.0, False, False, {}

    def render(self):
        return np.zeros((224, 224, 3), dtype=np.uint8)


def test_wrapper_space_is_the_state_vector_plus_the_embedding():
    wrapped = ResnetObservationWrapper(FakeRenderEnv(), StubExtractor())
    assert wrapped.observation_space.shape == (STATE_DIM + STUB_EMBED_DIM,)
    assert wrapped.observation_space.dtype == np.float32
    # The embedding half is unbounded: nothing constrains a ResNet feature.
    assert np.all(np.isinf(wrapped.observation_space.high[STATE_DIM:]))
    assert np.all(wrapped.observation_space.low[STATE_DIM:] < 0)


def test_wrapper_concatenates_state_then_embedding_on_reset_and_step():
    extractor = StubExtractor()
    wrapped = ResnetObservationWrapper(FakeRenderEnv(), extractor)

    obs, _ = wrapped.reset()
    assert obs.shape == (STATE_DIM + STUB_EMBED_DIM,)
    assert obs.dtype == np.float32
    # Order is load-bearing: the state vector first, the embedding after it.
    assert np.array_equal(obs[:STATE_DIM], np.zeros(STATE_DIM))
    assert np.array_equal(obs[STATE_DIM:], np.full(STUB_EMBED_DIM, 7.0))

    obs, *_ = wrapped.step(np.zeros(6, dtype=np.float32))
    assert np.array_equal(obs[:STATE_DIM], np.ones(STATE_DIM))
    assert np.array_equal(obs[STATE_DIM:], np.full(STUB_EMBED_DIM, 7.0))

    # A frame per observation, not one cached from construction.
    assert extractor.frames == [(224, 224, 3), (224, 224, 3)]


def test_wrapper_refuses_an_env_that_cannot_render():
    with pytest.raises(ValueError, match="render_mode='rgb_array'"):
        ResnetObservationWrapper(FakeRenderEnv(render_mode=None), StubExtractor())


def test_make_iss_env_rejects_an_unknown_obs_mode_and_a_missing_extractor():
    with pytest.raises(ValueError, match="unknown obs_mode"):
        make_iss_env(None, seed=0, obs_mode="pixels")
    with pytest.raises(ValueError, match="needs an extractor"):
        make_iss_env(None, seed=0, obs_mode="vector_resnet")


@pytest.mark.parametrize(
    ("vec", "device", "expected"),
    [("subproc", "cuda:0", "cpu"), ("dummy", "cuda:0", "cuda:0"), ("dummy", "cpu", "cpu")],
)
def test_extractor_device_defaults_to_cpu_only_in_subproc_workers(vec, device, expected):
    kwargs = extractor_kwargs(
        {"vec": vec, "device": device, "obs_resnet": {"variant": "resnet18", "device": None}}
    )
    assert kwargs["device"] == expected


def test_an_explicit_extractor_device_is_never_overridden():
    kwargs = extractor_kwargs(
        {"vec": "subproc", "device": "cpu", "obs_resnet": {"device": "cuda:0"}}
    )
    assert kwargs["device"] == "cuda:0"


def test_frozen_extractor_embeds_deterministically_and_carries_no_gradient():
    extractor = FrozenResnetExtractor(variant="resnet18", image_size=224, device="cpu")
    assert extractor.embed_dim == 512
    assert not any(p.requires_grad for p in extractor._model.parameters())

    frame = np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    first = extractor(frame)
    assert first.shape == (512,)
    assert first.dtype == np.float32
    # A frozen network in eval mode: the same frame is the same embedding, and
    # a different frame is a different one.
    assert np.array_equal(first, extractor(frame))
    assert not np.array_equal(first, extractor(np.zeros_like(frame)))


def test_frozen_extractor_resizes_frames_that_are_not_its_input_size():
    extractor = FrozenResnetExtractor(image_size=224, device="cpu")
    frame = np.random.default_rng(1).integers(0, 256, (512, 512, 3), dtype=np.uint8)
    assert extractor(frame).shape == (512,)


def test_frozen_extractor_rejects_frames_it_cannot_read():
    extractor = FrozenResnetExtractor(image_size=224, device="cpu")
    with pytest.raises(ValueError, match="HxWx3 uint8"):
        extractor(np.zeros((224, 224), dtype=np.uint8))
    with pytest.raises(ValueError, match="HxWx3 uint8"):
        extractor(np.zeros((224, 224, 3), dtype=np.float32))


def test_frozen_extractor_rejects_a_layer_it_does_not_implement():
    with pytest.raises(ValueError, match="only 'avgpool'"):
        FrozenResnetExtractor(layer="layer4")


def test_wrapper_refuses_an_observation_space_it_cannot_append_to():
    env = FakeRenderEnv()
    env.observation_space = Box(-np.inf, np.inf, (5, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="flat Box observation"):
        ResnetObservationWrapper(env, StubExtractor())
