import asyncio
import base64
import sys
import types

import numpy as np
import pytest

from colosseum_policy_server import colosseum_pb2 as pb
from colosseum_policy_server.backends.base import LocalModel
from colosseum_policy_server.backends.dreamlabs import DreamLabsBackend
from colosseum_policy_server.tensors import tensor_from_numpy

cv2 = pytest.importorskip('cv2')


class FakeDreamClient:
    """Answers each get_action() with rows whose values say which side, row and call they came from."""

    def __init__(self):
        self.observations, self.commanded, self.closed = [], [], False

    def policy(self, config):
        self.config = config

    def set_observation(self, frames, observation, *, timestamp):
        self.observations.append((frames, observation))

    def get_action(self, *, commanded=None):
        self.commanded.append(commanded)
        rows = np.arange(48, dtype=np.float32)[:, None] / 100  # left joints 1.00.., right joints 2.00..
        return {'q_l_chunk': np.c_[np.repeat(rows + 1, 6, 1), np.full((48, 1), .2)],
                'q_r_chunk': np.c_[np.repeat(rows + 2, 6, 1), np.full((48, 1), 1.001)]}

    def refill_due(self, executed, remaining):
        return remaining <= 3

    session_id = 'session'
    mode = 'async'  # 'sync' while the client waits for each chunk to run out

    def close(self):
        self.closed = True


def model():
    return LocalModel.from_mapping(dict(name='dream-1', url='https://huggingface.co/dreamlabs/dream-1', revision='0'*40,
        action_space='joint_position', action_dim=14, control_hz=30, max_horizon=30, endpoint='ws://127.0.0.1:8000'))


def observation(step):
    obs = pb.Observation(control_step=step, instruction='stack the cups')
    obs.state['joint_position'].CopyFrom(tensor_from_numpy(np.r_[np.full(6, -.1), np.full(6, .1)].astype(np.float32)))
    obs.state['gripper_position'].CopyFrom(tensor_from_numpy(np.array([.25, .75], np.float32)))
    for sensor in ('head_image', 'left_image', 'right_image'):
        obs.sensors.add(sensor_id=sensor, encoding=pb.RAW_RGB, width=640, height=480, data=bytes(640 * 480 * 3))
    return obs


async def started(monkeypatch, fake):
    monkeypatch.setitem(sys.modules, 'dream', types.SimpleNamespace(DreamClient=lambda: fake))
    backend = DreamLabsBackend({'checkpoint': 'dreamlabs/dream-1-beta'})
    await backend.start_session(model(), {})
    return backend


@pytest.mark.asyncio
async def test_maps_client_order_to_dream_and_back(monkeypatch):
    fake = FakeDreamClient()
    backend = await started(monkeypatch, fake)
    actions = await backend.infer(model(), observation(0))

    frames, state = fake.observations[0]
    assert sorted(frames) == ['image_window.left', 'image_window.right', 'image_window.top']
    jpeg = cv2.imdecode(np.frombuffer(base64.b64decode(frames['image_window.top']), np.uint8), cv2.IMREAD_COLOR)
    assert jpeg.shape == (360, 640, 3)  # 640x480 center-cropped to 16:9, not squished
    assert state['state.arm_l.joint_pos'] == pytest.approx([-.1] * 6) and state['state.arm_l.gripper_pos'] == [.25]
    assert state['state.arm_r.joint_pos'] == pytest.approx([.1] * 6) and state['state.arm_r.gripper_pos'] == [.75]
    assert state['prompt'] == 'stack the cups' and fake.commanded == [None]
    assert actions.shape == (3, 14)
    assert actions[:, 0] == pytest.approx([1, 1.01, 1.02]) and actions[:, 7] == pytest.approx([2, 2.01, 2.02])
    assert actions[:, 6] == pytest.approx([.2] * 3) and actions[:, 13] == pytest.approx([1] * 3)  # 1.001 clipped

    await backend.end_session()
    assert fake.closed


@pytest.mark.asyncio
async def test_next_chunk_takes_over_from_the_last_commanded_row(monkeypatch):
    fake = FakeDreamClient()
    backend = await started(monkeypatch, fake)
    for step in range(0, 51, 3):
        await backend.infer(model(), observation(step))
    await backend.end_session()  # waits for the request in flight
    # The buffer falls to 3 rows after rows 0-44 run, so the second request carries row 44 as each arm's last command.
    assert fake.commanded[1] == {'q_l_chunk': pytest.approx([1.44] * 6 + [.2]), 'q_r_chunk': pytest.approx([2.44] * 6 + [1])}



@pytest.mark.parametrize('mode, calls', [('sync', 5), ('async', 1)])
@pytest.mark.asyncio
async def test_requests_each_chunk_on_arrival_until_five_replies_while_sync(monkeypatch, mode, calls):
    fake = FakeDreamClient()
    fake.mode = mode
    backend = await started(monkeypatch, fake)
    for step in range(0, 30, 3):  # 30 rows: under the 45 that run before the buffer falls to 3
        await backend.infer(model(), observation(step))
        if backend.refill is not None:
            await asyncio.wait({backend.refill})
    await backend.end_session()
    assert len(fake.commanded) == calls


def joints(values):
    return lambda obs: obs.state['joint_position'].CopyFrom(tensor_from_numpy(np.asarray(values, np.float32)))


@pytest.mark.parametrize('change, error', [
    (lambda obs: obs.sensors.pop(), 'invalid RGB sensor right_image'),
    (lambda obs: obs.sensors.add().CopyFrom(obs.sensors[0]), 'Duplicate camera'),
    (joints(np.zeros(6)), 'invalid observation state joint_position'),
    (joints(np.full(12, np.nan)), 'invalid observation state joint_position'),
    (lambda obs: obs.state['gripper_position'].CopyFrom(tensor_from_numpy(np.array([.5, 1.5], np.float32))), 'grippers'),
    (lambda obs: setattr(obs, 'instruction', ' '), 'instruction is empty'),
])
@pytest.mark.asyncio
async def test_rejects_malformed_observations(monkeypatch, change, error):
    fake = FakeDreamClient()
    backend = await started(monkeypatch, fake)
    obs = observation(0)
    change(obs)
    with pytest.raises(ValueError, match=error):
        await backend.infer(model(), obs)
    await backend.end_session()
    assert fake.observations == [] and fake.commanded == []  # nothing reached the model


@pytest.mark.asyncio
async def test_rejects_non_finite_model_actions(monkeypatch):
    fake = FakeDreamClient()
    fake.get_action = lambda *, commanded=None: {'q_l_chunk': np.full((48, 7), np.nan), 'q_r_chunk': np.zeros((48, 7))}
    backend = await started(monkeypatch, fake)
    with pytest.raises(ValueError, match='non-finite'):
        await backend.infer(model(), observation(0))
    await backend.end_session()
    assert fake.closed
