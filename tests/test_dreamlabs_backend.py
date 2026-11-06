import asyncio
import base64
import sys
import time
import types

import numpy as np
import pytest

from colosseum_policy_server import colosseum_pb2 as pb
from colosseum_policy_server.backends.base import LocalModel
from colosseum_policy_server.backends import dreamlabs
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
    monkeypatch.setattr(dreamlabs, 'HOLD_ASKS', 0)  # no first-call hold unless a test sets one
    backend = DreamLabsBackend({'checkpoint': 'dreamlabs/dream-1-beta'})
    await backend.start_session(model(), {})
    return backend


def fake_clock(monkeypatch):
    """Gives the backend a clock the test moves; returns the function that moves it, by seconds."""
    now = [100.0]
    monkeypatch.setattr(dreamlabs, 'time', types.SimpleNamespace(monotonic=lambda: now[0]))
    return lambda seconds: now.__setitem__(0, now[0] + seconds)


# The plan rows of a call's 3 loop steps at 30 Hz, from its first: 20 plan rows/s.
STEP_ROWS = np.arange(3) * 20 / 30


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
    assert actions[:, 0] == pytest.approx(1 + STEP_ROWS / 100) and actions[:, 7] == pytest.approx(2 + STEP_ROWS / 100)
    assert actions[:, 6] == pytest.approx([.2] * 3) and actions[:, 13] == pytest.approx([1] * 3)  # 1.001 clipped

    await backend.end_session()
    assert fake.closed


@pytest.mark.asyncio
async def test_next_chunk_takes_over_from_the_last_commanded_row(monkeypatch):
    fake = FakeDreamClient()
    advance = fake_clock(monkeypatch)
    backend = await started(monkeypatch, fake)
    for step in range(0, 72, 3):
        await backend.infer(model(), observation(step))
        advance(0.1)
    await backend.end_session()  # waits for the request in flight
    # Call 23 starts at plan row 46, 2 rows from the end, so the second request carries call 22's last target, plan
    # row 44 + 4/3, as each arm's last command.
    assert fake.commanded[1] == {'q_l_chunk': pytest.approx([1 + .44 + .04 / 3] * 6 + [.2]),
                                 'q_r_chunk': pytest.approx([2 + .44 + .04 / 3] * 6 + [1])}


@pytest.mark.asyncio
async def test_plays_twenty_plan_rows_a_second_at_any_loop_rate(monkeypatch, capsys):
    fake = FakeDreamClient()
    advance = fake_clock(monkeypatch)
    backend = await started(monkeypatch, fake)
    played = []
    for step in range(0, 30, 3):  # 1 s of a 30 Hz loop
        played.append(await backend.infer(model(), observation(step)))
        advance(0.1)
    await backend.end_session()
    assert np.concatenate(played)[:, 0] == pytest.approx(1 + np.arange(30) * 20 / 30 / 100)  # rows 0 to 19.3, in 1 s
    assert '20.0 rows/s' in capsys.readouterr().out



@pytest.mark.parametrize('mode, calls', [('sync', 5), ('async', 1)])
@pytest.mark.asyncio
async def test_requests_each_chunk_on_arrival_until_five_replies_while_sync(monkeypatch, mode, calls):
    fake = FakeDreamClient()
    fake.mode = mode
    advance = fake_clock(monkeypatch)
    backend = await started(monkeypatch, fake)
    monkeypatch.setattr(dreamlabs, 'WARMUP_GAP_S', 0.0)
    for step in range(0, 30, 3):  # 20 plan rows: under the 45 that play before the buffer falls to 3
        await backend.infer(model(), observation(step))
        if backend.refill is not None:
            await asyncio.wait({backend.refill})
        advance(0.1)
    await backend.end_session()
    assert len(fake.commanded) == calls



@pytest.mark.asyncio
async def test_first_call_holds_the_arms_and_keeps_the_reply_asked_for_after_the_hold(monkeypatch):
    fake = FakeDreamClient()
    calls = []

    def get_action(*, commanded=None):
        calls.append(time.monotonic())
        reply = FakeDreamClient.get_action(fake, commanded=commanded)
        reply['q_l_chunk'][:, 0] += len(calls)  # tags each reply with its call number
        return reply

    fake.get_action = get_action
    backend = await started(monkeypatch, fake)
    monkeypatch.setattr(dreamlabs, 'HOLD_ASKS', 3)
    monkeypatch.setattr(dreamlabs, 'HOLD_PERIOD_S', 0.05)
    start = time.monotonic()
    actions = await backend.infer(model(), observation(0))
    assert time.monotonic() - start >= 0.15  # no rows reach the arms during the hold
    assert len(calls) == 4  # the first ask and 3 more
    assert min(b - a for a, b in zip(calls, calls[1:])) >= 0.05  # never faster than HOLD_PERIOD_S
    assert actions[:, 0] == pytest.approx(5 + STEP_ROWS / 100)  # rows of the 4th reply
    second = time.monotonic()
    await backend.infer(model(), observation(3))
    assert time.monotonic() - second < 0.1  # later calls do not hold
    await backend.end_session()


@pytest.mark.asyncio
async def test_warmup_waits_the_minimum_gap_between_requests(monkeypatch):
    fake = FakeDreamClient()
    fake.mode = 'sync'
    advance = fake_clock(monkeypatch)
    backend = await started(monkeypatch, fake)
    monkeypatch.setattr(dreamlabs, 'WARMUP_GAP_S', 10.0)
    for step in range(0, 30, 3):
        await backend.infer(model(), observation(step))
        if backend.refill is not None:
            await asyncio.wait({backend.refill})
        advance(0.1)
    await backend.end_session()
    assert len(fake.commanded) == 1  # the warm-up never asked again inside the gap


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
