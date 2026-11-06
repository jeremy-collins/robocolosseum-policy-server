"""Dream Labs backend: a Dream model on a bimanual YAM through the public `dreamlabs-client` library.

The Client sends one observation per inference call and plays the returned rows at 30 Hz. The dream client keeps
the model's frame window (9 frames at 5 fps) and fetches the next action chunk in the background while the arms play
the current one, so a call normally returns a few rows from a local buffer and waits on the network only when the
next chunk has not arrived yet. One trial is one model session: start_session() opens it and end_session() closes it.
The trial's first call holds the arms still for about 6 s while it asks the model again, so the first rows the arms
run already follow the plan of the model's server-side task planner, whose first plan takes 4 to 5.5 s.

The API key is read by the dream client itself, from `DREAM_API_KEY` or `~/.dreamlabs/credentials`.
"""
from __future__ import annotations

import asyncio
import base64
import collections
import time

import numpy as np

from .base import LocalModel
from .. import colosseum_pb2 as pb
from ..model_adapters.bounds import check_bounds
from ..model_adapters.observation import _rgb, _state

# The YAM's control rate, and the rate of the model's rows.
CONTROL_HZ = 30
# Rows returned per call. The Client sends its next observation only once these have run, so 3 rows at 30 Hz is
# one observation every 0.1 s, twice the rate of the model's 5 fps frame window.
ROWS_PER_CALL = 3
# The size the model sees; a camera of another shape is center-cropped to 16:9 first, never squished.
FRAME_SIZE = (640, 360)
JPEG_QUALITY = 85
# Client camera -> the model's camera: the head camera and the left and right wrist cameras.
CAMERAS = {"head_image": "top", "left_image": "left", "right_image": "right"}
# The dream client decides between fetching ahead and waiting for each chunk to run out from the median of its last
# five round trips, and a session's first round trip is slow (the model opens the session). Until this many replies
# have outvoted it, the next chunk is requested as soon as one lands, but never sooner than WARMUP_GAP_S after the
# last request: the API allows a session one request per 0.7 s, three at most in a burst.
WARMUP_REPLIES = 5
WARMUP_GAP_S = 1.0
# The model's server-side task planner makes its first plan 4 to 5.5 s after a session opens, proposing at most every
# 1.5 s, and no reply says when a plan is in force. So a trial's first call asks again HOLD_ASKS times, HOLD_PERIOD_S
# apart, the arms still, and plays the last reply, asked for 6 s after the first.
HOLD_ASKS = 3
HOLD_PERIOD_S = 2.0
# Tells the model it drives a YAM and answers joint targets; grippers are 0 closed to 1 open, as the Client's.
YAM = {"robot.arm": "yam", "robot.exec_head": "joint", "robot.max_joint_vel_rad_s": 0.0, "robot.gripper_open": 1.0}


def create_backend(options):
    return DreamLabsBackend(options)


def _jpeg(rgb: np.ndarray) -> str:
    import cv2

    height, width = rgb.shape[:2]
    crop_h, crop_w = min(height, round(width * 9 / 16)), min(width, round(height * 16 / 9))
    top, left = (height - crop_h) // 2, (width - crop_w) // 2
    frame = rgb[top:top + crop_h, left:left + crop_w]
    if (crop_w, crop_h) != FRAME_SIZE:
        frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    # The model decodes with OpenCV, which reads BGR.
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


class DreamLabsBackend:
    """One trial's dream client session and the buffer of rows the arms have not run yet."""

    def __init__(self, options):
        self.checkpoint = options["checkpoint"]
        if not isinstance(self.checkpoint, str) or not self.checkpoint:
            raise ValueError("dreamlabs backend requires a checkpoint, e.g. dreamlabs/dream-1-beta")
        self.client = None

    async def start_session(self, model: LocalModel, request) -> None:
        if model.action_space != "joint_position" or model.action_dim != 14 or model.control_hz != CONTROL_HZ:
            raise ValueError(f"dreamlabs backend requires 14-D YAM joint_position at {CONTROL_HZ} Hz")
        if model.max_horizon < ROWS_PER_CALL:
            raise ValueError(f"dreamlabs backend requires max_horizon of at least {ROWS_PER_CALL}")
        import dream

        self.client = dream.DreamClient()
        self.client.policy({"checkpoint": self.checkpoint, "strict": False, "action_hz": CONTROL_HZ})
        self.rows = collections.deque()  # (14,) rows in Client order, not yet returned
        self.installed = 0  # rows of the current chunk when it was installed
        self.since_observation = 0  # rows returned since the in-flight request's observation
        self.observed_at = 0.0
        self.last = None  # the last row returned: what the arms were last commanded
        self.refill = None  # the in-flight get_action() task
        self.replies = 0  # replies this session has received

    async def infer(self, model: LocalModel, observation: pb.Observation) -> np.ndarray:
        now = time.monotonic()
        self._collect(now)
        self.client.set_observation(*self._wire(observation), timestamp=now)
        if self.replies == 0 and self.refill is None:
            await self._hold()
        warming = (self.replies < WARMUP_REPLIES and self.client.mode == "sync"
                   and now - self.observed_at >= WARMUP_GAP_S)
        due = not self.rows or warming or self.client.refill_due(self.installed - len(self.rows), len(self.rows))
        if self.refill is None and due:
            self._start_refill(now)
        while not self.rows:  # the first chunk, or a slow link: the arms hold until a chunk lands
            if self.refill is None:  # the last one landed too late to use
                self._start_refill(now)
            await asyncio.wait({self.refill})
            self._collect(time.monotonic())
        actions = np.stack([self.rows.popleft() for _ in range(min(ROWS_PER_CALL, len(self.rows)))])
        self.since_observation += len(actions)
        self.last = actions[-1]
        return actions

    async def end_session(self) -> None:
        if self.client is None:
            return
        if self.refill is not None:
            await asyncio.wait({self.refill})  # the client takes one call at a time, close() included
        print(f"dreamlabs: closing session {self.client.session_id}", flush=True)
        self.client.close()
        self.client = None

    def _wire(self, observation: pb.Observation):
        if not observation.instruction.strip():
            raise ValueError("observation instruction is empty")
        images = {image.sensor_id: image for image in observation.sensors}
        if len(images) != len(observation.sensors):
            raise ValueError("Duplicate camera sensor IDs")
        joints = _state(observation, "joint_position", (12,))  # left 6, right 6
        grippers = _state(observation, "gripper_position", (2,))  # left, right
        check_bounds(grippers, 0, 1, ("left_gripper", "right_gripper"), "YAM grippers must be in [0, 1]")
        frames = {f"image_window.{camera}": _jpeg(_rgb(images.get(source), source)) for source, camera in CAMERAS.items()}
        state = {"prompt": observation.instruction, **YAM,
                 "state.arm_l.joint_pos": joints[:6].tolist(), "state.arm_l.gripper_pos": grippers[:1].tolist(),
                 "state.arm_r.joint_pos": joints[6:].tolist(), "state.arm_r.gripper_pos": grippers[1:].tolist()}
        return frames, state

    async def _hold(self) -> None:
        """Asks the model HOLD_ASKS more times with the first observation, HOLD_PERIOD_S apart; the arms stay still."""
        started = time.monotonic()
        for _ in range(HOLD_ASKS):
            self._start_refill(time.monotonic())
            await asyncio.wait({self.refill})
            self.refill.result()  # this reply is dropped, but a failed call still raises
            self.refill = None
            self.replies += 1
            await asyncio.sleep(max(0.0, self.observed_at + HOLD_PERIOD_S - time.monotonic()))
        self._start_refill(time.monotonic())
        await asyncio.wait({self.refill})
        print(f"dreamlabs: held the first chunk {time.monotonic() - started:.1f} s for the planner", flush=True)

    def _start_refill(self, now: float) -> None:
        # The joints each arm was last commanded: the new chunk takes over from the row after them.
        commanded = None if self.last is None else {"q_l_chunk": self.last[:7].tolist(), "q_r_chunk": self.last[7:].tolist()}
        self.refill = asyncio.create_task(asyncio.to_thread(self.client.get_action, commanded=commanded))
        self.observed_at, self.since_observation = now, 0

    def _collect(self, now: float) -> None:
        """Installs the in-flight chunk if it has landed, past the rows the arms ran since its observation."""
        if self.refill is None or not self.refill.done():
            return
        reply, self.refill = self.refill.result(), None
        self.replies += 1
        left, right = np.asarray(reply["q_l_chunk"], np.float32), np.asarray(reply["q_r_chunk"], np.float32)
        if left.ndim != 2 or left.shape != right.shape or left.shape[1] != 7 or not len(left):
            raise ValueError(f"dream returned chunks of shapes {left.shape} and {right.shape}, not matching (H, 7)")
        chunk = np.concatenate([left, right], axis=1)  # left 6, left gripper, right 6, right gripper
        if not np.isfinite(chunk).all():
            raise ValueError("dream returned non-finite actions")
        # The model's gripper rows can pass fully open or closed by a hair (1.004); the gripper's range is [0, 1].
        chunk[:, [6, 13]] = np.clip(chunk[:, [6, 13]], 0, 1)
        # Row k is the target for the observation time + k rows; skip the rows the arms ran meanwhile, never more
        # than wall time says are due, so after a hold the plan resumes where the arms stopped.
        skipped = min(self.since_observation, round((now - self.observed_at) * CONTROL_HZ))
        if skipped >= len(chunk):
            print(f"dreamlabs: discarding a chunk that landed after all {len(chunk)} of its rows were due", flush=True)
            return
        self.rows = collections.deque(chunk[skipped:])
        self.installed = len(self.rows)
