"""Dream Labs backend: a Dream model on a bimanual YAM through the public `dreamlabs-client` library.

The Client sends one observation per inference call and plays the returned targets at 30 Hz. The model's rows are
played at PLAN_HZ, the rate Dream Labs' own YAM stations play them at, whatever the Client's rate: each call returns
the targets for its next loop steps, interpolated between the rows due then. The dream client keeps the model's frame
window (9 frames at 5 fps) and fetches the next action chunk in the background while the arms play the current one,
so a call normally returns from the installed chunk and waits on the network only when the next chunk has not
arrived yet. One trial is one model session: start_session() opens it and end_session() closes it.
The trial's first call holds the arms still for about 7.5 s while it asks the model again, so the first rows the arms
run already follow the plan of the model's server-side task planner, whose first plan takes 3.5 to 6.5 s.

The API key is read by the dream client itself, from `DREAM_API_KEY` or `~/.dreamlabs/credentials`.
"""
from __future__ import annotations

import asyncio
import base64
import time

import numpy as np

from .base import LocalModel
from .. import colosseum_pb2 as pb
from ..model_adapters.bounds import check_bounds
from ..model_adapters.observation import _rgb, _state

CONTROL_HZ = 30
# Model rows per second, as Dream Labs' YAM stations play them, ramping linearly from one row to the next.
PLAN_HZ = 20
# Targets per call: at 30 Hz, one observation every 0.1 s, twice the model's 5 fps frame window.
ROWS_PER_CALL = 3
# The most plan time one call may add, so a stalled Client resumes the plan where its arms stopped.
MAX_ADVANCE_S = 2 * ROWS_PER_CALL / CONTROL_HZ
# Frames are center-cropped to 16:9, never squished, then sized to FRAME_SIZE.
FRAME_SIZE = (640, 360)
JPEG_QUALITY = 85
CAMERAS = {"head_image": "top", "left_image": "left", "right_image": "right"}
# The dream client paces from its last 5 round trips and a session's first is slow, so until 5 replies are in, the
# next chunk is asked for as each lands, at most once per WARMUP_GAP_S: the API allows one request per 0.7 s.
WARMUP_REPLIES = 5
WARMUP_GAP_S = 1.0
# The task planner's first plan lands 3.5 to 6.5 s into a session and no reply says when, so a trial's first call
# asks HOLD_ASKS more times, HOLD_PERIOD_S apart, the arms still, and plays the reply asked for 7.2 s after the first.
HOLD_ASKS = 3
HOLD_PERIOD_S = 2.4
# Tells the model it drives a YAM and answers joint targets; grippers are 0 closed to 1 open, as the Client's.
YAM = {"robot.arm": "yam", "robot.exec_head": "joint", "robot.max_joint_vel_rad_s": 0.0, "robot.gripper_open": 1.0}


def create_backend(options):
    """Return the backend for the `dreamlabs` entry point."""
    return DreamLabsBackend(options)


def _jpeg(rgb: np.ndarray) -> str:
    """Return an RGB frame center-cropped to 16:9 and sized to FRAME_SIZE, as base64 JPEG."""
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
        """Read the model's checkpoint from the backend options; the session opens in start_session()."""
        self.checkpoint = options["checkpoint"]
        if not isinstance(self.checkpoint, str) or not self.checkpoint:
            raise ValueError("dreamlabs backend requires a checkpoint, e.g. dreamlabs/dream-1-beta")
        self.client = None

    async def start_session(self, model: LocalModel, request) -> None:
        """Check the model is a 14-D YAM joint model and open the trial's dream client session."""
        if model.action_space != "joint_position" or model.action_dim != 14 or model.control_hz != CONTROL_HZ:
            raise ValueError(f"dreamlabs backend requires 14-D YAM joint_position at {CONTROL_HZ} Hz")
        if model.max_horizon < ROWS_PER_CALL:
            raise ValueError(f"dreamlabs backend requires max_horizon of at least {ROWS_PER_CALL}")
        import dream

        self.client = dream.DreamClient()
        # action_hz times the client's prefetch in rows of the plan.
        self.client.policy({"checkpoint": self.checkpoint, "strict": False, "action_hz": PLAN_HZ})
        self.chunk = None  # (H, 14) rows in Client order
        self.chunk_at = 0.0  # plan time of the chunk's row 0
        self.start = 0.0  # the chunk's row the arms took it over at
        self.clock = 0.0  # plan time, s: moves only while the arms play
        self.clock_at = None  # wall time the plan clock last moved; None while the arms wait
        self.playing_s = 0.0
        self.observed_at = 0.0  # wall time of the in-flight request's observation
        self.observed_clock = 0.0
        self.last = None  # the target the arms were last commanded
        self.refill = None  # the in-flight get_action() task
        self.replies = 0

    async def infer(self, model: LocalModel, observation: pb.Observation) -> np.ndarray:
        """Return the targets of the Client's next ROWS_PER_CALL steps, requesting the next chunk when it is due."""
        now = time.monotonic()
        if self.clock_at is not None:  # the arms played the last call's targets
            self.clock += min(now - self.clock_at, MAX_ADVANCE_S)
            self.playing_s += now - self.clock_at
        self._collect()
        self.client.set_observation(*self._wire(observation), timestamp=now)
        if self.replies == 0 and self.refill is None:
            await self._hold()
        warming = (self.replies < WARMUP_REPLIES and self.client.mode == "sync"
                   and now - self.observed_at >= WARMUP_GAP_S)
        due = self.chunk is None or warming
        if not due:
            row = (self.clock - self.chunk_at) * PLAN_HZ
            due = self.client.refill_due(int(row - self.start), len(self.chunk) - int(row))
        if self.refill is None and due:
            self._start_refill(now)
        while (actions := self._targets()) is None:  # the first chunk, or a slow link: the arms hold until one lands
            self.clock_at = None
            if self.refill is None:  # the last one landed too late to use
                self._start_refill(time.monotonic())
            await asyncio.wait({self.refill})
            self._collect()
        self.clock_at = time.monotonic()
        self.last = actions[-1]
        return actions

    async def end_session(self) -> None:
        """Wait for the request in flight, print the playback rate and close the session."""
        if self.client is None:
            return
        if self.refill is not None:
            await asyncio.wait({self.refill})  # the client takes one call at a time, close() included
        if self.playing_s:
            print(f"dreamlabs: played {self.clock * PLAN_HZ:.0f} plan rows in {self.playing_s:.1f} s of play, "
                  f"{self.clock * PLAN_HZ / self.playing_s:.1f} rows/s", flush=True)
        print(f"dreamlabs: closing session {self.client.session_id}", flush=True)
        self.client.close()
        self.client = None

    def _wire(self, observation: pb.Observation):
        """Return the model's frames and state for a Client observation; raise ValueError on a malformed one."""
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
        """Ask the model HOLD_ASKS more times with the first observation, HOLD_PERIOD_S apart; the arms stay still."""
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
        """Request the next chunk in the background for the latest observation."""
        # The new chunk takes over from the row after each arm's last commanded joints.
        commanded = None if self.last is None else {"q_l_chunk": self.last[:7].tolist(), "q_r_chunk": self.last[7:].tolist()}
        self.refill = asyncio.create_task(asyncio.to_thread(self.client.get_action, commanded=commanded))
        self.observed_at, self.observed_clock = now, self.clock

    def _targets(self) -> np.ndarray | None:
        """Return the call's targets, interpolated between the rows due then; None once the chunk has run out."""
        if self.chunk is None:
            return None
        steps_s = np.arange(ROWS_PER_CALL) / CONTROL_HZ
        rows = (self.clock - self.chunk_at + steps_s) * PLAN_HZ  # fractional plan rows, one per loop step
        rows = rows[rows <= len(self.chunk) - 1]
        if not len(rows):
            return None
        below = np.floor(rows).astype(int)
        above = np.minimum(below + 1, len(self.chunk) - 1)
        share = (rows - below)[:, None]
        return self.chunk[below] + share * (self.chunk[above] - self.chunk[below])

    def _collect(self) -> None:
        """Install the in-flight chunk if it has landed, skipping the rows the arms played since its observation."""
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
        chunk[:, [6, 13]] = np.clip(chunk[:, [6, 13]], 0, 1)  # gripper rows can pass [0, 1] by a hair
        # Row k is due k / PLAN_HZ s of plan time after the observation; plan time moves only while the arms play.
        start = (self.clock - self.observed_clock) * PLAN_HZ
        if start > len(chunk) - 1:
            print(f"dreamlabs: discarding a chunk that landed after all {len(chunk)} of its rows were due", flush=True)
            return
        self.chunk, self.chunk_at, self.start = chunk, self.observed_clock, start
