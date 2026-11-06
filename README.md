# RoboColosseum Policy Server

Serve policy inference for the [RoboColosseum](https://robocolosseum.ai/)
Open Track. The Policy Server receives robot observations, runs inference,
and returns actions for the Client to execute.

## Installation

Requirements: Python 3.10 or later, Git, and `uv` available on your PATH.
Run the following on the machine that will host the Policy Server:

```bash
git clone git@github.com:frodobots-org/colosseum-policy-server.git
cd colosseum-policy-server
uv sync
```

## Host a Local Policy Server

Run the Policy Server on your inference computer and connect the Robot Client to it.

### Test Backend

Test communication without loading a model:

```bash
uv run colosseum-policy-test-backend
```

Connect the Client to `ws://127.0.0.1:8000` with `test: true`. The server returns
received joint and gripper positions as a one-step `joint_position` action.
The action dimension must match the combined state length. No model inference
is performed.

For MolmoAct2-BimanualYAM, follow the [YAM setup guide](docs/yam.md) and
use [the YAM runtime example](configs/local-runtime-yam.yaml.example).
For MolmoAct2 vs LeRobot GR00T YAM, use the
[two-model configuration](configs/local-runtime-yam-vla-pair.yaml.example) and
[deployment guide](docs/yam.md#molmoact2-vs-gr00t-yam-deployment).
For YAM with Grok 4.7, Opus 5.5 or GPT-6 Astra, use the
[YAM Inspect Agent example](configs/local-runtime-yam-agent.yaml.example).
For NUSMAGIC pi05-MolmoAct2-YAM, use the
[pi05 YAM configuration](configs/local-runtime-yam-pi05.yaml.example) and
[worker setup](docs/yam.md#pi05-yam).
For RoboColosseum G05 YAM, see the [bridge setup and training metadata](docs/yam.md#g05-yam)
and [runtime configuration](configs/local-runtime-yam-g05.yaml.example).

For LingBot-VLA v2 YAM, see the [worker setup](docs/lingbot-yam.md) and
[launcher configuration](configs/local-runtime-yam-lingbot.yaml.example).

To record and review YAM A/B evaluations with all five VLAs on this GPU and
the three LLMs under Inspect EE control, follow the Client's
[local evaluation guide](https://github.com/frodobots-org/colosseum-client/blob/main/docs/local-evals.md)
with [this configuration](configs/local-runtime-yam-local.yaml.example).

For the SO-ARM101 generalist checkpoints, follow the [SO101 setup guide](docs/so101.md)
and use [the SO101 runtime example](configs/local-runtime-so101.yaml.example).

### Run a Real Robot Policy Server

Install the DROID and LLM adapters once:

```bash
uv sync --extra droid --extra agent
```

Put your registered models and rig settings in `configs/local-runtime.yaml`,
then start:

```bash
.venv/bin/colosseum-policy-local
```

Omit `backend.name` to automatically choose VLA inference or Inspect LLM
inference from the assigned model. VLA inference selects its model adapter using
`backend_options.adapter`; it does not select a hardware driver. Both model types must be registered in the configuration.
VLA model services still need their weights and dependencies; start them
separately or configure their launchers. Use `--config /path/to/policy.yaml`
for a different configuration. Explicit `backend.name` and `--robot` remain
available for existing setups and custom robot backends.

### Run a Dream Labs model

Dream Labs models (`dreamlabs/dream-1-beta` on a bimanual YAM) run on Dream Labs' API through the public
`dreamlabs-client` library, so nothing runs on a local GPU:

```bash
uv sync --extra dreamlabs
export DREAM_API_KEY=<your key>  # or put it in ~/.dreamlabs/credentials
.venv/bin/colosseum-policy-local --config configs/local-runtime-yam-dreamlabs.yaml.example
```

The `dreamlabs` backend keeps the model's frame window and fetches the next action chunk while the arms play the
current one, so each inference call returns 3 rows (0.1 s at 30 Hz) from a local buffer and only the first call of
a trial waits on the network. Each trial is one model session, opened when the trial starts and closed when it
ends. Camera frames are center-cropped to 16:9 for the model.

### Use a cloud LLM through Inspect Robots

The optional `inspect_agent` backend runs GPT-6 Astra, Grok 4.7, or Claude
Opus 5.5 through provider APIs and returns the same Protobuf action plans.
It needs API credentials and rig bounds, but no GPU or model weights:

```bash
uv sync --locked --extra agent
uv run --locked --extra agent colosseum-policy-local --config configs/local-runtime-agent.yaml
```

Copy and fill in `configs/local-runtime-agent.yaml.example` first.
See [inference architecture](docs/inference-architecture.md) for robot/model
compatibility and mixed VLA/LLM setup. See
[Inspect Agent setup](docs/inspect-agent.md) for provider configuration,
existing Router registration, and holding position after an agent stop request.

## Host a Remote Policy Server

Host your models on your own inference machine so evaluators can use your compute
resources and models to run evaluations. The
[Robot Client](https://github.com/frodobots-org/colosseum-client) exchanges
observations and actions with your Policy Server through the Router.

### Run the Example

Install the demo dependencies and create the Policy Server configuration:

```bash
uv sync --extra demo
cp configs/policy.yaml.example configs/policy.yaml
```

Edit it with your Router endpoint and **Policy Server token** from the Router admin:

```yaml
url: wss://router.example.com:8443
token: pol_replace_with_token_from_router_ui
```

Start the SDK example:

```bash
uv run --extra demo python examples/test_policy.py
```

The example connects to the Router and waits for observations. When an evaluator
starts a remote evaluation and the Router matches their Client to your Policy
Server, you will see task instructions, joint/gripper state, and image shapes in
the terminal. Camera images appear in OpenCV windows, so run this example with a
graphical display. Press `q` or Escape in an image window to exit.

The example also sends demo actions: it holds the observed joint positions and
alternates the gripper target between `1` and `0`. To only inspect incoming data,
run:

```bash
uv run --extra demo python examples/test_policy.py --no-enable-action
```

Observation-only mode sends no actions, so the Client may reach its inference
deadline. After checking the connection and observation format, replace the demo
policy with your own model using the SDK below.

### Connect your model

Install your model's dependencies in the same Python environment. Create
`serve_policy.py` in the repository root using this template:

```python
from colosseum_policy_server import ColosseumPolicySDK
from your_policy import load_model  # Replace with your own model loader.


def main():
    model = load_model()

    with ColosseumPolicySDK.from_yaml("configs/policy.yaml") as sdk:
        print("Connected to Router. Waiting for observations...", flush=True)
        while True:
            obs = sdk.get_obs()
            actions = model.infer(obs)
            sdk.send_action(actions, observation=obs)


if __name__ == "__main__":
    main()
```

`your_policy` is your own module, not a package included in this repository.
Replace the import and model-loading call with your implementation. Your
`model.infer(obs)` adapter should convert the observation to your model's inputs
and return a NumPy array shaped `(horizon, action_dim)` in the Client's agreed
action space.

For example, observations expose:

```python
obs.instruction        # Task instruction.
obs.state.head_image   # RGB uint8 image; None if absent.
obs.state.left_image
obs.state.right_image
obs.state.joints       # Robot joint positions.
obs.state.gripper      # Gripper state.
```

For JPEG/PNG observations, install the image-decoding dependency with
`uv sync --extra demo`.

The YAML-based SDK defaults to `joint_position`, 15 Hz, and a maximum 16-step
action horizon. These settings must suit your model and the Client. To customize
them, construct `ColosseumPolicySDK` directly with `router_url`, `token`,
`action_spaces`, `control_hz`, and `max_horizon` instead of using `from_yaml`.

After matching, `sdk.robot` exposes the Client's hardware specification and
`sdk.selected_action_space` identifies the selected action space. The SDK checks
action dimensions, horizon, finite values, and observation deadlines before
sending actions.

### Navigation policies (Earth Rover Mini+)

The Earth Rover Mini+ (`robot_type: earth_rover_mini_plus`) is a mobile base
with no joints: its action space is `base_velocity`, two normalized values
`[linear, angular]` in `[-1, 1]` (`+angular` turns left) at 10 Hz. Declare it
in `configs/policy.yaml`:

```yaml
url: wss://YOUR_ROUTER
token: REPLACE_WITH_POLICY_TOKEN
action_spaces: [base_velocity]
control_hz: 10
max_horizon: 10
```

Each observation carries:

```python
obs.image("front_image")               # Always present; obs.image("rear_image") when the rover publishes it.
obs.raw_state["navigation_state"]      # float64 [latitude, longitude, heading_deg, speed, battery, rpm_1..rpm_4]
obs.raw_state.get("goal_gps")          # [latitude, longitude] of the destination; GPS goals only
obs.image("goal_image")                # The reference photo; photo goals only (no GPS is sent)
```

When the policy has reached the goal, send a zero chunk with `done=True`; the
Client confirms the stop and ends the trial:

```python
sdk.send_action(np.zeros((1, 2)), observation=obs, done=True)
```

The Client stops the rover on its own if no new chunk arrives within its
command TTL (1 s by default), so keep chunks at least a few steps long.
`examples/rover_policy.py` is a minimal working policy.

#### Policies written for the Earth Rovers SDK

A policy that drives the rover through the Earth Rovers SDK's HTTP API
(`localhost:8000`: `/v2/screenshot`, `/data`, `/control`, `/start-mission`,
`/checkpoints-list`, `/checkpoint-reached`) runs unchanged behind
`colosseum-rover-sdk-bridge`. The bridge serves those endpoints from the current
trial (cameras as base64 PNG, telemetry from `navigation_state`, the GPS goal as
the mission's only checkpoint) and answers every observation with the policy's
latest `/control` command. `/checkpoint-reached` is accepted within 15 m of the
goal, like the FrodoBots backend, and ends the trial with `done=True`.

```bash
colosseum-rover-sdk-bridge --config configs/policy.yaml \
  --cwd ../my-rover-policy --run "python missions/run.py --base-url http://127.0.0.1:8000"
```

`--run` starts the policy when a trial begins and sends it SIGINT when the trial
ends; if the policy exits during a trial, the trial is declared done. Without
`--run`, start the policy yourself after the operator starts the trial. SDK
commands are sent unchanged when the operator's `adapter_config.max_linear` and
`max_angular` are 1; otherwise pass the same values as `--max-linear` and
`--max-angular`. The Router is far from most rovers, so each chunk holds the
command for `--horizon` steps (5, half a second) before the Client asks again.

### Start the Policy Server

After implementing your model loader and inference adapter, run:

```bash
uv run python serve_policy.py
```

The script loads your model, connects to the Router, and waits for observations.
Keep it running while evaluators use your policy. Once the Router matches a
Client to your Policy Server, each observation is passed to your model and the
resulting actions are sent back through the Router.

### LLM relay endpoints

For typed `model_type: llm` entries, `url` is the HTTPS API base URL and must
match the Router deployment. Model names choose defaults: `gpt-*` uses Responses,
`grok-*` uses Chat Completions, and `claude-*` uses Anthropic Messages. The raw
model name is sent to the provider by default. If the Router's public name has
a rig suffix, set `backend_options.api_model` to the provider's actual model ID.
Use the Router preparation's `model.name`; the leaderboard `policyId` can have a
different suffix. If the Router's public URL is a relay, set `backend_options.api_base_url` to the
official API URL for direct inference. Neither override changes the Router
identity used by the local runtime. Claude Code account login uses the same
`api_model` with the official CLI.
A root Anthropic URL is normalized to `/v1` for the Messages client.

Keys still come from the Client, per trial; no keys belong in this config or
Router. Optional per-model `backend_options.api_format` accepts Inspect wire
names `responses`, `chat`, or `messages`. Anthropic defaults to `x-api-key`;
set `backend_options.api_auth: bearer` for a relay requiring Bearer tokens.
Set `backend_options.llm_endpoint: official` to keep the Router `url` as the
model identity only and send official Client keys to the OpenAI, xAI or
Anthropic API instead.
See `configs/local-runtime-yhlxj.yaml.example`. Copy your existing measured
robot limits, gripper polarity, camera names, and notes; example null limits
intentionally prevent execution until configured.

Upgrade Router first, then Client and Policy Server. New Clients report
`llm_providers` alongside legacy `llm_api_urls`; old Clients remain eligible
only for matching official URLs. Start a new assignment after switching the
catalog endpoint. Synthetic API smoke tests do not establish model provenance
or robot task performance; the tested Grok relay reported `grok-4.7-build`.

The pinned Inspect Responses client automatically adds explicit GPT prompt-cache
breakpoints. These are disabled for relay hosts because the tested gateway
rejects them; full conversation history is still sent. Official OpenAI behavior
is retained. Recheck this compatibility shim when upgrading Inspect.
