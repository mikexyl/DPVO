# Online CBS on the Jetsons

The deployment runs DPVO and one CPU CBS agent on each Jetson. Each
native agent constructs exactly one BPSAM instance and performs only that robot's
updates. Frozen separator/anchor beliefs are exchanged directly between ROS peers
on `/dpvo_multi_robot/cbs/beliefs`. Robot0 hosts snapshot scheduling, result
aggregation, and the Viser fleet panel; the laptop is only a browser client.
The scheduler does not execute an optimizer for other robots.

Each job contains a fixed graph snapshot and the participating session IDs.
Connected groups with at least two live agents can optimize without the rest of
the fleet. Agents synchronize their first round after initialization, send one
frozen snapshot per round, receive that round's neighbor beliefs, and perform one
local update. Retransmission reuses the frozen snapshot. Duplicates, stale epochs,
and unexpected senders do not advance local state. A failed/disconnected agent or
changed tracking session aborts the epoch. Transforms are committed for a group
only after every member returns a valid result from that same epoch.

This is a synchronous, snapshot-based distributed optimizer, with robot0 as the
scheduler. It is not a leaderless system: existing transforms and local tracking
survive scheduler loss, but new optimization epochs require robot0. The current
runtime supports alternating pose/anchor stages. Camera trackers run in separate
containers and are not restarted to run CBS. Native optimizer processes exist only
while an epoch is active; idle agent processes publish lightweight heartbeats.

## Native build

Build inside a dedicated Docker image on robot0. GTSAM 4.3, aria_common, and CBS
sources are snapshotted by `native/stage_sources.py`; record source commits and
any local changes. Build with `Dockerfile.coordinator.jetson`, then tag its
`builder` stage `dpvo:cbs-native-builder`. `Dockerfile.cbs-agent` reuses those
native dependencies and builds the network adapter. Its source preparation
removes only the unused C++ Rerun visualization frontend in the isolated copy and
adds a single-agent selection to the reference CBS constructor. Solver equations,
factor construction, and belief operations are reused from CBS.

The ARM64 CPU bundle can be layered into each board's existing Ubuntu 24.04 image
with `Dockerfile.cbs-agent.runtime`. Do not replace the board's Torch/CUDA image or
copy its TensorRT engines between JetPack versions. `run_cbs_agent.sh robotX`
starts a CPU-only agent, with no camera devices or GPU runtime. `DPVO_CBS_CPUS`
(default 2) controls its CPU quota. The scheduler/fleet image runs `coordinator.py`
with `DPVO_CBS_MODULE=dpvo_multi_robot.distributed_cbs` (default).

## Validation

`native_agent_smoke.py` compares two and three separate native processes against
the reference CBS runner using the same graphs, stages, and parameters.
`distributed_smoke.py` tests ROS exchange in an isolated domain; `--remote` uses
agents launched on different boards, and `--count 2` checks a missing third robot.
No synthetic test graphs should be published to the production ROS domain.

The fleet and local panels show each CBS agent's phase, round, and received-message
count alongside map merge indicators. Per-agent results are recorded in
`output/robotX/cbs-agent/latest.json`; the scheduler records
`output/coordinator/cbs/distributed_latest.json`. These audits include agent host
names and `optimizer_instances: 1`, plus sent/received belief counts.

## Deployment verified 2026-09-16

Fleet panel: http://192.168.0.156:9091 (robot0). The laptop coordinator was
stopped and its automatic restart disabled. All three production agents and the
robot0 scheduler/panel use Docker `unless-stopped` restart policies. Trackers
remain idle until Start; dense mapping remains disabled by default.

Isolated ROS domain 176 network tests completed 40 belief-exchange rounds across
two Jetsons with robot2 absent, then across all three distinct Jetson hosts.
Every result reported one native optimizer instance. Nonidentity synthetic
2/3-agent graphs matched the reference solver within 1e-6 in native-process tests.
These verify optimizer and transport behavior; a live camera map merge still
requires the cameras to observe enough shared texture for verified loop closures.

The native build passed four C++ tests. The deployed ROS image passed 17 protocol
and session tests; `pixi run verify-multi-robot` passed 18 frontend/graph tests.

## Final-only visualization

Agents exchange beliefs each iteration, but publish map estimates only after the
full cycle. The scheduler validates all participating results before displaying
the alignment. Provisional messages from older agents are ignored.

## Loop threshold trial and rollback

`fleet.yaml` sets `loop_verification.min_inliers: 15` for all robot pairs.
The inlier ratio remains 0.20 and the other geometric checks are unchanged.
Restore this value to 30 and regenerate worker parameters to roll back. Active
workers must be stopped and started to load the threshold; this starts fresh maps
and removes any questionable constraints from the trial. On deployed boards the
previous worker parameters are saved as `robotX.worker.yaml.before-inlier-trial`.
