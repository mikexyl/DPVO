# Wi-Fi-independent local operation

## Transport selection

`fleet.yaml` currently selects `network.transport: cyclone` for a direct DDS
comparison on the three Jetsons. `network.interface_address: 192.168.0.0`
pins DDS to the fleet LAN subnet; automatic selection at boot previously chose
robot0's USB interface and robot2's Docker interface before Wi-Fi was ready.
It uses unicast discovery with the configured robot addresses
(plus localhost). Multicast remains disabled. Stop `dpvo-network-robotX` and set
its Docker restart policy to `no` in this mode; do not run both transports.
The selected subnet must exist before DDS can start. Keep the DPVO ROS containers
on an automatic Docker restart policy so they retry when Wi-Fi becomes available.
This direct mode does not provide a local panel before the LAN is available;
use bridge mode for offline boot operation.
Restart all DPVO ROS containers after changing `cyclonedds.xml`. Existing tracker
sessions end on that restart; use Start to create a fresh session.

Direct mode persists across reboot, but the offline startup and interface-change
guarantees below apply only to `network.transport: zenoh` (also the default when
the setting is absent). Direct DDS recovery after Wi-Fi roaming must be tested
on the actual access points. To roll back, select `zenoh`, regenerate the files,
restart the ROS containers, and restore the bridges' `always` restart policy.
Do not change Wi-Fi profiles to switch ROS transports.

## Zenoh bridge mode

Every ROS process, including each bridge, uses the generated `cyclonedds.xml`:
DDS binds only to 127.0.0.1 and discovers local participants with unicast. No
V4RL address is required to start a camera, controller, local panel, or CBS agent.
Browser servers continue listening on all interfaces, including USB Ethernet.

A CPU-only `dpvo-network-robotX` container bridges selected DPVO topics and
services between boards using pinned `zenoh-bridge-ros2dds` 1.10.1. Native ROS 2
Jazzy/Cyclone DDS and the per-board CBS optimizers are unchanged. Bridges form a
peer mesh; no laptop or central network router process is required. Robot0 still
hosts the CBS snapshot scheduler and fleet panel.

The bridge listens on TCP 17473 (configurable with `network.bridge_port`), retries
connections indefinitely with 1–4 second backoff, and never binds its listener to
a DHCP address. Multicast and gossip scouting are disabled. Use a dedicated port:
robot1 already has unrelated Zenoh services on 7447. Remote interfaces are limited
to configured `/<robot>/dpvo/*` and `/dpvo_multi_robot/*` namespaces; raw camera
images and unrelated sensor topics are not bridged. Congestion drops remote
publications rather than blocking local camera processing. CBS has explicit
round synchronization/timeouts/retransmission and aborts incomplete epochs.

## Behavior

- Boot without V4RL: local DPVO and Start/Stop work over USB Ethernet. Camera
  remains off until Start. Robot0 USB fleet URL: http://192.168.55.1:9091; local
  panel: http://192.168.55.1:9090. An unplugged network does not restart tracking.
- Wi-Fi loss: local camera/tracking continues; remote heartbeats become stale.
  Local progress does not depend on CBS. An interrupted CBS job is cancelled;
  existing display alignment may remain until a fresh valid result arrives.
- Wi-Fi returns: bridges reconnect automatically; remote status, previews,
  services and CBS become available again. Discovery recovery is eventual,
  typically several seconds. A timed-out button call is not queued for later
  execution; retry after the robot shows online.
- Saved V4RL profiles retain NetworkManager autoconnect. Explicitly disabling
  Wi-Fi or manually disconnecting in NetworkManager requires re-enabling it.
- Peer addresses in `fleet.yaml` must remain valid (use DHCP reservations or
  update/regenerate after an address change). This handles interface loss and
  reappearance, not arbitrary changes to every peer's configured address.

## Build and deploy

Generate matching configs for the whole fleet. All DPVO ROS services on a board
must use local-only DDS before starting its bridge, to avoid duplicate routes.
Each board requires exactly one production bridge. Do not bridge unrelated
colleague ROS domains or change existing host network services.

```bash
python3 deploy/jetson/multi_robot/fleet.py --output deploy/jetson/multi_robot/generated
python3 deploy/jetson/multi_robot/prepare_network_bridge.py --arch aarch64
# JP7.2; use dpvo:online-jp62-shared for JP6.2
 docker build --network host --build-arg BASE_IMAGE=dpvo:online-jp72-shared \
   -f deploy/jetson/multi_robot/Dockerfile.network -t dpvo:network-bridge-1.10.1 .
bash deploy/jetson/multi_robot/run_network_bridge.sh robot0
```

The preparation script verifies official release SHA-256 hashes; downloaded
binaries are ignored by Git. `DPVO_BRIDGE_IMAGE`, `DPVO_BRIDGE_CONTAINER_NAME`,
and `DPVO_FLEET_DIR` override launcher defaults. Bridge containers restart after
boot. Switching an existing fleet requires one restart of its idle controllers,
panels, CBS agents and scheduler to reload DDS settings. Later network changes
require no process restarts. Save the old XML for rollback and stop the new
bridge before restoring direct DDS discovery.

## Validation

`network_smoke.py --binary /path/to/zenoh-bridge-ros2dds` uses two isolated Docker
network namespaces and ROS domain 177. It starts with no external interface,
attaches a network, removes it for 18 seconds, then reconnects. It asserts local
message progress during the outage, remote topics and Trigger service recovery,
and unchanged ROS process identities. Its resources are removed in `finally`.

`distributed_smoke.py --remote --count 2` in isolated domain 176 checks actual
CBS agents across boards with separate test bridges. Never inject synthetic
loops into the production domain. `wifi_observe.py` records real camera previews,
controller boot IDs and DPVO session IDs during a separately controlled outage;
it does not change Wi-Fi or camera state itself.

Upstream: https://github.com/eclipse-zenoh/zenoh-plugin-ros2dds/tree/1.10.1

## Verified on 2026-09-16

- Offline-boot/network-attachment/disconnect/reconnect Docker test passed, including
  remote Trigger service calls and unchanged ROS process identities.
- Real robot0 + robot1 CBS completed 40 rounds over the new bridges.
- Robot0 Wi-Fi was disconnected for 20 seconds, then reconnected automatically.
  The camera/tracker worker stayed running; processed frames advanced during the
  outage, with one controller boot ID and one DPVO session ID throughout.
- Fleet configuration tests: 8 passed. Multi-robot core regression suite: 18 passed.
- Robot2 received the same update in a separate rollout once it returned online.
  Its offline ROS startup, local controls, bridge heartbeats, and panel were checked
  on robot2; robot0 was not accessed or modified during that rollout.
