"""Compare separate native agent processes against the reference CBS solver."""
import json
from pathlib import Path
import subprocess
import tempfile
import threading

import numpy as np
from scipy.spatial.transform import Rotation
from dpvo.loop_closure.centralized import CentralizedPgoResult, RobotMapConstraint, Sim3
from dpvo.loop_closure.pose_graph import build_keyframe_graph, write_json
from dpvo_multi_robot.cbs_agent import NativeAgent
from dpvo_multi_robot.cbs_pgo import _read_trajectory, _map_transforms


def synthetic_graph(count):
    robots = [f'robot{i}' for i in range(count)]
    paths = {r: [Sim3([j*.4, .1*j*j, 0.], np.eye(3), 1.) for j in range(4)] for r in robots}
    frames = {r: Sim3([i*.8, i*.2, 0.], Rotation.from_euler('z', i*.1).as_matrix(), 1.+i*.15)
              for i,r in enumerate(robots)}
    constraints = []
    for a,b in zip(robots, robots[1:]):
        for j in (0,2):
            query, match = paths[a][j], paths[b][j]
            global_query = frames[a].compose(query)
            global_match = frames[b].compose(match)
            constraints.append(RobotMapConstraint(query_robot=(a,'test'), match_robot=(b,'test'),
                query_pose=query, match_pose=match,
                query_to_match=global_match.inverse().compose(global_query), weight=1.,
                query_keyframe_id=j, match_keyframe_id=j))
    return build_keyframe_graph(constraints, CentralizedPgoResult({},True,0.,0.),
        paths, {r:'test' for r in robots}, robots[0], align_to_global=False), robots


def main():
    for count in (2,3):
        graph, robots = synthetic_graph(count)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(graph, root/'graph.json')
            subprocess.run(['/opt/cbs/run',f'--input_graph={root}/graph.json',
                f'--output_dir={root}', '--iterations=40', '--run_centralized=false',
                '--run_explicit_anchor_centralized=false', '--write_rerun_rrd=false', '--rerun_stream=false'], check=True,
                stdout=None, stderr=subprocess.STDOUT)
            expected = _map_transforms(graph, _read_trajectory(root/'cbs.csv'))
            agents = {}
            try:
                for robot in robots:
                    agents[robot] = NativeAgent('/opt/cbs/agent', graph.to_dict(), robot, threading.Event())
                assert len({a.process.pid for a in agents.values()}) == count
                for round_id in range(40):
                    snapshots = {r: dict(sender=r, **a.call(op='snapshot', anchor=round_id%40>=20))
                                 for r,a in agents.items()}
                    for robot, agent in agents.items():
                        agent.call(op='update', packets=[snapshots[p] for p in sorted(agent.ready['neighbors'])])
                        estimate = agent.call(op='preview')
                        assert np.isfinite(estimate['translation']).all()
                        assert estimate['scale'] > 0
                for robot, agent in agents.items():
                    result = agent.call(op='result')
                    reference = expected[(robot,'test')]
                    np.testing.assert_allclose(result['translation'], reference.translation, atol=1e-6, rtol=1e-6)
                    np.testing.assert_allclose(Rotation.from_quat(result['quaternion']).as_matrix(),reference.rotation,atol=1e-6)
                    np.testing.assert_allclose(result['scale'], reference.scale, atol=1e-6, rtol=1e-6)
                print(json.dumps(dict(agents=count, separate_processes=True, optimizer_instances_per_process=1,
                                      matches_reference=True, rounds=40)), flush=True)
            finally:
                for agent in agents.values(): agent.close()


if __name__ == '__main__':
    main()
