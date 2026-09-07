# Office: offline Sim3 pose graph from virtual-sphere matches

`optimize_sphere_pose_graph.py` adds the frozen Office virtual-SPHORB / TEASER++
measurements to a separate graph and optimizes camera-to-world Sim3 poses.
Original trajectories, depth maps, sphere caches, and retrieval results are
read-only inputs. It does not rerun extraction, registration, learned models,
or DPVO tracking, and it does not change DPVO's live loop-closure code.

## Graph and coordinates

The graph contains the 683 accepted depth keyframes saved in
`saved_dense_maps/office_01_full_local_spheres_w30.json`, including all 132
sphere anchors. The older Office run has no independent retained-keyframe
audit for frames without saved depth; these 683 nodes should not be described
as every keyframe ever retained internally by DPVO. Its 682 backbone edges
are relative poses between consecutive saved keyframes from the final TUM
trajectory. They are a pose-graph approximation, not the original patch factors.

The loop input is `teaser_20260907_verified/report.json`, method `teaser-0.03`:
22 of the 115 frozen top-one sphere pairs have at least six depth inliers.
All 22 enter the solve. The remaining 93 have no accepted TEASER measurement
and are listed with their reasons in the output report. In particular,
822 -> 525 is excluded because TEASER's scale is out of range; the separate
seven-inlier RANSAC hypothesis is not silently substituted.

For each node, `p_world = s_i R_i p_camera + t_i`. Stored TEASER measurements
map **query local RDF points to candidate local RDF points**. Therefore,
`Z_candidate_query = inverse(S_candidate) * S_query`. Sphere data are already
in the physical anchor camera's local coordinates, despite differences
between online snapshot world poses and final trajectory world poses. Do not
conjugate the measurements by those world-pose differences. The loader checks
every saved inlier mask by applying the measured Sim3 to its actual 3D points.
It also checks correspondence hashes, distinct past-only endpoints, and source
keyframe disjointness against the frozen retrieval reports.

The error is the full seven-dimensional Lie logarithm
`Log((inverse(S_candidate) * S_query) * inverse(Z_candidate_query))`, ordered
translation, rotation, log-scale. It uses spherical 3D points and unrestricted
proper rotations. There is no image projection or heading gate. The first
camera-to-world Sim3, including its scale, stays fixed to remove gauge freedom.

## Optimizer and weights

The existing LieTorch C++ backend supplies Sim3 group operations, exponential,
and logarithm. SciPy's sparse trust-region least-squares solver uses sparse
finite-difference Jacobians and LSMR in float64, with four CPU workers. The
variables are right Sim3 increments about the initial saved poses. Neither
the optional TEASER backend nor any environment package is rebuilt or changed.

Odometry has a quadratic loss. Each entire seven-dimensional loop residual
has a pseudo-Huber loss with transition delta 3 after whitening; the loss is
not applied independently to image coordinates or rotation components. Low
confidence scales the loss after robustification. Every candidate remains in
the graph, while inconsistent candidates exert less influence. Final robust
weights are diagnostics, not probabilities of loop correctness.

Defaults are explicit heuristics, not calibrated covariance estimates:

| Constraint | Translation sigma | Rotation sigma | Log-scale sigma |
| --- | --- | --- | --- |
| Consecutive keyframes | 0.1 × median consecutive step | 1 degree | 0.02 |
| Sphere loop | max(3% of median target range, 0.5 × median step) | 5 degrees | 0.1 |

Loop confidence is `min(depth_inliers/30, 1) * min(depth_inlier_ratio/0.5, 1)`.
This caps the influence of correlated descriptor matches rather than treating
hundreds of matches as independent pose measurements. Pseudo-Huber influence
is `1 / sqrt(1 + (normalized_residual_norm / 3)^2)`; no hard post-optimization
loop rejection or ground-truth-driven selection is performed.

## Reproduce and inspect

The saved run is
`saved_spheres/office_01_full_local_spheres_w30/sphorb_virtual_20260906_complete/sim3_pose_graph_20260907`.
It converged by relative cost tolerance after 185 function evaluations
(3,886 residual calls including sparse finite differences), in 79.43 seconds.

| Measurement | Before | After |
| --- | ---: | ---: |
| Robust objective | 2,455.7698 | 360.1023 |
| Median loop translation-log residual, DPVO units | 0.20852 | 0.03025 |
| Median loop rotation residual | 3.685 degrees | 2.828 degrees |
| Median absolute loop log-scale residual | 0.12734 | 0.07711 |

The median odometry residual after optimization is 0.000141 DPVO units in
translation, 0.078 degrees in rotation, and 0.00376 in log-scale. Four loop
candidates receive robust weights below 0.1: 605 -> 545, 804 -> 525,
899 -> 812, and 1060 -> 556. They remain in the objective with reduced influence.
The 336 -> 243 candidate retains weight 0.997; 1156 -> 243 retains 0.875.

Absolute camera-to-world scales range from 0.89194 to 7.27319 (median 3.18232).
Median position change is 0.77964 DPVO units; maximum is 4.95487. These large
changes reflect the selected loop constraints and heuristic weights. The
objective alone does not establish improved trajectory accuracy; the subsequent
reference comparison below evaluates the frozen poses without another solve.

```bash
pixi run python optimize_sphere_pose_graph.py \
  --trajectory saved_trajectories/office_01_full_local_spheres_w30.txt \
  --keyframes saved_dense_maps/office_01_full_local_spheres_w30.json \
  --spheres saved_spheres/office_01_full_local_spheres_w30/manifest.json \
  --loops saved_spheres/office_01_full_local_spheres_w30/sphorb_virtual_20260906_complete/teaser_20260907_verified/report.json \
  --output saved_spheres/office_01_full_local_spheres_w30/sphorb_virtual_20260906_complete/sim3_pose_graph_new

pixi run python -m unittest test_sphere_pose_graph test_teaser test_sphere_bow \
  test_sphere_match_gates test_sphere_loop_graph test_virtual_sphere -v
```

The output directory must be fresh. It contains:

- `graph.npz`: timestamped initial and optimized poses, edges, measurements,
  noise scales, loop flags, and confidence; sufficient to replay optimization.
- `initial_keyframes_sim3.txt`, `optimized_keyframes_sim3.txt`: timestamp,
  translation xyz, quaternion xyzw, and camera-to-world scale. Companion TUM
  files contain the translations and orientations; TUM cannot store scale.
- `report.json`, `edges.json`: provenance hashes, solver termination, objective,
  per-edge before/after errors and weights, and excluded-candidate reasons.
- `pose_graph.png` / `.pdf`: original, optimized, and overlaid X-Z trajectories.
- `pose_graph.rrd`: original and optimized 3D graphs, plots, and all appearance
  matches for each inserted candidate using the existing panned sphere style.
  The strongest appearance pair, 336 -> 243, is first in the loop tab. Selecting
  a loop highlights its graph edge. Green match lines are the original TEASER
  depth support, not newly estimated PGO inliers.
- `optimization.png`: per-keyframe scales and objective progress.

Tests independently check the query-to-candidate transform direction, Sim3
scale, near-180-degree rotation, rear-hemisphere points, invariance under a
common world Sim3, exact gauge fixing, input/output ownership, repeated and
single-/multi-worker calls, synthetic scale-drift correction, robust handling
of a deliberately false loop, and rejection of disconnected/corrupt graphs.

## Refined reference overlay

The `ground_truth_20260907/` subdirectory adds the local
`/data/scalemaster/Office_01/optimized_odometry.csv`. ScaleMaster documents this
as its [SE3-refined ARKit reference trajectory](https://github.com/JooHyoSeok/ScaleMaster-Dataset#-pose-refinement-pipeline),
using manually verified loop closures, metric DA3 relative poses, and GTSAM.
It is a dataset-provided reference, not independent motion-capture ground truth.
The dataset positions are meters in an ARKit Y-up world; only camera positions
are evaluated. No orientation-error metric or camera projection is used here.

All 683 graph nodes are matched by the CSV's explicit `frame` column using
`raw_frame = (DPVO_timestamp + 1) * 5 - 1`, with skip zero. This matches DPVO's
actual video reader and every validated original-image cache record. The CSV
and video have 6,010 frames; the matched range is raw frame 39 through 6,004.
There is no row-number assumption, nearest-timestamp match, missing-node
exclusion, interpolation, or offset fitted from trajectory similarity.

| Trajectory | Position ATE RMSE | Median error | p95 error |
| --- | ---: | ---: | ---: |
| Original DPVO | 5.838 m | 2.396 m | 15.985 m |
| Sphere-loop Sim3 PGO | 3.415 m | 2.172 m | 6.956 m |

Each trajectory is independently aligned to the same 683 reference positions
with one global Umeyama Sim3, using the existing `evo` implementation. All
matched keyframes receive equal weight, with no piecewise or local fitting.
The fitted global scales are 4.487652 and 3.960879, respectively. A separate
common-alignment tab applies the original trajectory's fitted transform to both
trajectories; the optimized RMSE under that shared transform is 4.252 m.
Reference data are used only for display and evaluation, never as graph factors
or to retune the optimizer, weights, loop selection, or measurements.

```bash
pixi run python show_sphere_pose_graph_reference.py \
  --graph saved_spheres/office_01_full_local_spheres_w30/sphorb_virtual_20260906_complete/sim3_pose_graph_20260907 \
  --reference /data/scalemaster/Office_01/optimized_odometry.csv \
  --source-cache saved_spheres/office_01_full_local_spheres_w30/virtual_source_cache_20260906 \
  --output saved_spheres/office_01_full_local_spheres_w30/sphorb_virtual_20260906_complete/sim3_pose_graph_20260907/ground_truth_new
```

`pose_graph.rrd` opens on reference comparisons: green reference, gray original,
blue optimized. The original graph and panned sphere-match tabs are retained.
`reference_comparison.png` / `.pdf`, `reference.npz`, `matched_reference.tum`,
and `reference_report.json` preserve the visual comparison, exact associations,
alignment transforms, errors, and input hashes. Eight focused tests cover the
reference association/alignment and Sim3 solver; the RRD is schema-verified.

Lower graph cost indicates better agreement with the chosen measurements and
weights. It does not establish lower ground-truth trajectory error. Optimizing
poses does not regenerate the existing spherical textures or deform the saved
dense point cloud; their original recordings retain their original geometry.
