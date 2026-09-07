# Loris SPHORB port evaluation — 6 September 2026

SPHORB extracted features faster on this host. It did not improve the measured place-matching diagnostics, so cube ORB remains the default. No candidate has depth support under the retained checks.

The final artifacts are in [`sphorb_comparison_20260906_complete`](../saved_spheres/loris_corridor1_2_full_keyframes/sphorb_comparison_20260906_complete). Earlier exploratory runs are separate. The input consists of all 53 saved 1024×512 Loris spheres; separate 4,096-word vocabularies use the first 12. Both backends retain 36 eligible queries and 1,061 past-only, disjoint-source pairs.

| Metric | Cube ORB | SPHORB |
|---|---:|---:|
| Mean feature count | 1213.40 | 1245.28 |
| Mean occupied equal-area bins / 72 | 35.32 | 29.81 |
| Top-1 mutual matches, sum | 47.00 | 20.00 |
| Top-3 mutual matches, sum | 156.00 | 56.00 |
| Top-3 depth inliers, sum | 0.00 | 0.00 |
| Extraction, median / p95 ms | 113.32 / 132.52 | 24.61 / 35.58 |
| Native, median / p95 ms | 36.04 / 48.34 | 23.88 / 34.77 |
| Wrapper, median / p95 ms | 77.28 / 85.20 | 0.67 / 0.79 |
| Input file reads, median / p95 ms | 16.92 / 18.28 | 17.78 / 19.19 |

Warmed median extraction speedup: 4.61×. Each backend ran sequentially over three warmed passes (159 samples). Inputs were memory resident; output writes, vocabulary fitting, plotting and recording are outside these timings. Native SPHORB table/setup took 89.8 ms; total setup including extension import took 128.2 ms.

Host: Intel Core i7-13650HX, four requested CPU workers, no CPU affinity pinning. SPHORB uses system OpenCV 4.5.4 with one inner thread; cube ORB uses Python OpenCV 4.11.0 with four threads. Cube native timings sum OpenCV call durations; its wrapper includes NumPy projection and Python filtering. SPHORB native time covers its C++ core; wrapper time includes image preparation, array export and depth lookup. Percentiles of components do not add to percentiles of total time.

Across the 53 pairs appearing in both top-three lists, cube ORB produced 76 mutual matches versus 29 for SPHORB; neither produced depth inliers. These are counts of diagnostic correspondences, not measured place accuracy. There is no place ground truth, and candidates are not loop closures.

## Known spherical rotations

The suite includes six rotations on one full analytic sphere and three saved partial spheres (48 backend/case records). Full analytic texture is rendered directly from a continuous 3D field, avoiding seam/pole discontinuities. Saved images are rotated with strict observed-pixel validity. No depth is used. Identity yielded 100% repeatability and matching precision for every source/backend.

Analytic full-sphere results below exclude identity. Repeatability uses one-to-one angular correspondences within 1°, divided by the smaller feature count; precision uses the unchanged mutual .75 ratio / Hamming≤64 matches.

| Rotation | Cube repeatability | SPHORB repeatability | Cube precision | SPHORB precision |
|---|---:|---:|---:|---:|
| yaw_72 | 66.6% | 98.3% | 99.3% | 100.0% |
| seam_yaw_173 | 78.5% | 69.3% | 97.6% | 99.8% |
| pole_pitch_90 | 92.6% | 62.5% | 100.0% | 97.7% |
| roll_45 | 53.0% | 64.9% | 99.2% | 99.4% |
| oblique | 41.4% | 60.1% | 98.7% | 94.2% |

The results depend on rotation: 72° yaw aligns SPHORB grid sections; 90° pitch aligns cube views. The oblique case improves SPHORB repeatability while reducing its match precision. This suite does not establish general superiority. `rotations.json` includes per-source, seam (|longitude|≥150°) and pole (|latitude|≥60°) match counts, precision, errors and rotation matrices.

## Implementation validation

* 21 unit/regression tests passed, including cube ORB, coordinate round-trips, transparent-RGB invariance, black observed pixels, invalid depth, corrupt/missing tables, ownership, repeated instances, and single/multiple worker determinism.
* 141,183 supported full-sphere descriptors were byte-identical to the modernized upstream reference, with identical scores and angles. Its additional 3,345 sites depended on unmapped grid support and were rejected. This comparison restores upstream sampling to isolate kernels from the intentional half-pixel coordinate fix.
* ASan/UBSan passed. The unsuppressed leak audit reports 792 bytes in three system-TBB shutdown allocations; the audit passes with only `libtbb.so.2` suppressed. Logs retain both results.
* Imports passed alongside NumPy 2.2.6, Python OpenCV 4.11.0, Torch 2.3.1, CUDA 12.1, TensorRT bindings 10.13.2.6, and the existing DPVO extensions. The lockfile is unchanged.
* All 106 feature archives, both vocabulary tags, identical eligibility masks, 331-node/36-candidate graphs, and all four Rerun recordings were verified. Features are byte-identical before and after adding baseline timing instrumentation.

## Artifacts

* [Machine-readable comparison](../saved_spheres/loris_corridor1_2_full_keyframes/sphorb_comparison_20260906_complete/comparison.json) and [validation record](../saved_spheres/loris_corridor1_2_full_keyframes/sphorb_comparison_20260906_complete/validation.json).
* [Feature overlay example](../saved_spheres/loris_corridor1_2_full_keyframes/sphorb_comparison_20260906_complete/overlays/000082.png), [similarity matrices](../saved_spheres/loris_corridor1_2_full_keyframes/sphorb_comparison_20260906_complete/similarity_side_by_side.png), and [candidate graphs](../saved_spheres/loris_corridor1_2_full_keyframes/sphorb_comparison_20260906_complete/candidate_graphs_side_by_side.png).
* [Known-rotation report](../saved_spheres/loris_corridor1_2_full_keyframes/sphorb_comparison_20260906_complete/rotations.json).
* [Cube ORB recording](../saved_spheres/loris_corridor1_2_full_keyframes/sphorb_comparison_20260906_complete/cube-orb/sphere_retrieval.rrd) and [SPHORB recording](../saved_spheres/loris_corridor1_2_full_keyframes/sphorb_comparison_20260906_complete/sphorb/sphere_retrieval.rrd). Each backend also has `graph/loop_graph.rrd`, PNG/PDF/JSON graph exports, feature NPZs, overlays and pair montages.

Build/reproduction commands and licensing information are in [sphorb.md](sphorb.md). No learned models or live pipeline were rerun, and no commits or pushes were made.
