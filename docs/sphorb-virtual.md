# Virtual SPHORB prototype

The [corridor1 image-directory experiment](sphorb-corridor1-virtual.md) adds
source-stream validation and a completed virtual-SPHORB/TEASER++ comparison.

This optional offline path samples original keyframe images directly onto the
published SPHORB geodesic grids. It runs neither DPVO, DA3, SAM, nor Gaussian
optimization, and does not insert loop closures. The sphere is an anchor pose
and a set of directions. Preview panoramas are generated **after extraction**
and are used only for visualization.

## Inputs and geometry

`dpvo/virtual_sphere.py` reconstructs camera-frame depth points from the grouped
dense PLY and its final trajectory, then transforms them with the source poses
saved in each original sphere snapshot. Depth-grid addresses are recovered by
projection and nearest stride-grid rounding; residuals must be below 0.05 model
pixels (the saved Office grid has stride 2). Exact comparisons of decoded image
colors against every saved point's RGB independently validate video frame
indexing, calibration, and recovered grid addresses.

The source loader supports four pinhole intrinsics and follows the relevant
DPVO stream preprocessing. Video input uses the half-resize/crop convention:
for stride 5 and skip 0, processed timestamp `t` comes from zero-based raw frame
`5*(t+1)-1`. Image directories use sorted PNG/JPEG files, retain full resolution
before cropping, and select raw index `skip + t*stride`. This is the path used
by Loris corridor1-2. The inverse resize uses pixel centers; source UVs refer to
the full-resolution original image. Nonzero lens distortion is rejected by
this loader. These intrinsics describe individual source cameras, while sphere
directions and Sim(3) verification remain full 3D with no pinhole sphere model.

A source depth surface contains triangles only across fully observed 2x2
sample cells, rejecting cells with more than 5% relative depth variation.
Missing samples and depth-discontinuity cells are not bridged. Float32 PLY
serialization slightly perturbs recovered camera points; grid residuals and
color-validation results are saved in the source cache.

## Sampling and coherence

The native owned BVH intersects unit RDF rays. It finds the closest surface per
source and the nearest surface globally. A source may provide texture only
when its closest intersection is front-facing and within 3% of the global
nearest range. Back-facing surfaces still occlude. These are approximate
visibility checks on DA3 geometry, not proof of physical visibility.

Each eligible source is sampled independently. Perspective-correct triangle
interpolation returns original fractional UVs. A ray/UV Jacobian estimates the
largest local image footprint; area-filtered image mip levels and bilinear
sampling provide a conservative isotropic filter. Image borders without full
bilinear support are invalid. Original-image mip samples are not blended
across sources.

The unchanged spherical FAST, intensity-centroid orientation, Gaussian kernel,
and 256-bit pattern operate on each source's geodesic grids. All required
samples must have the same source and valid visibility. After detection,
features from different sources are sorted by response and suppressed within
1.5 median grid spacings at each octave before the published feature quotas
are applied. Ties favor newer sources and then deterministic native site order.

Virtual descriptors carry `sphorb-virtual / e5f2ccf-source-grid-v1`, distinct
from panorama-input SPHORB. Train a separate vocabulary using only the initial
training spheres; incompatible reuse is rejected. Existing temporal and
source-overlap exclusions, appearance thresholds, and depth checks are reused.

## Run

```sh
pixi run build-sphorb
pixi run python virtual_sphere_place_recognition.py \
  --spheres saved_spheres/office_01_full_local_spheres_w30 \
  --dense-map saved_dense_maps/office_01_full_local_spheres_w30.ply \
  --trajectory saved_trajectories/office_01_full_local_spheres_w30.txt \
  --video /data/scalemaster/Office_01/rgb.mp4 --calibration calib/office_01.txt \
  --source-cache saved_spheres/office_01_full_local_spheres_w30/virtual_source_cache_20260906 \
  --output /tmp/office-virtual-new --threads 4 --words 4096 --train-spheres 12
pixi run python inspect_sphere_matches.py \
  --retrieval /tmp/office-virtual-new/retrieval \
  --output /tmp/office-virtual-new/raw_matches
pixi run python inspect_virtual_source_matches.py \
  --experiment /tmp/office-virtual-new \
  --output /tmp/office-virtual-new/original_image_matches
```

The source cache stores lossless original images and validation records.
For an image sequence, replace `--video ...` with `--images ...`; both accept
`--source-stride` and `--source-skip` (`--video-stride` and `--video-skip` remain
aliases). Image cache provenance includes the sorted file listing fingerprint
and per-keyframe original file hashes. A regression test checks both source
types against the actual DPVO stream functions, including nonzero skip, stride,
crop, resize, exact point colors and recovered original-image coordinates.
`grids/<anchor>/source_<timestamp>_level_<octave>.npz` records every valid sampled
grid address, intensity, fractional source UV, mip level, radial depth, and
triangle index. Unlisted grid addresses are invalid. `provenance/<anchor>.npz`
maps accepted features to source frames, original pixels, and native grid
addresses. These artifacts permit exact replay without a panorama.

`virtual_input.json` separates image preparation/I/O, scene construction,
visibility traversal, source sampling, texture filtering, native feature
computation, provenance writes, and preview rendering. Three warmed passes
at anchor 336 verify identical features and provenance; these single-anchor
timings are not a sequence-wide warmed benchmark. The prototype prioritizes
auditable sampling and uses repeated CPU ray queries; it is much slower than
the panorama extractor.

## Office measurement, 2026-09-06

The completed experiment is in
`saved_spheres/office_01_full_local_spheres_w30/sphorb_virtual_20260906_complete`.
All three methods use the same 132 anchors, initial 12 training spheres,
separate 4,096-word vocabularies, 115 eligible queries, and 7,934 eligible
past-only pairs. Training and candidate eligibility were checked identical.

| Input | Mean features | Top-1 appearance / depth inliers | Top-3 appearance / depth inliers |
| --- | ---: | ---: | ---: |
| Full panorama | 2,311.7 | 353 / 142 | 868 / 287 |
| Half panorama | 2,039.3 | 347 / 116 | 846 / 214 |
| Virtual source images | 1,116.7 | 474 / 260 | 1,153 / 501 |

On the full-panorama baseline's fixed 115 top-1 pairs, virtual extraction
produces 423 appearance matches and 222 depth inliers, compared with 353 and
142. On anchor pair 336 to 243, the counts are 89 / 81 virtual, 50 / 48 full,
and 53 / 51 half. Angular coverage is narrower: mean occupied equal-area bins
falls from 40.9 / 72 (full) to 29.6 / 72 (virtual).

These are diagnostics, not ground-truth accuracy or confirmed loop counts.
The virtual path changes texture sampling, feature support, and depth lifting
(source surfaces instead of panorama depth samples), so the depth improvement
cannot be attributed solely to descriptors. No learned models were rerun and
no poses or loop constraints were changed.

Virtual extraction across the sequence has a 9.23 s median and 15.24 s p95 on
the first pass, excluding scene construction, source I/O, provenance writes,
and preview generation. Three warmed passes at anchor 336 give a 10.68 s
median: 1.38 s visibility traversal, 8.22 s repeated source ray queries,
0.55 s image sampling, and 0.30 s native SPHORB computation. This is an offline
CPU prototype; the earlier panorama runs have warmed sequence medians of
29.96 ms full and 24.07 ms half. These timing populations differ and are
explicitly retained in `comparison.json`.

The first optimization opportunity is to return and reuse all eligible
source intersections from the initial traversal instead of tracing those rays
again for each source. Its speedup has not been measured. GPU rasterization
would be a separate implementation.

`original_image_matches/source_image_matches.rrd` shows all 474 top-1
appearance matches grouped into 320 original-image pairs, with all 260 depth
inliers and all rejected matches distinguished. `raw_matches/sphere_matches.rrd`
also includes every one-way nearest neighbor before appearance filtering.
All six generated recordings (retrieval, two match views, three candidate
graphs) pass `rerun rrd verify`. The graphs use the 683 keyframes with saved
dense depth as their trajectory subset; every retrieval edge remains a
candidate.

All 25,567,252 saved point colors across 683 source frames matched the decoded
video exactly. The largest recovered-grid residual was 0.023324 model pixels.
Replaying sparse source grids for anchors 243 and 336 reproduced all 4,759
retained descriptors and their provenance exactly without image or panorama
reads. The combined regression suite passes all 28 tests.

### Depth-status correction, 2026-09-07

The initial match viewers labeled every non-inlier as a depth outlier, including
pairs for which verification was skipped. The verifier now reports an explicit
status; viewers render matches without a fitted depth model (or valid depth)
gray as **unverified**, reserving the outlier color for tested matches. This
does not change matching thresholds, fitted models, inlier counts, or rankings.
The updated regression suite has 29 passing tests, including exact 180-degree
rotation and the distinction between skipped checks and tested outliers.

For virtual pair 822 to 525, the saved anchor orientation differs by 178.64
degrees. There are 2,317 one-way nearest neighbors, 45 forward ratio/distance
matches, and only 3 matches after the reverse ratio and mutual tests. All three
have depth, but the verifier requires six, so **RANSAC never ran**. The old
display's three "rejected" matches were actually unverified. The synthetic
180-degree Sim(3) check retains all 60 correct correspondences.

Corrected full recordings are under `raw_matches_status_20260907/` and
`original_image_matches_status_20260907/` in the completed experiment;
`pair_822_525_status_20260907/` contains the isolated pair. Earlier recordings
are preserved as the original experiment artifacts.

The subsequent [appearance-gate experiment](sphorb-match-gates.md) compares
disabled mutual matching and angular patch gates on the frozen Office pairs,
with separate diagnostics for duplicate-target correspondences.

## Checks

```sh
pixi run python -m unittest test_virtual_sphere test_sphorb test_sphere_bow test_sphere_loop_graph -q
pixi run python scripts/build_sphorb.py --sanitize
LSAN_OPTIONS=suppressions="$PWD/native/sphorb/lsan.supp" \
  build/sphorb-sanitize/sphorb_audit "$PWD/models/sphorb"
```

Tests cover perspective-correct UVs, source visibility, back-face occlusion,
mesh holes and discontinuities, valid black pixels, invalid arrays/rays, empty
scenes, output ownership, source coherence, hidden-pixel invariance, and
deterministic one-/four-worker output. The native ASan/UBSan audit includes the
new scene and direct grids. The only leak suppression is the pre-existing
792-byte system TBB shutdown allocation. Panorama-input reference behavior
remains byte-identical for all 141,183 supported upstream descriptors.
