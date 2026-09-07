# Corridor1 virtual-sphere SPHORB and TEASER++, 2026-09-07

Completed all 53 saved OpenLORIS corridor1-2 spheres using original keyframe
images sampled directly onto SPHORB's geodesic grids. The saved panorama RGB
and depth rasters are not extraction inputs. Saved DA3 geometry still supplies
the visibility surfaces and radial depth. Preview panoramas are generated after
feature extraction, solely for display. No learned model, DPVO tracking, or
pose optimization was rerun.

Authoritative output:
`saved_spheres/loris_corridor1_2_full_keyframes/sphorb_virtual_20260907/`.

## Source camera and indexing

This sequence uses the calibrated **d400_color_optical_frame**, a pinhole source
camera with zero distortion, not either T265 fisheye camera. The individual
source cameras use their calibration; spherical sampling and Sim(3) estimation
operate on full 3D RDF directions, without a pinhole or fisheye sphere model.

The source is an image directory. DPVO's actual image reader uses full-size
848x480 images and raw index `5 * processed_timestamp`. Office's video path
uses half-size preprocessing and raw index `5*(timestamp+1)-1`. The virtual
loader now supports both conventions explicitly, retaining the existing Office
cache fingerprint and video behavior.

All 286 original keyframes and all **10,679,486** saved point colors matched
exactly after the corresponding DPVO preprocessing. The sorted filenames agree
with `color.txt`; per-file hashes, camera intrinsics, zero distortion and every
selected raw index were checked. Maximum recovered depth-grid error was
0.009301 model pixels. `source_validation.json` records these checks.

## Results

There are **4,929 virtual features**, mean **93 per sphere**. Anchors 499 and 534
have no fully supported features. The first 12 training spheres contain only
435 distinct descriptors, so the separate virtual vocabulary has **435 words**,
versus 4,096 for the earlier panorama vocabulary. No descriptors were duplicated
to inflate that count and no Office vocabulary was reused.

The same past-only, training-window and source-overlap rules leave 34 eligible
query spheres and 977 eligible pairs after excluding empty feature sets. The
earlier panorama run had 36 queries and 1,061 pairs. The masks were verified to
differ only because of the two empty virtual feature sets.

Using patch-ratio-mutual .80, one-degree patches and Hamming <=64 on each
virtual query's top-ranked candidate:

| Metric | Virtual-sphere result |
| --- | ---: |
| Appearance matches | 112 |
| Distinct-target correspondences | 85 |
| Pairs with at least six depth correspondences | 3 |
| RANSAC depth support | 6 on one pair |
| TEASER++ .03 depth support | 6 on one pair |

The only supported pair is **472 -> 307**: 10 appearance matches, seven distinct
targets, six residual inliers with either estimator. TEASER estimates scale
0.92879 and rotation 107.39 degrees. The saved relative orientation is 12.92
degrees, and the rotation error against that reference is **94.84 degrees**.
The reference poses were not used by the solver or acceptance gate. This large
disagreement makes the six-match consensus suspicious; it is not a confirmed
place match or loop closure.

TEASER noise bounds were .01, .02, .03, .05 and .10 times the median target
range. Only .03 and .05 produced the six-match consensus. All methods retained
the existing final 3%-of-individual-target-range residual gate and [.25,4] scale
range. Results and masks were stable over the three warmed verification passes.

Own-ranking comparisons are affected by the new vocabulary and features. For
a fixed-pair comparison, the old panorama run's exact 36 top-one endpoints were
also evaluated with the virtual features and identical .80 patch gate:

| Fixed original panorama pairs | Panorama features | Virtual features |
| --- | ---: | ---: |
| Appearance matches | 369 | 42 |
| Distinct-target correspondences | 353 | 38 |
| TEASER++ .03 depth support | 0 | 0 |

Thus this experiment does not reproduce the matching benefit seen on Office.
Virtual sampling still depends on the saved depth surfaces and full descriptor
support. These counts do not establish the ground-truth correctness of either
set of visual matches.

First-pass virtual extraction, excluding mesh construction, source I/O,
provenance writes and preview generation, took median 5.56 s and p95 7.47 s.
Three warmed passes at anchor 235 took median 6.10 s with identical outputs;
native detection was approximately .31 s of that time. At the .03 setting,
TEASER verification took median .472 ms when actually invoked, including the
wrapper and common residual gate. Only three pairs invoked it per pass, so
overall medians dominated by skipped checks are not meaningful speed comparisons.

## Inspect and reproduce

`teaser/review/teaser_comparison.rrd` is the preferred recording. It preserves
the panned panorama layout and displays every appearance match on the left,
with RANSAC/TEASER depth tabs on the right. Pair slider 0 is 472 -> 307; the next
two are the richest appearance pairs, 285 -> 145 and 280 -> 145. All 34 pairs
remain available. The review reuses the recorded models and masks exactly;
`teaser/review/display.json` records the display ordering.

Other artifacts include:

- `retrieval/`: separate vocabulary, tagged feature archives, retrieval matrix,
  report and sphere-feature recording.
- `features/`, `provenance/`, `grids/`: direct-source feature data and exact replay
  inputs. `previews/` contains display-only panoramas.
- `teaser/report.json`, `teaser/matches/`: solver sweep, timing and all masks.
- `comparison.json`, `comparison_*.png`: panorama-versus-virtual diagnostics.
- `fixed_panorama_patch_comparison.json`: identical-endpoint .80 comparison.
- `validation.json`: **39 passing tests**, 328 descriptors/provenance records
  replayed exactly, three verified RRDs, checked archive copies and input hashes.

```sh
pixi run python virtual_sphere_place_recognition.py \
  --spheres saved_spheres/loris_corridor1_2_full_keyframes \
  --dense-map saved_dense_maps/loris_corridor1_2_full_keyframes.ply \
  --trajectory saved_trajectories/loris_corridor1_2_full_keyframes.txt \
  --images /data/loris/corridor1-2_5-package/corridor1-2/color \
  --calibration calib/loris_corridor.txt --source-stride 5 --source-skip 0 \
  --source-cache saved_spheres/loris_corridor1_2_full_keyframes/virtual_source_cache_20260907 \
  --output /tmp/corridor1-virtual-new --threads 4 --words 4096 --train-spheres 12 \
  --warm-anchors 235
pixi run python compare_sphere_teaser.py \
  --retrieval /tmp/corridor1-virtual-new/retrieval \
  --output /tmp/corridor1-virtual-new/teaser --focus-anchors 472 285 280
```

The requested word budget is capped by the number of distinct training
descriptors. Output directories must be fresh. Defaults in the live pipeline,
Python/native dependency versions and upstream SPHORB/TEASER source remain
unchanged. No commits, pushes or loop constraints were made.
