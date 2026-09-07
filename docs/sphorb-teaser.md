# Office spherical Sim(3): TEASER++ comparison, 2026-09-07

The optional backend now calls the **upstream TEASER++ C++ library** through
pybind11. It estimates scale, a proper unrestricted 3D rotation, and translation.
There is **no pinhole, fisheye, essential-matrix, or panorama-displacement model**.
Input points are the existing RDF unit bearings multiplied by radial depth.
Negative Z, seam crossings, poles and opposite headings are valid inputs.

The experiment reused all 115 frozen top-one candidate pairs from the saved
Office virtual-sphere SPHORB run. It reused the patch-ratio-mutual appearance
gate at ratio .80, Hamming <=64 and one-degree patches: 4,688 appearance matches,
3,703 distinct-target correspondences. No extraction, vocabulary training,
model inference, ranking changes, live integration, or loop insertion occurred.

## Measured results

The representative recording uses a solver noise bound of .03 times median
target range. The final acceptance gate is unchanged: Euclidean residual less
than 3% of each target's range, scale in [.25,4], and sufficient support.
The absolute TEASER bound is not the same noise model as that final gate.

| Pair/set | Appearance / distinct targets | RANSAC depth support | TEASER++ depth support |
| --- | ---: | ---: | ---: |
| 822 -> 525 | 78 / 56 | 7 | 0 |
| 336 -> 243 | 453 / 311 | 260 | 263 |
| All 115 pairs, recorded run | 4,688 / 3,703 | 1,317 across 29 pairs | 1,234 across 22 pairs |

For **822 -> 525**, all five tested solver bounds (.01, .02, .03, .05 and .10
times median range) fail. TEASER estimates scale 19.82–20.25 and retains a clique
of only two correspondences. This is neither a usable Sim(3) nor evidence that
the visual matches are wrong. RANSAC's existing scale is 1.94. The stored sphere
headings differ by 178.64 degrees, and the input includes 16 rear-hemisphere
query points and 23 rear-hemisphere candidate points. None is removed for its
direction. The failure occurs in registration, not camera projection.

For **336 -> 243**, TEASER finds scale 1.000014 and 263 supporting matches. Its
rotation differs from the saved relative orientation by 1.71 degrees. Saved
poses are recorded only as an audit reference after fitting, never as a prior
or acceptance gate, and are not ground truth.

| Solver noise fraction | Recorded depth support | Supported pairs |
| --- | ---: | ---: |
| .01 | 1,064 | 20 |
| .02 | 1,145 | 22 |
| .03 | 1,234 | 22 |
| .05 | 1,201 | 19 |
| .10 | 1,134 | 16 |

PMC can select different equal cliques across threaded runs. At .03, the three
warmed passes totalled 1,240, 1,240 and 1,234 supporting matches; the report
records the affected pair and every repeat count. No claim of deterministic
output on ambiguous real data or certified optimal registration is made.
The recording shows the initial warm-up result consistently, with the exact
inlier mask saved in the corresponding NPZ. Depth support is diagnostic only;
all displayed edges remain candidates.

## Timing

One warm-up followed by three measured passes, four CPU workers, all feature
arrays and correspondence lists already loaded. No file I/O or visualization
is included in these measurements. The final timing run occurred after native
builds and tests completed.

| Operation | Warm median | Warm p95 | Samples |
| --- | ---: | ---: | ---: |
| Existing Python/NumPy RANSAC verification, all pairs | 27.31 ms | 31.72 ms | 345 |
| TEASER .03 verification, all pairs | .572 ms | 1.789 ms | 345 |
| TEASER .03 C++ solve, when called | .344 ms | 1.554 ms | 264 |
| TEASER .03 wrapper and common residual check, when called | .324 ms | .626 ms | 264 |
| TEASER .03 total verification, when native called | .672 ms | 2.083 ms | 264 |

The other 81 pair-passes had insufficient points and skipped the native solve.
Native import/setup was .219 ms in this process, cached feature/match loading
took .332 s, and panorama reading plus Rerun/PNG export took 7.41 s. These are
single setup/I/O observations, not cold-start benchmarks. Separate percentile
columns need not sum. This measures verification speed; SPHORB extraction speed
is unaffected. TEASER was faster here but did not improve overall depth support.

## Artifacts and reproduction

Authoritative output directory:

`saved_spheres/office_01_full_local_spheres_w30/sphorb_virtual_20260906_complete/teaser_20260907_verified/`

- `teaser_comparison.rrd`: all 115 pairs using the existing flattened-sphere
  panorama style. Left shows every appearance match; right has solver tabs.
  Pair slider 0 is 822 -> 525; slider 1 is 336 -> 243. Green is depth support,
  magenta a tested rejection, gray unverified. Failed models do not turn all
  visually convincing correspondences into rejected matches.
- `report.json`: pinned input locations, descriptor tag, correspondence hashes,
  every fitted model/status, noise bounds, repeat variability, timing and pose
  audit. No post-fit least-squares refinement is applied to TEASER outputs.
- `matches/*.npz`: all appearance matches, distinct-target inputs, and per-solver
  masks; focused PNGs preserve the same panorama layout.
- `validation.json` and accompanying logs: tests, import compatibility, native
  memory audit, RRD verification.

The earlier `teaser_20260907` and `teaser_20260907_final` directories are
intermediate runs. Use `teaser_20260907_verified` for the final measurements.

```sh
pixi run build-teaser
pixi run python compare_sphere_teaser.py \
  --retrieval saved_spheres/office_01_full_local_spheres_w30/sphorb_virtual_20260906_complete/retrieval \
  --gates saved_spheres/office_01_full_local_spheres_w30/sphorb_virtual_20260906_complete/match_gates_ratio080_20260907 \
  --focus-anchors 822 336 \
  --output saved_spheres/office_01_full_local_spheres_w30/sphorb_virtual_20260906_complete/teaser_new_run
```

Output must be a fresh directory. The standalone comparison leaves the existing
RANSAC default untouched. Python callers can use
`dpvo.teaser.estimate_sim3(source_Nx3, target_Nx3, noise_bound=absolute_bound)`
or `verify_teaser_geometry(query_features, candidate_features, matches)`.

## Validation and implementation boundaries

All **38 tests passed**, including five new TEASER tests covering known full
sphere Sim(3), a 180-degree turn, 80% synthetic outliers, independent sphere
coordinate frames, unused panorama coordinates, invalid/empty depth, coincident
locations, repeated calls, output ownership and one/four-worker recovery.
Existing cube-ORB/SPHORB/virtual-sphere/graph regression tests remain passing.
ASan, UBSan and LeakSanitizer passed the repeated native registration audit.
LeakSanitizer required running outside the sandbox's process restrictions.
The RRD passed `rerun rrd verify`, and the focused sphere overlay was inspected.

Imports passed alongside NumPy 2.2.6, Python OpenCV 4.11.0, Torch 2.3.1/CUDA 12.1,
TensorRT bindings 10.13.2.6, SPHORB and the existing DPVO CUDA extensions. Torch's
thread count was preserved after a TEASER call. No package versions changed.

See [source pins, licenses and build details](../native/teaser/THIRD_PARTY.md).
TEASER itself is MIT; its linked PMC dependency is GPL-3.0-or-later. This
optional extension is kept separate from DPVO's MIT CUDA build. Upstream
registration source is unchanged. The wrapper removes exact coincident locations
before fitting to avoid zero-length scale measurements, records the mapping,
and checks the fitted model against all distinct-target input matches afterward.

The experiment does not establish that DA3 depth is the sole cause of rejection,
or that a larger consensus is necessarily correct. TEASER's absolute-bound scale
estimation also failed on the difficult pair. It is therefore retained as an
optional diagnostic backend, not adopted as an improved default verifier.
