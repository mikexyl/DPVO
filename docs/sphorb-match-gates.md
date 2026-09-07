# Office appearance-gate experiment, 2026-09-07

The preferred match visualization is the existing flattened sphere-panorama
style: query and candidate panoramas with correspondence lines, all appearance
matches visible, and a separate depth-check view. Keep this as the default for
future comparisons. Original source photographs remain available as an
optional diagnostic view.

The matcher now exposes optional gates through `match_features(..., gate=...)`
and `sphere_place_recognition.py --match-gate`. The strict baseline remains the
default. `--match-ratio` defaults to 0.75; `--patch-radius-degrees` defaults to
1 degree. No feature extraction, vocabulary training, learned models, pose
optimization, or loop insertion was rerun for this experiment.

`dpvo/sphere_match_gates.py` implements:

| Gate | Acceptance |
| --- | --- |
| `strict` | Exact mutual nearest neighbors; ratio in both directions |
| `forward-ratio` | Forward ratio only; mutual and reverse-ratio gates disabled |
| `mutual-forward-ratio` | Exact mutual nearest neighbors; forward ratio only |
| `patch-mutual` | Forward ratio; the reverse nearest neighbor may land within 1 degree of the original query feature |
| `patch-ratio-mutual` | Patch mutual cycle; compare against a descriptor outside the best candidate's angular patch |
| `distance-only` | One-way nearest neighbor; ratio and mutual gates disabled |

All gates retain the Hamming distance cap of 64. Patch distances use unit
bearings within each sphere and therefore work across seams, poles, and
arbitrary independent camera rotations. They do not compare panorama pixel
displacement or use poses/depth. A patch here is a neighborhood of feature
centers; this implementation does not perform pixel-patch correlation.

The patch ratio searches the closest 32 candidate descriptors for the nearest
competitor outside the best candidate's angular patch. If it cannot find one,
it rejects the match conservatively. This addresses ambiguity from nearby
detections across scales without assuming every nearby detection is the same
physical feature.

## Controlled comparison

`compare_sphere_match_gates.py` reuses all 115 frozen top-1 Office candidates
and their saved descriptors. It asserts exact strict-baseline correspondences
and depth inlier counts. All methods use the existing 500-trial Sim(3) RANSAC,
3% relative range threshold, six-match minimum, and scale range [0.25, 4].
The two focused pairs additionally use seeds 7, 17, 27, 37, and 47.

Relaxed matching can be many-to-one. Each report therefore includes a second
depth fit after retaining the lowest-Hamming match per candidate feature ID.
This removes exact target duplication but not necessarily multiple detections
of the same physical structure. Counts from the two fits must not be mixed.

| Gate / ratio | Appearance matches | Distinct target IDs | Depth inliers, all matches | Depth inliers after target deduplication |
| --- | ---: | ---: | ---: | ---: |
| Strict / 0.75 | 474 | 474 | 260 | 260 |
| Forward only / 0.75 | 2,483 | 1,611 | 924 | 634 |
| Exact mutual, forward ratio / 0.75 | 1,401 | 1,401 | 580 | 580 |
| Patch mutual / 0.75 | 1,771 | 1,474 | 814 | 622 |
| Patch ratio mutual / 0.75 | 2,889 | 2,289 | 1,508 | 1,052 |
| Patch ratio mutual / 0.80 | 4,688 | 3,703 | 1,975 | 1,317 |
| Distance only | 91,179 | 39,811 | 5,904 | 2,750 |

The much larger distance-only count is not evidence of higher precision.
Repeated office structures, duplicated sites, noisy depth, and low inlier
fractions can produce misleading consensus. These are diagnostics, not
ground-truth matching accuracy or confirmed loop counts.

For **822 to 525**:

| Gate / ratio | Matches | Distinct target IDs | Depth inliers, all matches | Depth inliers after target deduplication |
| --- | ---: | ---: | ---: | ---: |
| Strict / 0.75 | 3 | 3 | 0 (not run) | 0 (not run) |
| Forward only / 0.75 | 45 | 22 | 11 | 0 |
| Patch ratio mutual / 0.75 | 41 | 27 | 10 | 0 |
| Patch ratio mutual / 0.80 | 78 | 56 | 12 | 7 |

The seven distinct-target inliers at ratio 0.80 recur in four of five seeds;
one seed fails to find six-point consensus. The all-match fit is less reliable:
seed 7 chooses a scale of 1.23 supported by 12 matches on only four target IDs,
whereas the distinct-target fit estimates scale 1.94. The other all-match seeds
give 13–15 inliers at scales near 1.95. This pair remains weak geometrically.
The source-image viewer can display the independent-target check alongside
every appearance match, marking duplicate-target matches gray as untested.

The known **336 to 243** pair has stronger evidence: patch ratio mutual at
0.80 yields 453 appearance matches, 311 target IDs, and 260 depth inliers after
deduplication, compared with 89 / 89 / 81 strict. The extra correspondences
are useful candidates; defaults remain unchanged pending broader validation.

## Reproduce and inspect

```sh
pixi run python compare_sphere_match_gates.py \
  --retrieval saved_spheres/office_01_full_local_spheres_w30/sphorb_virtual_20260906_complete/retrieval \
  --output /tmp/office-gates-new
pixi run python compare_sphere_match_gates.py \
  --retrieval saved_spheres/office_01_full_local_spheres_w30/sphorb_virtual_20260906_complete/retrieval \
  --output /tmp/office-gates-080-new --ratio .80
pixi run python inspect_virtual_source_matches.py \
  --experiment saved_spheres/office_01_full_local_spheres_w30/sphorb_virtual_20260906_complete \
  --output /tmp/office-patch-822-new --query-anchor 822 \
  --match-gate patch-ratio-mutual --match-ratio .80 --depth-unique-targets
```

Actual outputs live under the completed virtual experiment:

- `match_gates_20260907/`: ratio-0.75 report, per-pair match arrays, and a
  tabbed comparison recording for 822 to 525.
- `match_gates_ratio080_20260907/`: ratio-0.80 experiment, with the strict
  baseline held at its original 0.75 setting.
- `match_gates_20260907/original_forward_822/`: all 45 forward-ratio matches
  in 24 original-image groups.
- `match_gates_20260907/original_patch080_unique_822/`: all 78 patch matches
  in 47 original-image groups, with the seven distinct-target depth inliers
  green, 49 tested outliers magenta, and 22 duplicate-target matches gray.

All five generated recordings pass `rerun rrd verify`. The full 32-test suite
passes, including baseline equivalence, descriptor ties, neighboring patch
competitors, seam/pole and independent-rotation invariance, target
deduplication, empty input, and descriptor-family rejection.

## Crossing-line audit and panorama display correction

For 822 to 525, the patch-ratio-0.80 model fit to all 78 matches accepts 12
matches on only four distinct targets. Its estimated rotation is 13.64
degrees, inconsistent with the saved 178.64-degree relative anchor rotation.
The fit to one match per target accepts seven distinct targets at 174.81
degrees, with 4.56 degrees of rotation error against the saved poses. Six of
these inliers have positive panorama-column displacement and one negative.
The saved poses are used only for this audit, never to guide RANSAC.

The panorama comparison previously colored the duplicate-sensitive all-match
fit. It now displays the distinct-target fit by default while keeping every
appearance line on the left. Duplicate targets excluded from the depth fit
are gray. The JSON reports still retain both fits for comparison.

The depth verifier operates solely on 3D points and supports either direction
of panorama lines. A new regression checks all 60 correct correspondences in
an opposite-heading synthetic pair, with more than 20 lines in each direction.
All survive, including after moving the panorama seam by a 137-degree frame
rotation. For the actual Office pair, a half-width seam shift preserves all
seven inlier IDs while changing their displayed directions from 6 right / 1
left to 1 right / 6 left. A one-sided-looking line bundle therefore does not
by itself establish a camera-heading bias.

The updated panorama recording and audit are in
`sphorb_virtual_20260906_complete/match_gates_crossing_audit_20260907/`:
`match_gates.rrd`, `crossing_audit.json`, and
`panorama_seam_invariance.png`. All 33 tests pass and the recording verifies.
