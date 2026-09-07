# Optional sphere-native SPHORB

The optional CPU backend ports the published spherical FAST, orientation and
binary descriptor to C++17/OpenCV 4.5.4. It operates on SPHORB's connected
geodesic grid. Cube-face ORB remains the default. This is an offline retrieval
experiment; it does not run DPVO/SAM/DA3 or insert loop closures.

The completed [53-sphere Loris evaluation](sphorb-loris-20260906.md) contains
measured timings, correspondence diagnostics, rotation results and artifact links.

The native source is GPL-derived, separately from DPVO's MIT code. Read
[provenance and modifications](../native/sphorb/MODIFICATIONS.md) before
distributing a combined build.

```sh
pixi run build-sphorb
pixi run python sphere_place_recognition.py \
  --spheres saved_spheres/loris_corridor1_2_full_keyframes \
  --extractor sphorb --output /tmp/loris-sphorb-new \
  --words 4096 --train-spheres 12 --benchmark-passes 3
```

The build downloads 49 checksum-verified lookup tables into `models/sphorb`
and pinned pybind11 2.13.6 headers into `build/`. It uses the existing Pixi
Python interpreter, system CMake/G++ and system OpenCV 4.5.4 development
libraries, without installing or changing Python packages. Python, NumPy,
Python OpenCV, Torch, CUDA and TensorRT versions remain unchanged. Network
access is needed only on the first download. Native imports are optional;
the cube ORB path does not import the extension.

Use `--downsample 2` to area-average the input panoramas to half width/height
before extraction. Every contributing pixel must be observed; unknown regions
stay transparent. Depth uses the nearest original pixel to each output center
(ties toward the lower index), with invalid depths kept as NaN. Feature UVs and
sizes refer to the reduced image, including in the raw-match exporter. The
published seven internal geodesic grids remain unchanged. Downsampling is timed
separately from file reads and extraction; warmed passes use reduced images
already in memory. Retrain the vocabulary on the same initial sphere window for
a resolution comparison.

```python
from dpvo.sphorb import SphorbExtractor

extract = SphorbExtractor(features=3000, levels=7, threshold=20, workers=4)
features = extract(bgra, radial_depth)
# descriptors: uint8[N,32]; uv: float32[N,2]; bearings/points: float32[N,3]
# responses, octaves, orientations, sizes; descriptor_family/version
print(extract.setup_timing, extract.last_timing)
```

Keep one extractor per stream to reuse its buffers. `tables=` accepts an
explicit directory; the default is anchored to the source tree, not the working
directory. Alpha 255 means observed, including black pixels. Any unknown pixel
in interpolation, FAST, orientation or descriptor/blur support rejects a
feature before the quota. Sparse spheres can therefore yield substantially
fewer than 3,000 features. Feature sizes use original panorama pixels; angles
are in the local hexagonal tangent grid. Bearings use DPVO RDF coordinates and
panorama coordinates use continuous pixel centers. NaN/Inf/nonpositive radial
depths produce NaN points; depth never changes descriptors or ranking.

`SphereFeatures.face_ids` and `.face_xy` are `None` for SPHORB and are omitted
from its NPZ files. Visualizations use octave colors. SPHORB vocabularies must
be trained separately: cross-family/version reuse and matching are rejected;
legacy untagged vocabularies remain cube-ORB-only.

### Virtual source-image input

The experimental [virtual-input driver](../virtual_sphere_place_recognition.py)
uses the saved sphere manifest for its anchor/window poses, the grouped dense
PLY and final trajectory to recover source depth samples, and the original
keyframe video for texture. Saved sphere PNGs and radial-depth rasters never
enter feature extraction. See [implementation and validation](sphorb-virtual.md).

To inspect matches from an existing run before depth filtering:

```sh
pixi run python inspect_sphere_matches.py \
  --retrieval saved_spheres/office_01_full_local_spheres_w30/sphorb_bow_4096_20260906 \
  --output /tmp/office-sphere-matches-new
```

This reuses saved descriptors and candidate rankings. The recording shows every
appearance match in orange before the depth check, alongside green depth inliers
and magenta non-inliers. Another tab shows all one-way nearest neighbors before
ratio, distance and mutual filtering. All lines are included, with separate
entities for toggling their visibility. Match indices, distances and inlier masks
are saved in NPZ files. `--query-anchor 336` exports just that query.

Reproduce the full comparison in a fresh directory:

```sh
MPLCONFIGDIR=/tmp/dpvo-sphorb-mpl pixi run python compare_sphere_extractors.py \
  --spheres saved_spheres/loris_corridor1_2_full_keyframes \
  --trajectory saved_trajectories/loris_corridor1_2_full_keyframes.txt \
  --keyframes rerun_recordings/loris_corridor1_2_full_keyframes.keyframes.json \
  --output /tmp/loris-sphere-comparison-new
```

The driver runs both backends sequentially, trains independent 4,096-word
vocabularies on the first 12 spheres, and retains the existing past-only,
training-overlap and source-keyframe exclusions, two-way .75 ratio / 64-bit
Hamming threshold and depth Sim(3) checks. It measures initial file reads
separately from extraction, then three memory-resident warmed passes. Native
SPHORB timing includes scheduling/core computation; wrapper timing includes
grayscale/validity preparation, array export and depth projection. Setup timing
includes checksum checks, table parsing and native construction. Cube ORB's
native time sums OpenCV API call durations (including per-call ORB construction);
its wrapper includes Python and NumPy computation. These different native
boundaries should not be compared as equivalent kernels. Output file
writes, vocabulary training and plotting are outside extraction timings.

Artifacts include every feature archive and overlay, 53 side-by-side overlays,
similarity matrices, candidate graphs (PNG/PDF/JSON), four verified Rerun
recordings, `comparison.json`, and `rotations.json`. The rotation suite uses
a continuous analytic texture rendered directly on the sphere, plus three
saved partial spheres. Six known rotations include identity, rear seam and
pole crossings. One-degree one-to-one repeatability and mutual-match accuracy
are computed directly from bearings, without DA3 geometry. All graph edges
remain unverified candidates, even when depth inliers exist.

Validation commands:

```sh
pixi run python -m unittest test_sphere_bow test_sphere_loop_graph test_sphorb -q
build/sphorb/sphorb_audit "$PWD/models/sphorb"
pixi run python scripts/build_sphorb.py --sanitize
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  LSAN_OPTIONS=suppressions="$PWD/native/sphorb/lsan.supp" \
  build/sphorb-sanitize/sphorb_audit "$PWD/models/sphorb"
```

The sanitizer executable includes all native code and the reference audit;
it does not replace the normal Python extension. LeakSanitizer must run without
ptrace supervision. The full-support reference test compares all supported
sites under upstream sampling, so intentional pixel-center changes do not
confound descriptor agreement. Strict support rejects unmapped grid padding
even for full panoramas; identity and known-rotation tests exercise the
production pixel-center path separately.

The unsuppressed audit on this host reports 792 bytes in three system TBB
shutdown allocations. `lsan.supp` filters that external library only; the
filtered audit passes with no SPHORB leaks, invalid accesses or undefined
behavior. Both raw and filtered logs are retained with the comparison results.
