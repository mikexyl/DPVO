# SPHORB provenance and modifications

Upstream: <https://github.com/tdsuper/SPHORB>, revision
`e5f2ccfbb924935c3d4718c58c3c628dff0fe89f`. The `upstream/` directory preserves
the published source and notices unchanged. Its README states that SPHORB is
distributed under the GNU General Public License; it does not specify a version.
`GPL-3.0.txt` supplies a copy of that GPL version for reference, without changing
upstream's version statement. Copyright (C) 2015 Tianjin University, Qiang Zhao.

The GPL-derived native port and bindings in this directory are separate from
DPVO's MIT code and are **not covered by the root MIT license**. The optional
combined build must respect the upstream GPL terms. The Python environment is
not modified to install this backend.

Changes made for this port (September 2026):

* `kernels.hpp` extracts the published cross-section extension, intensity-centroid
  orientation, hexagonal smoothing kernel and all 256 binary comparisons. The
  computations and descriptor pattern are unchanged. All five sections belong
  to the connected geodesic grid; they are not perspective cube faces.
* `detector.cpp`, `detector.h`, `nonmax.cpp` retain the trained spherical FAST
  decision tree, score and hexagonal suppression. C++17 types replace `CvPoint`
  and ambiguous `byte`; allocations are checked; suppression indices are bounded.
  Production callers use RAII with `free` for upstream malloc/realloc results.
* `core.cpp` replaces upstream's globals, constructor/destructor, unchecked PFM
  loading and extraction orchestration. All 49 pinned assets (32,147,216 bytes)
  have compiled-in SHA-256 checks plus header/dimension/range checks. Paths are
  absolute. Immutable tables are shared between live instances with a weak cache;
  buffers are private to each instance and reused. Native calls on one instance
  serialize; independent levels use at most the requested worker count. System
  OpenCV's inner threading is set to one. Results merge in octave order; stable
  response ordering resolves ties and enforces the exact per-level quotas.
* Validity is alpha==255, independent of intensity and depth. Unknown samples
  are zeroed only for deterministic arithmetic; they cannot support accepted
  features. Every nonzero interpolation contribution is checked through area
  resizing and bilinear geodesic sampling. Validity follows the published
  cross-section extension. Unmapped padding remains invalid. FAST support is
  checked before suppression. Orientation's full radius-15 hexagon and all
  rotated descriptor samples, including the nonzero smoothing footprint, must
  be observed **before** quotas are applied. Unused quotas are not redistributed.
* The published tables encode panorama angular coordinates at pixel edges.
  The production sampler applies -0.5 in the resized panorama to honor DPVO's
  pixel-center convention, wraps longitude and clamps latitude at the poles.
  Original-resolution output uses independent width/height scales. For upstream
  rotated geodesic coordinates `(gx,gy,gz)`, the RDF bearing is normalized
  `(gy,-gz,gx)`. No bearing is reconstructed from rounded image coordinates.
* Feature sizes are `31*original_width/(5*cells[octave])` panorama pixels.
  Orientations retain upstream local geodesic tangent-grid degrees; they are not
  image-plane angles in the distorted panorama. Depth is sampled at the nearest
  panorama pixel center; nonpositive/nonfinite depths give NaN points and never
  affect appearance extraction.
* `bindings.cpp` releases the GIL for construction/extraction and returns owned
  NumPy arrays. The `upstream_sampling` switch is an audit-only native interface;
  it restores upstream table sampling, while keeping support checks. It is not
  exposed by the production `SphorbExtractor` interface or retrieval CLI.
* The optional virtual-input prototype exposes immutable grid bearings and
  `extract_grids`. It reuses the exact detector, orientation, smoothing, pattern,
  and boundary-extension code. Direct grids carry source IDs; the FAST,
  orientation and smoothed/rotated descriptor support must belong to one source.
  Grid addresses are returned for source-pixel provenance in this interface.
* `virtual_scene.cpp` and `virtual_bindings.cpp` add an owned triangle BVH and
  bounded parallel ray queries. They recover source UV with perspective-correct
  interpolation, estimate a source-image mip footprint from the ray/UV Jacobian,
  and expose per-source near-front visibility. Closest back-facing surfaces
  occlude but do not provide texture. No color interpolation between keyframes
  or learned optimization is performed. This is new prototype code, not part
  of published SPHORB.
* `reference/` is a minimally modernized, serial upstream implementation used
  only by the standalone audit. Changes: OpenCV4/C++17 names, explicit asset
  directory, checked reads, clamped final sampling row, correct deallocation,
  and preserved grid addresses instead of final panorama mapping. It uses the
  same allocation/bounds fixes in the detector and suppression. One reference
  instance is constructed per audit process, since its upstream global lifetime
  design is deliberately retained. It is never linked into the Python module.
* The standalone CMake build uses system OpenCV 4.5.4 and system C++/OpenSSL,
  plus checksum-pinned pybind11 2.13.6 headers. Pixi compiler/linker environment
  flags are cleared for this target. No DPVO CUDA build or dependency solve is
  required. The 32 MB assets and build outputs are ignored by existing rules.

Descriptor identity: `sphorb / e5f2ccf-mask-rdf-v1`. Cube ORB retains
`cube-orb / opencv-orb-wta2-v1`. Feature archives and vocabularies carry both
fields; untagged legacy vocabularies are only accepted as cube ORB.

Virtual-input identity: `sphorb-virtual / e5f2ccf-source-grid-v1`. It has a
separate vocabulary. The Python wrapper samples each visible source separately,
then suppresses nearby same-octave bearings across source observations before
applying the published per-octave budgets. This source-selection stage changes
the resulting features and is explicitly experimental.

See [the usage and verification guide](../../docs/sphorb.md) for commands and
the offline comparison artifacts.
