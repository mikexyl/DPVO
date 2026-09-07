# Optional TEASER++ backend

This extension calls the original MIT-SPARK TEASER++ C++ registration solver;
it is not a Python reimplementation. Source archives are downloaded to ignored
`build/teaser-sources/`, checked on every build, and kept with their original
copyright notices. No upstream solver source is patched.

| Component | Pinned revision | Archive SHA-256 | License |
| --- | --- | --- | --- |
| [TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus) | `52a9c52ee7d4c838c5e8a75458c33178be5bfb70` | `dc9ec391613b470b175267ca60c95099e74af818be3586cbb98ecd32cf806d59` | MIT, Massachusetts Institute of Technology |
| [PMC](https://github.com/jingnanshi/pmc) | `a2dfd612a501bca83c47206255dbbff619481f97` | `34a2c9716c903ccff3b04358470fb6e3cf1b97d0f2badc534d4c003613084851` | GPL-3.0-or-later, Ryan A. Rossi |
| [pybind11](https://github.com/pybind/pybind11) | `v2.13.6` | `e08cb87f4773da97fa7b5f035de8763abc656d87d5773e62f6da0587d1f0ec20` | BSD-3-Clause |

The optional `_teaser` binary statically links GPL PMC; it must not be described
as MIT-only. The corresponding notices and full GPL text are retained here.
DPVO's existing MIT code and CUDA extensions are built separately.

Local additions are the CMake build, pybind11 binding, input validation, uniform
coordinate normalization, timing, and the Python diagnostic interface. The build
compiles upstream `registration.cc` and `graph.cc` and PMC directly, leaving out
unneeded PLY I/O, FPFH and certification targets. It uses system Eigen and OpenMP,
not a camera library. Solver sources, scale TLS, CHAIN GNC-TLS rotation, and TLS
translation remain upstream implementations. No certification is claimed.

The binding releases the GIL and uses a fresh solver for every call because the
upstream solver mutates its rotation noise bound. It caps inputs at 4096 points,
caps requested workers at 32 (default 4), restores the caller's OpenMP limit,
and requests a two-second PMC search limit. That limit is not a hard timeout for
the complete solve. PMC's API does not expose whether optimality was proved or
its search timed out. Equal maximum cliques can differ across threaded calls.

Exact coincident source locations cause zero-length translation-invariant
measurements and division by zero in upstream scale TLS. The Python interface
keeps the first correspondence per exact source and target location, records
the original indices, and subsequently checks the resulting model against all
input correspondences. The native binding rejects coincident source input.

Build: `pixi run build-teaser`. Native memory audit:

```sh
pixi run python scripts/build_teaser.py --sanitize
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 build/teaser-sanitize/teaser_audit
```

The sanitizer target does not replace the normal Python extension. No Python,
NumPy, Torch, CUDA, OpenCV, or TensorRT dependency is installed or upgraded.
