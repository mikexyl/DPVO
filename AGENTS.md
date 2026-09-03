# Repository Guidelines

## Project Structure & Module Organization

`dpvo/` contains the Python visual-odometry/SLAM implementation and its CUDA/C++ extensions. Top-level scripts such as `demo.py`, `train.py`, and `evaluate_*.py` are the main entry points. Repository tests live in `tests/`; use names such as `test_pose_graph.py`. ROS 2 packages are under `ros2/`, while reproducible workstation and staged-experiment tooling lives in `deploy/blackwell_ros2/`. The `cbs/` submodule provides the C++ distributed Sim(3) optimizer and its tests. Camera parameters and runtime settings belong in `calib/` and `config/`. `DBoW2/`, `DPRetrieval/`, `DPViewer/`, and `Pangolin/` are external or supporting components; avoid unrelated edits there. The paper is maintained as the separate `paper/icra2027/` submodule.

## Build, Test, and Development Commands

- `git submodule update --init --recursive`: initialize CBS, Pangolin, DBoW2, and the paper.
- `pixi run build`: install DPVO editable and compile its CUDA extensions.
- `pixi run verify`: rebuild and confirm Torch, CUDA, and DPVO extension imports.
- `pixi run verify-multi-robot`: build retrieval dependencies and run the core multi-robot checks.
- `pixi run python -m unittest tests.test_pose_graph -v`: run one focused Python test module.
- `colcon build --base-paths ros2 --symlink-install`: build the ROS 2 packages after sourcing ROS Humble.
- `./deploy/blackwell_ros2/build_cbs.sh`: build CBS and run its Sim(3) tests. For an existing CBS build, use `ctest --test-dir build/cbs --output-on-failure`.
- `pixi run python demo.py --imagedir=<path> --calib=<file> --viewer=rerun`: run local inference.

## Coding Style & Naming Conventions

Use four spaces, `snake_case` functions/modules, and `PascalCase` classes in Python. Keep imports explicit and preserve type hints in newer orchestration code. CBS C++ follows `cbs/.clang-format`: Google style, two-space indentation, no tabs, and an 80-column target. Use lowercase, descriptive YAML and script names. Keep deployment defaults configurable through documented `DPVO_*` environment variables rather than hard-coded machine paths.

## Testing Guidelines

Add Python regression tests as `tests/test_<feature>.py`; CBS tests use `cbs/tests/test_<feature>.cpp`. Prefer deterministic synthetic graphs and temporary directories over checked-in outputs. Run the smallest relevant test first, then `verify-multi-robot` for transport or ROS-facing changes. No coverage threshold is enforced, but bug fixes should include a failing-then-passing regression test.

## Commit & Pull Request Guidelines

History uses short imperative subjects, for example `Add Rerun visualization and deployment setup`. Keep commits scoped and update submodule pointers deliberately. Pull requests should explain behavior and configuration changes, list commands run, link the issue or experiment, and include plots/screenshots for trajectory or visualization changes. Do not commit datasets, model weights, credentials, `results/`, build trees, or generated recordings.
