# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `dpvo/`. Neural-network and odometry logic is in modules such as `net.py`, `dpvo.py`, and `patchgraph.py`; custom PyTorch operators are under `dpvo/altcorr/`, `dpvo/fastba/`, and `dpvo/lietorch/` with C++/CUDA sources beside their Python wrappers. Top-level `demo.py`, `train.py`, and `evaluate_*.py` are the main entry points. Runtime presets and camera intrinsics belong in `config/` and `calib/`. `datasets/` contains checked-in ground truth, while `logs/` contains reference evaluation results. `Pangolin`, `DBoW2`, `DPViewer`, and `DPRetrieval` are optional native viewer or loop-closure components; avoid unrelated edits to bundled/submodule code.

## Build, Test, and Development Commands

- `git submodule update --init --recursive` initializes Pangolin and DBoW2 after cloning.
- `conda env create -f environment.yml && conda activate dpvo` creates the supported Python 3.10, PyTorch, and CUDA environment.
- `pip install .` builds and installs DPVO's CUDA extensions. Eigen 3.4.0 must first exist at `thirdparty/eigen-3.4.0` as described in `README.md`.
- `./download_models_and_data.sh` fetches pretrained weights and sample/training data (about 2 GB).
- `python demo.py --imagedir=<path> --calib=calib/euroc.txt --plot` runs a local sequence.
- `python evaluate_euroc.py --trials=5 --plot` exercises an evaluation pipeline; substitute another `evaluate_*.py` for its dataset.
- `python train.py --steps=240000 --lr=0.00008 --name=<run>` starts training and writes to `runs/`.

## Coding Style & Naming Conventions

Use four-space indentation and follow the surrounding PEP 8-style Python. Name functions, modules, and variables with `snake_case`, classes with `CapWords`, and configuration keys with `UPPER_SNAKE_CASE`. Keep tensor shape/device assumptions explicit and place CUDA/C++ changes with the owning operator. No repository-wide formatter is configured; keep diffs focused and do not reformat vendored code.

## Testing Guidelines

There is no general pytest suite or stated coverage threshold. Run `PYTHONPATH=dpvo python dpvo/lietorch/run_tests.py` for Lie-group forward and gradient checks; it requires the compiled extensions and a CUDA-capable setup. For pipeline changes, run the closest evaluation script on a representative sequence and report command, GPU/CUDA environment, and resulting metrics. Name new Python tests `test_*.py` and keep deterministic fixtures small.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects such as `Fix format...`, `Update...`, and `Add...`; follow that pattern and keep each commit scoped. Pull requests should explain motivation and behavior changes, link relevant issues, list validation commands/results, and call out dataset or model assumptions. Include trajectory plots or viewer screenshots for visual changes. Do not commit downloaded models, full datasets, `runs/`, or build artifacts already covered by `.gitignore`.
