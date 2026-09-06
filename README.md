# Deep Patch Visual Odometry/SLAM
This repository contains the source code for our papers:

[Deep Patch Visual Odometry](https://arxiv.org/pdf/2208.04726.pdf)<br/>
Zachary Teed<sup>\*</sup>, Lahav Lipson<sup>\*</sup>, Jia Deng <sub></sub><br/>
[Deep Patch Visual SLAM](http://arxiv.org/pdf/2408.01654)<br/>
Lahav Lipson, Zachary Teed, Jia Deng<br/>
<a target="_blank" href="https://colab.research.google.com/drive/1VSFGNB7YCveqKF7XNz4RlV9EnfQA3fhQ?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a><a target="_blank" href="https://github.com/princeton-vl/DPVO_Docker">
  <img src="https://img.shields.io/badge/Docker-grey?logo=Docker" alt="Open In Colab"/>
</a>

[<img src="https://i.imgur.com/6ZQPbR1.png?1" width="600">](https://www.youtube.com/watch?v=e5wanf71YFs)

```
@article{teed2023deep,
   title={Deep Patch Visual Odometry},
   author={Teed, Zachary and Lipson, Lahav and Deng, Jia},
   journal={Advances in Neural Information Processing Systems},
   year={2023}
 }
```
```
@inproceedings{lipson2024deep,
    author={Lipson, Lahav and Teed, Zachary and Deng, Jia},
    title={{Deep Patch Visual SLAM}},
    booktitle={European Conference on Computer Vision},
    year={2024}
}
```
## Setup and Installation
The code was tested on Ubuntu 20/22 and Cuda 11/12.</br>

Clone the repo
```
git clone https://github.com/princeton-vl/DPVO.git --recursive
cd DPVO
```
Create and activate the dpvo anaconda environment
```
conda env create -f environment.yml
conda activate dpvo
```

Next install the DPVO package
```bash
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty

# install DPVO
pip install .

# download models and data (~2GB)
./download_models_and_data.sh
```


### Recommended - Install the Pangolin Viewer
Note: You will need to have CUDA 11 and CuDNN installed on your system.

1. Step 1: Install Pangolin (need the custom version included with the repo)
```
./Pangolin/scripts/install_prerequisites.sh recommended
mkdir Pangolin/build && cd Pangolin/build
cmake ..
make -j8
sudo make install
cd ../..
```

2. Step 2: Install the viewer
```bash
pip install ./DPViewer
```

For installation issues, our [Docker Image](https://github.com/princeton-vl/DPVO_Docker) supports the visualizer.

### Classical Backend (optional)

We provide a classical backend for closing very large loops, which requires extra installation.

Step 1. Install the OpenCV C++ API. On Ubuntu, you can use
```bash
sudo apt-get install -y libopencv-dev
```
Step 2. Install DBoW2
```bash
cd DBoW2
mkdir -p build && cd build
cmake .. # tested with cmake 3.22.1 and gcc/cc 11.4.0 on Ubuntu
make # tested with GNU Make 4.3
sudo make install
cd ../..
```

Step 3. Install the image retrieval
```bash
pip install ./DPRetrieval
```

## Demos
DPVO can be run on any video or image directory with a single command. The
Pangolin backend requires DPViewer; Rerun is available as an alternative. You
can also save completed reconstructions and view them in COLMAP. The pretrained
models can be downloaded from google drive
[models.zip](https://drive.google.com/file/d/1dRqftpImtHbbIPNBIseCv9EvrlHEnjhX/view?usp=sharing)
if you have not already run the download script.


```bash
python demo.py \
    --imagedir=<path to image directory or video> \
    --calib=<path to calibration file> \
    --viewer=rerun \
    --plot \
    --save_ply \
    --save_trajectory \
    --save_colmap
```

Use `--viewer=rerun` for Rerun, `--viewer=pangolin` for Pangolin, or `--viz`
as the backwards-compatible Pangolin flag.

Rerun can also write a recording without opening a window. This is useful on
remote or headless machines:

```bash
python demo.py --imagedir=<path> --calib=<path> \
    --viewer=rerun --rerun-save=rerun_recordings/result.rrd
pixi run rerun rerun_recordings/result.rrd
```

### YOLO26 TensorRT overlays

The Pixi environment includes Ultralytics and TensorRT 10.13 Python bindings.
Export the engine on the GPU that will run it (TensorRT engines are specific to
the TensorRT version and target GPU):

```bash
pixi run python export_yolo26_engine.py --model yolo26n-seg.pt
```

Run instance segmentation on every stride-selected DPVO frame. Rerun overlays
the class-colored masks and confidence-labeled boxes on the camera image:

```bash
pixi run python demo.py \
    --imagedir=/data/scalemaster/Office_01/rgb.mp4 \
    --calib=calib/office_01.txt \
    --stride=5 \
    --viewer=rerun \
    --yolo-model=yolo26n-seg.engine \
    --yolo-task=segment \
    --scene-graph
```

With `--scene-graph`, YOLO mask instances are associated with DPVO patch
landmarks to form persistent object nodes. Rerun shows both a top-down graph
view and the object graph in 3D. The graph is also written to
`saved_scene_graphs/<name>.json`; use `--scene-graph-output` to choose another
path. Positions and relation distances use DPVO's monocular (scale-ambiguous)
world coordinates.

For open-vocabulary instance segmentation, bake text prompts into a YOLOE-26
TensorRT engine before running the same command:

```bash
pixi run python export_yolo26_engine.py \
    --model yoloe-26n-seg.pt \
    --classes "office chair" desk "computer monitor" laptop keyboard \
              "computer mouse" "filing cabinet" bookshelf "potted plant" \
              sofa "coffee table" door whiteboard "trash bin" printer
```

The exported engine retains those prompted class names and works with
`--yolo-model=yoloe-26n-seg.engine --yolo-task=segment`.

### SAM 2.1 Hiera Tiny masks (without YOLO or DA3)

SAM is an alternative **class-agnostic** region segmenter. The installed
Ultralytics SAM2 runtime loads Meta's official Hiera Tiny checkpoint with strict
weight matching; no PyTorch upgrade or new dependency is needed. The optimized
path exports **both the image encoder and prompt/mask decoder to TensorRT**.
It runs on every stride-selected DPVO frame, with no stale-mask reuse or reduced
prompt density. A `.pt` checkpoint selects the slower FP32 reference; a bundle
`.json` selects TensorRT, with no PyTorch model loaded at inference time.

```bash
mkdir -p models
curl -L --fail -o models/sam2.1_hiera_tiny.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt

# Build on the target GPU. Keep both engines and their manifest together.
pixi run python export_sam2_engine.py --batch-size=8 --fp16-io \
    --output-dir=models/sam21-tiny-trt-fp16io

# Keep experiments to the first 30 seconds unless a full sequence is wanted.
ffmpeg -n -i /data/scalemaster/Office_01/rgb.mp4 -t 30 -an \
    -c:v copy /tmp/office_01_30s.mp4
pixi run python demo.py \
    --imagedir=/tmp/office_01_30s.mp4 --calib=calib/office_01.txt \
    --stride=5 --viewer=rerun \
    --sam-model=models/sam21-tiny-trt-fp16io/sam2.json --sam-points-per-side=16 \
    --rerun-save=rerun_recordings/office_01_30s_sam21_tiny_trt.rrd \
    --name=office_01_30s_sam21_tiny_trt --save_trajectory
pixi run rerun rerun_recordings/office_01_30s_sam21_tiny_trt.rrd

# Six-viewpoint accuracy and timing comparison against FP32 PyTorch.
pixi run python benchmark_sam2.py \
    --model=models/sam21-tiny-trt-fp16io/sam2.json
```

Omitting `--da3-engine` disables depth inference/dense mapping. `--sam-model`
and `--yolo-model` are mutually exclusive. SAM regions do not populate the
semantic scene graph.

Rerun overlays colored region masks on RGB and logs mask count, covered pixel
fraction, and segmentation wall time under `segmentation/`. Region IDs/colors
are **frame-local**, not tracked identities or semantic labels; SAM scores are
predicted mask IoU, not class confidence. Large surfaces are painted first and
smaller overlapping regions take precedence. Unassigned pixels stay transparent.

The default prompt grid is 32×32; `--sam-points-per-side=16` is a faster option
with fewer small regions. TensorRT's prompt batch is fixed at export time; the
grid can change without rebuilding, and a partial last batch is padded then
discarded. `--sam-points-per-batch` applies only to the PyTorch reference.
`--sam-min-mask-area` defaults to 100 original-image pixels, and quality
filters are `--sam-pred-iou-thresh=0.8 --sam-stability-thresh=0.92`. Statistics and
a final-frame overlay PNG are saved to `saved_segmentations/<name>.json`/`.png`
(override the JSON path with `--sam-output`). Test mask geometry without a GPU:
`pixi run python -m unittest test_sam_segmenter -v`.

TensorRT keeps encoder features on the GPU and shares their buffers directly
with the decoder. CUDA graphs replay the encoder and all prompt batches with
fixed allocations; full-resolution stability filtering, box NMS, and mask
resizing run on the GPU before transfer to CPU/Rerun. The two-engine split also
follows [NVIDIA's SAM2 deployment architecture](https://github.com/NVIDIA/DeepStream/tree/main/tools/sam2-onnx-tensorrt),
but this image-only path does not export the unused video-memory networks.

On the RTX 4070 Laptop GPU, six sampled Office_01 frames with 256 prompts per
frame measured **222 ms mean end-to-end** (232 ms p95) versus **822 ms** for
FP32 PyTorch, a **3.7× speedup**. TensorRT neural inference alone was 166 ms;
end-to-end includes preparation, filtering, CPU transfer, and mask composition,
but excludes DPVO and Rerun logging. Corresponding high-quality raw mask
hypotheses had 99.88% mean IoU with FP32; this measures agreement, not semantic
accuracy. Threshold/NMS boundary effects can add or remove a region. See
`saved_segmentations/sam21_tensorrt_fp16io_benchmark.json` for per-frame results.
Startup/engine loading and warm-up are excluded from this benchmark.

### Stateful SAM 2.1 video masks

`--sam-video` treats the stride-selected DPVO frames as one ordered video stream.
It calls SAM2's actual `track_step` with learned temporal memory, memory encoding,
and object pointers, rather than generating independent masks or just matching
their boxes. RGB, masks, and DPVO poses stay on the same `frame` timeline in Rerun.

```bash
pixi run python demo.py \
    --imagedir=/tmp/office_01_30s.mp4 --calib=calib/office_01.txt \
    --stride=5 --viewer=rerun \
    --sam-model=models/sam21-tiny-trt-fp16io/sam2.json --sam-points-per-side=16 \
    --sam-video --sam-video-max-tracks=8 --sam-video-memory=3 \
    --sam-video-refresh=10 \
    --rerun-save=rerun_recordings/office_01_30s_sam21_video_refresh10.rrd \
    --name=office_01_30s_sam21_video_refresh10 --save_trajectory
pixi run rerun rerun_recordings/office_01_30s_sam21_video_refresh10.rrd
```

This is a **hybrid video backend**, not a fully TensorRT video export:

- TensorRT FP16: image encoder, plus automatic prompt-grid seeding on refreshes.
- PyTorch BF16: temporal attention, tracking mask heads, and memory encoder.
- No YOLO or DA3; the matching official `.pt` checkpoint must still be available
  at the path recorded in the TensorRT manifest. Its SHA-256 is verified.

SAM video segmentation needs prompts for the regions to follow. Here those
prompts are automatic: choose up to eight large, high-quality, nonredundant SAM
regions, then propagate them through the following frames. The cap controls GPU
cost and means video mode does **not** preserve the full per-image proposal set.
The default three memory slots include the prompt frame; use
`--sam-video-memory=7` for the original seven-slot context, at higher cost.
At most 16 recent object-pointer records plus the prompt record are retained.

Every 10 processed frames (about 1.7 input-video seconds at 30 FPS/stride 5),
automatic discovery re-prompts the tracker to cover newly visible regions. IDs
are retained through a refresh only for one-to-one same-frame mask IoU matches
above 0.3; other proposals receive new IDs. This explicitly resets the learned
memory on refresh and is not long-term re-identification. Use
`--sam-video-refresh=0` to seed once and propagate uninterrupted, without
automatically discovering new regions. Lost/occluded masks are hidden, and
their IDs/colors remain stable while the tracks remain active.

Rerun shows stable track colors and logs `active_tracks`, `memory_records`, and
`refresh` alongside mask count/coverage/time. Video scores are object-presence
estimates, not predicted mask IoU or semantic class confidence. The JSON report
records visible IDs on each frame. Video mode is opt-in because temporal
consistency does not automatically imply lower latency than the fully TensorRT
image-only path. Run tests with:
`pixi run python -m unittest test_sam_video test_sam_segmenter -v`.

Office_01 first-30-second trial on the RTX 4070 Laptop GPU, stride 5, 180
processed frames, DA3/YOLO disabled:

| Mode | Mean segmentation time | Mean visible regions | Mean mask coverage |
| --- | ---: | ---: | ---: |
| Image-only TensorRT FP16 | 234 ms | 31.6 | 75.2% |
| Video, 8 tracks / 3 memories, refresh every 30 frames | 225 ms | 5.7 | 51.6% |
| Video, 8 tracks / 3 memories, refresh every 10 frames | 222 ms | 7.3 | 62.9% |

The 10-frame refresh trial averaged 190 ms on propagation frames and 516 ms
on discovery/refresh frames, with no empty frames and at most 10 memory records.
These are segmentation wall times including postprocessing, not total DPVO
pipeline times. Coverage measures assigned pixels, not segmentation accuracy;
tracking can drift, and the video cap deliberately follows fewer regions.
Video tracking is not a substantial speed improvement over the image-only
TensorRT path in this experiment. Reports and per-frame IDs are in
`saved_segmentations/office_01_30s_sam21_video_refresh10.json` and
`saved_segmentations/office_01_30s_sam21_video.json`.

### Two-view Depth Anything 3 dense mapping

The optional DA3 path runs depth on consecutive retained DPVO keyframes,
robustly aligns each two-view prediction to DPVO's inverse-depth patches, and
backprojects confidence-filtered RGB points into the DPVO world frame. Rerun
shows the aligned depth image and the accumulating dense map. The final map is
saved as a binary PLY with alignment statistics in an adjacent JSON file.

Fetch the Apache-2.0 DA3-SMALL source and checkpoint used by the exporter:

```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git \
    thirdparty/depth-anything-3
git -C thirdparty/depth-anything-3 checkout \
    3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
pixi run hf download depth-anything/DA3-SMALL \
    --revision e08cab65ca0ec38e7826075418411ab90cab4da3 \
    --local-dir models/DA3-SMALL
```

Export a fixed two-view, 378x504 TensorRT engine. FP32 is the default because
FP16 can noticeably distort the relative-depth output on some TensorRT/GPU
combinations.

```bash
pixi run python export_da3_engine.py \
    --output da3-small-2view-378x504.engine
```

Run the Office_01 video at stride 5 without passing any YOLO arguments, which
keeps the segmentation branch disabled:

```bash
pixi run python demo.py \
    --imagedir=/data/scalemaster/Office_01/rgb.mp4 \
    --calib=calib/office_01.txt \
    --network=dpvo.pth \
    --stride=5 \
    --viewer=rerun \
    --da3-engine=da3-small-2view-378x504.engine \
    --dense-map-stride=7 \
    --name=office_01_da3_dense
```

DA3 depth is relative, so the PLY remains in DPVO's monocular,
scale-ambiguous coordinate system. `--dense-map-stride` controls point density,
`--dense-map-max-error` rejects poorly aligned startup/keyframe pairs, and
`--dense-map-output` overrides the default `saved_dense_maps/<name>.ply` path.

#### Color DA3 points with SAM video regions

Passing both `--da3-engine` and `--sam-model` automatically adds a SAM-colored
dense map. Use `--sam-video` for temporally tracked region IDs:

```bash
pixi run python demo.py \
    --imagedir=/tmp/office_01_30s.mp4 --calib=calib/office_01.txt \
    --stride=5 --viewer=rerun \
    --da3-engine=da3-small-2view-378x504.engine --dense-map-stride=7 \
    --sam-model=models/sam21-tiny-trt-fp16io/sam2.json --sam-points-per-side=16 \
    --sam-video --sam-video-max-tracks=8 --sam-video-memory=3 --sam-video-refresh=10 \
    --rerun-save=rerun_recordings/office_01_30s_da3_sam_video.rrd \
    --name=office_01_30s_da3_sam_video --save_trajectory
pixi run rerun rerun_recordings/office_01_30s_da3_sam_video.rrd
```

Each frame's integer mask atlas is cached on CPU at DA3 resolution using
center-aligned nearest-neighbor sampling. Delayed keyframe depth uses the mask
with the **same timestamp**, and labels undergo the same confidence/depth
filtering as the 3D points. SAM colors match the camera overlay; unassigned
points are gray (region ID 0). RGB colors and geometry remain unchanged.

Rerun opens a `SAM-colored Dense Map` tab, with an `RGB Dense Map` comparison
tab. Both maps follow the same evolving DPVO keyframe poses. The normal RGB
PLY is preserved, and a sibling `<name>_sam.ply` contains region colors plus
`region_id` (uint16) and `keyframe_timestamp` (uint32) per vertex. The adjacent
JSON records per-keyframe region palettes/counts and total labeled coverage.
These are SAM region IDs, not semantic class names or globally fused 3D object
identities. Video discovery can change IDs; without `--sam-video`, IDs are
frame-local and must be interpreted together with the keyframe timestamp.

The 30-second Office_01 combined trial completed 180 processed frames and
produced 277,186 points from 91 aligned keyframes; 193,712 points (69.9%) received
SAM labels. Six depth pairs failed the existing alignment-error gate. Sampled
peak total GPU use was 3,338 MiB of 8,188 MiB (one-second `nvidia-smi` sampling,
including the desktop, viewer opened afterward). Both models remained loaded;
inference is scheduled sequentially, not on overlapping GPU streams. SAM
averaged 235 ms/frame; this excludes DPVO and DA3 time. SAM's temporal modules
remain PyTorch BF16, while its image encoder/seeding and DA3 use TensorRT.
Tests: `pixi run python -m unittest test_dense_regions test_sam_video test_sam_segmenter -q`.

### iPhone
```bash
python demo.py --imagedir=movies/IMG_0492.MOV --calib=calib/iphone.txt --stride=5 --plot --viz
```

### TartanAir
Download a sequence from [TartanAir](https://theairlab.org/tartanair-dataset/) (several samples are availabe from download directly from the webpage)
```bash
python demo.py --imagedir=<path to image_left> --calib=calib/tartan.txt --stride=1 --plot --viz
```

### EuRoC
Download a sequence from [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) (download ASL format)
```bash
python demo.py --imagedir=<path to mav0/cam0/data/> --calib=calib/euroc.txt --stride=2 --plot --viz
```

## SLAM Backends
To run DPVO with a SLAM backend (i.e., DPV-SLAM), add
```bash
--opts LOOP_CLOSURE True
```
to any `evaluate_X.py` script or to `demo.py`

If installed, the classical backend can also be enabled using 
```
--opts CLASSIC_LOOP_CLOSURE True
```

## Evaluation
We provide evaluation scripts for TartanAir, EuRoC, TUM-RGBD and ICL-NUIM. Up to date result logs on these datasets can be found in the `logs` directory.

### TartanAir:
Results on the validation split and test set can be obtained with the command:
```
python evaluate_tartan.py --trials=5 --split=validation --plot --save_trajectory
```

### EuRoC:
```
python evaluate_euroc.py --trials=5 --plot --save_trajectory
```

### TUM-RGBD:
```
python evaluate_tum.py --trials=5 --plot --save_trajectory
```

### ICL-NUIM:
```
python evaluate_icl_nuim.py --trials=5 --plot --save_trajectory
```

### KITTI:
```
python evaluate_kitti.py --trials=5 --plot --save_trajectory
```

## Training
Make sure you have run `./download_models_and_data.sh`. Your directory structure should look as follows

```Shell
├── datasets
    ├── TartanAir.pickle
    ├── TartanAir
        ├── abandonedfactory
        ├── abandonedfactory_night
        ├── ...
        ├── westerndesert
    ...
```

To train (log files will be written to `runs/<your name>`). Model will be run on the validation split every 10k iterations
```
python train.py --steps=240000 --lr=0.00008 --name=<your name>
```

## Change Log
* **Aug 2022**: Initial release
* **Sep 2022**: Add link to docker
* **Mar 2023**: Google Colab, TUM + ICL-NUIM evaluation code, flags for saving output
* **July 2024**: Add DPV-SLAM. Update output-saving utilities.


## Acknowledgements
* Our Viewer is adapted from DSO.
