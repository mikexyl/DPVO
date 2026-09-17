# Build on the matching Jetson; reuse its validated DPVO/CUDA image.
ARG DPVO_BASE_IMAGE=dpvo:jetson-jp62-viser
FROM ${DPVO_BASE_IMAGE}
ARG DPVO_NUMPY_MAJOR=1
ARG DPVO_BUILD_JOBS=2
ENV DEBIAN_FRONTEND=noninteractive CMAKE_BUILD_PARALLEL_LEVEL=${DPVO_BUILD_JOBS}
SHELL ["/bin/bash", "-c"]
RUN . /etc/os-release && test "$VERSION_CODENAME" = noble && \
    apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git \
    && curl --retry 5 --retry-all-errors -fsSL https://github.com/ros-infrastructure/ros-apt-source/releases/download/1.3.0/ros2-apt-source_1.3.0.noble_all.deb -o /tmp/ros-source.deb \
    && dpkg -i /tmp/ros-source.deb && apt-get update \
    && apt-get install -y --no-install-recommends ros-jazzy-ros-base ros-jazzy-rmw-cyclonedds-cpp \
       python3-colcon-common-extensions python3-dev libopencv-dev libeigen3-dev cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*
# Freeze NVIDIA's installed stack before resolving learned-model dependencies.
ENV PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL="" PIP_DEFAULT_TIMEOUT=120
RUN python -c "import importlib.metadata as m; names={'torch','torchvision','torchaudio','numpy','scipy','opencv-python','opencv-python-headless','tensorrt'}; print('\\n'.join(name+'=='+m.version(name) for name in sorted(names.intersection(d.metadata['Name'].lower() for d in m.distributions()))))" > /tmp/base-constraints.txt
RUN python -m pip install --no-cache-dir -c /tmp/base-constraints.txt \
      kornia==0.8.2 timm huggingface-hub safetensors einops pyyaml viser==1.1.0 && \
    python -c "import numpy; assert int(numpy.__version__.split('.')[0]) == ${DPVO_NUMPY_MAJOR}"
COPY DBoW2 /tmp/DBoW2
COPY DPRetrieval /tmp/DPRetrieval
# The bundled pybind11 2.9 predates Python 3.12 / NumPy 2. Patch only the image copy.
RUN python -m pip install --no-cache-dir pybind11==2.13.6 && \
    sed -i 's/add_subdirectory(pybind11)/find_package(pybind11 CONFIG REQUIRED)/' /tmp/DPRetrieval/CMakeLists.txt
RUN cmake -S /tmp/DBoW2 -B /tmp/dbow-build -DBUILD_Demo=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    && cmake --build /tmp/dbow-build && cmake --install /tmp/dbow-build && ldconfig \
    && CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -Dpybind11_DIR=$(python -m pybind11 --cmakedir)" python -m pip install --no-deps --no-build-isolation /tmp/DPRetrieval \
    && python -m pip install --no-deps 'git+https://github.com/MIT-SPARK/TEASER-plusplus.git@52a9c52ee7d4c838c5e8a75458c33178be5bfb70'
COPY ros2 /opt/online_ws/src
RUN source /opt/ros/jazzy/setup.bash && cd /opt/online_ws && \
    /usr/bin/colcon build --packages-select dpvo_multi_robot_interfaces dpvo_multi_robot \
      --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
COPY dpvo /opt/dpvo/dpvo
COPY config /opt/dpvo/config
COPY deploy/jetson /opt/dpvo/deploy/jetson
ENV DPVO_USE_SYSTEM_PYTHON=1 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    TORCH_HOME=/models/loop_frontend/torch HF_HOME=/models/loop_frontend/huggingface \
    HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
# TensorRT needs host driver libraries injected by --runtime=nvidia.
RUN python -c 'import dpretrieval, teaserpp_python, pyrealsense2'
WORKDIR /opt/dpvo
ENTRYPOINT ["bash", "/opt/dpvo/deploy/jetson/multi_robot/entrypoint.sh"]
