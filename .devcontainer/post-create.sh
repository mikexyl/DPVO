#!/bin/bash
set -e

echo "=== DPVO Post-Create Setup ==="

# Install pixi package manager
echo "Installing pixi..."
curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"

cd /home/developer/workspace/DPVO

# Initialize pixi environment
echo "Setting up pixi environment..."
pixi install

# Create thirdparty directory and link Eigen
mkdir -p thirdparty
if [ ! -d "thirdparty/eigen-3.4.0" ]; then
    ln -sf /opt/eigen-3.4.0 thirdparty/eigen-3.4.0
fi

# Install DPVO package in editable mode
echo "Installing DPVO package..."
pixi run build

# Build and install Pangolin viewer (if submodule exists)
if [ -d "Pangolin" ] && [ -f "Pangolin/CMakeLists.txt" ]; then
    echo "Building Pangolin..."
    cd Pangolin
    sudo ./scripts/install_prerequisites.sh recommended || true
    mkdir -p build && cd build
    cmake ..
    make -j$(nproc)
    sudo make install
    cd ../..
    
    # Install DPViewer
    echo "Installing DPViewer..."
    pixi run pip install ./DPViewer
fi

# Build and install DBoW2 (for loop closure)
if [ -d "DBoW2" ] && [ -f "DBoW2/CMakeLists.txt" ]; then
    echo "Building DBoW2..."
    cd DBoW2
    mkdir -p build && cd build
    cmake ..
    make -j$(nproc)
    sudo make install
    cd ../..
    
    # Install DPRetrieval
    if [ -d "DPRetrieval" ]; then
        echo "Installing DPRetrieval..."
        pixi run pip install ./DPRetrieval
    fi
fi

# Download models and data if not already present
if [ ! -d "checkpoints" ]; then
    echo "Downloading models and data..."
    ./download_models_and_data.sh || echo "Note: Model download may require manual intervention"
fi

echo "=== DPVO Setup Complete ==="
echo "You can now run demos with: pixi run python demo.py"
echo "Or use: pixi shell to activate the environment"
