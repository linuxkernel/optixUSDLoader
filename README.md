# optixUSDLoader
This is a OpenUSD loader to be used in the NVIDIA CUDA and Optix programming environment.
# OptiX OpenUSD Path Tracer (optixUSDLoader)

`optixUSDLoader` is a high-performance, GPU-accelerated ray tracing and physically-based rendering (PBR) application built using **NVIDIA OptiX**, **CUDA**, and **OpenUSD**. Evolving from traditional ray-tracing sample architectures, this project integrates native OpenUSD asset parsing directly into a custom OptiX ray-generation and traversal pipeline.

---

## Acknowledgments & Credits

* **NVIDIA:** Special thanks to NVIDIA for providing the foundational OptiX SDK sample architecture (`optixTriangle`) and robust GPU ray-tracing frameworks that made this project possible.
* **Google Gemini:** Portions of the code architecture, data loader integration logic, and project structure were developed and refined with the technical collaboration and assistance of **Google Gemini**.

---

## Key Features

* **OpenUSD Integration:** Custom data loader bridging Universal Scene Description (USD) scene hierarchies and primitives directly into GPU memory buffers.
* **NVIDIA OptiX Acceleration:** Leverages hardware-accelerated ray tracing, custom bounding volume hierarchy (BVH) building, and ray traversal.
* **CUDA-Powered Rendering:** High-throughput GPU parallel computing for Monte Carlo path tracing and shader evaluation.
* **Modern C++ Architecture:** Clean, modular host-device pipeline separation built on modern C++ standards.

---

## Prerequisites & Dependencies

To build and run this project, ensure you have the following installed and configured:

* **NVIDIA GPU** with Ray Tracing (RTX) capabilities (Compute Capability 7.5+ recommended)
* **NVIDIA CUDA Toolkit** (Version 11.0 or newer)
* **NVIDIA OptiX SDK** (Version 7.0 or newer)
* **Pixar OpenUSD** libraries
* **CMake** (Version 3.20 or higher)
* **GLFW / OpenGL** (for windowed interactive display and buffer presentation)

---

## Building and Compilation

1. Clone the repository:
   ```bash
   git clone [https://github.com/linuxkernel/optixUSDLoader.git](https://github.com/linuxkernel/optixUSDLoader.git)
   cd optixUSDLoader
Create a build directory and configure with CMake:

Bash
mkdir build && cd build
cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..
Compile the project:

Bash
cmake --build . --config Release
Usage
Run the compiled executable, pointing it to your target .usda or .usdc scene file:

Bash
./optixUSDLoader path/to/scene.usda
License
This project is open-source and available under the MIT License.
