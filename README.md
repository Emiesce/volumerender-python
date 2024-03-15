# volumerender-python
Volume Render a Datacube

## Create Your Own Volume Rendering (With Python)

### Philip Mocz (2020) Princeton Univeristy, [@PMocz](https://twitter.com/PMocz)

### [📝 Read the Algorithm Write-up on Medium](https://medium.com/swlh/create-your-own-volume-rendering-with-python-655ca839b097)

Volume render a 3D datacube


```
python volumerender.py
```

![Simulation](./volumerender.png)



# Volume Rendering Project

## Introduction

This project explores the fascinating world of volume rendering, a technique for visualizing 3D volumetric data. With implementations ranging from basic CPU-based approaches to advanced GPU-accelerated algorithms, this repository offers a comprehensive suite of scripts tailored for different computational environments and performance needs. Whether you're new to volume rendering or seeking to leverage the power of GPU for complex visualizations, this collection provides valuable insights and tools.

## Prerequisites

Before diving into the volume rendering scripts, ensure you have the following dependencies installed:

- Python 3.x
- NumPy
- Matplotlib
- H5Py
- CuPy (for GPU-accelerated versions)
- LineProfiler (optional, for performance profiling)
- SciPy (for certain interpolation methods)

## Installation

1. Clone the repository to your local machine:

```bash
git clone https://github.com/Emiesce/volumerender-python.git
cd volumerender-python-main
```

2. Ensure your Python environment is set up and install the required dependencies:

```bash
pip install numpy matplotlib h5py cupy line_profiler scipy
```

## Files Overview

- `volumerender_original.py`: Basic volume rendering using NumPy and Matplotlib. Original version. 
- `volumerender_vectorized.py`: An improved version with vectorized operations for better performance on CPUs.
- `volumerender_cupy_improved.py`: Utilizes CuPy for GPU-accelerated volume rendering, significantly enhancing performance.
- `volumerender_cupy_improved_vectorized.py`: An advanced, vectorized, GPU-accelerated approach for top-tier performance and efficiency.
- `test.py`: Contains tests for validating functionality, ensuring the correct installation of dependencies and the availability of `datacube.hdf5`.

## Running the Scripts

### For CPU-based Rendering:

1. To run the basic or vectorized volume rendering scripts:

```bash
python volumerender_original.py
python volumerender_vectorized.py
```

### For GPU-accelerated Rendering:

1. Ensure you have a CUDA-compatible GPU and CuPy installed.
2. Run the GPU-accelerated scripts:

```bash
python volumerender_cupy_improved.py
python volumerender_cupy_improved_vectorized.py
```

## Data Preparation

The volume rendering scripts requires the `datacube.hdf5` file to be able to execute. 

## Performance Profiling

To analyze the performance of the rendering process, especially for the GPU-accelerated versions, you can utilize the LineProfiler. Sample commands for profiling are included in the scripts. Note that performance will vary based on your system's specifications and configurations.
