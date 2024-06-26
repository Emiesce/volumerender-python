# CuPy w/ Vectorization
import cupy as cp
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import h5py as h5
from line_profiler import LineProfiler
from cupyx.scipy.interpolate import interpn as interpn_cupy

"""
@brief Volume Rendering with Python using the CuPy library and vectorization techniques.
@details This script demonstrates how to use CuPy for efficient numerical computations
and volume rendering. The simulation involves the Schrodinger-Poisson system
with the spectral method. The code includes examples of using CuPy for
data manipulation, interpolation, and visualization with matplotlib.
"""

# Define the transfer function to adjust color intensity based on density values
def transferFunction(x):
	"""
    @brief Transfer function to adjust color intensity based on density values.
    @param x: Density value for which color intensity is calculated.
    @return Tuple of RGBA values.
    @details Applies cupy exponential functions to simulate different material densities and calculates RGBA values.
    """
	
    # Apply exponential functions to simulate different material densities
    calculation_1 = cp.exp(-(x - 9.0) ** 2 / 1.0)
    calculation_2 = cp.exp(-(x - 3.0) ** 2 / 0.1)
    calculation_3 = cp.exp(-(x - - 3.0) ** 2 / 0.5)

    # Calculate RGBA values based on the density calculations
    r = 1.0 * calculation_1 + 0.1 * calculation_2 + 0.1 * calculation_3
    g = 1.0 * calculation_1 + 1.0 * calculation_2 + 0.1 * calculation_3
    b = 0.1 * calculation_1 + 0.1 * calculation_2 + 1.0 * calculation_3
    a = 0.6 * calculation_1 + 0.1 * calculation_2 + 0.01 * calculation_3
    return r, g, b, a

def main(Nangles, num_runs, interpolationMethod):
    """
    @brief Main function for volume rendering.
    @param Nangles: Number of angles to render.
    @param num_runs: Number of runs to perform.
    @param interpolationMethod: Method for interpolation ('nearest' or 'linear').
    @details This function loads a data cube, performs volume rendering from different angles,
    and measures the performance of rendering. It uses CuPy for GPU-accelerated computations.
    """
	
	
  """ Volume Rendering """

  # Load Datacube
  with h5.File('datacube.hdf5', 'r') as f:
    datacube = cp.array(f['density'])

  # Datacube Grid
  Nx, Ny, Nz = datacube.shape
  x = cp.linspace(-Nx/2, Nx/2, Nx)
  y = cp.linspace(-Ny/2, Ny/2, Ny)
  z = cp.linspace(-Nz/2, Nz/2, Nz)
  points = (x.get(), y.get(), z.get())  # Convert to NumPy for scipy function compatibility

  N = 180
  average_times = []
  total_time = cp.zeros(num_runs)
  scene_times_all_runs = [[] for _ in range(Nangles)]

  for run in range(num_runs):
    print(f"Run {run+1} of {num_runs}")

    # Intialise 1D empty array of size Nangles
    rendering_times_gpu = cp.zeros(Nangles)

    for i in range(Nangles):
      print('Rendering Scene ' + str(i+1) + ' of ' + str(Nangles) + '.\n')
      start_gpu = cp.cuda.Event()
      end_gpu = cp.cuda.Event()

      start_gpu.record()

      # Camera Grid / Query Points -- rotate camera view
      angle = cp.pi / 2 * i / Nangles
      c = cp.linspace(-N / 2, N / 2, N)
      qx, qy, qz = cp.meshgrid(c, c, c)
      qxR = qx
      qyR = qy * cp.cos(angle) - qz * cp.sin(angle)
      qzR = qy * cp.sin(angle) + qz * cp.cos(angle)
      qi = cp.stack([qxR.ravel(), qyR.ravel(), qzR.ravel()], axis=1)

      # Interpolate onto Camera Grid
      camera_grid = interpn_cupy(points, datacube, qi, method=interpolationMethod).reshape((N, N, N))

      # Apply transfer function to the entire camera grid at once
      r, g, b, a = transferFunction(cp.log(camera_grid))

      # Initialize image array
      image = cp.zeros((camera_grid.shape[1], camera_grid.shape[2], 3), dtype=cp.float32)

      # Apply transfer function to each slice of camera_grid
      image[:, :, 0] = cp.sum(a * r, axis=0)
      image[:, :, 1] = cp.sum(a * g, axis=0)
      image[:, :, 2] = cp.sum(a * b, axis=0)

      end_gpu.record()
      cp.cuda.stream.get_current_stream().synchronize()

      print('Time to render image (GPU): ' + str(cp.cuda.get_elapsed_time(start_gpu, end_gpu)) + ' milliseconds.\n')
      rendering_times_gpu[i] = cp.cuda.get_elapsed_time(start_gpu, end_gpu)

      scene_times_all_runs[i].append(cp.cuda.get_elapsed_time(start_gpu, end_gpu))

      # Clip the image values to be between 0 and 1 before displaying
      image = cp.clip(image, 0.0, 1.0)

      # Display and save the rendered image
      plt.figure(figsize=(4, 4), dpi=80)
      plt.imshow(image.get().astype('float32'))
      plt.axis('off')
      plt.savefig(f'volumerender{i}.png', dpi=240, bbox_inches='tight', pad_inches=0)
    
    average_times.append(rendering_times_gpu.get())

    total_time[run] = np.sum(rendering_times_gpu)

    # Print mean and standard deviation of rendering times for GPU
    print('Mean rendering time for images (GPU): ' + str(cp.mean(rendering_times_gpu)) + ' milliseconds.')
    print('Standard deviation of rendering times (GPU): ' + str(cp.std(rendering_times_gpu)) + ' milliseconds.')
    print('Max and min rendering times (GPU): ' + str(cp.max(rendering_times_gpu)) + ' milliseconds, ' + str(cp.min(rendering_times_gpu)) + ' milliseconds.\n')

    # Generate and save a projection of the datacube along an axis
    plt.figure(figsize=(4, 4), dpi=80)
    projection = cp.log(cp.mean(datacube, axis=0))
    plt.imshow(cp.asnumpy(projection), cmap='viridis')
    plt.clim(-5, 5)
    plt.axis('off')
    plt.savefig('projection.png', dpi=240, bbox_inches='tight', pad_inches=0)
    plt.show()

  plt.figure()
  plt.plot(np.arange(1, num_runs + 1), [np.sum(times) / 1000 for times in average_times], marker='o', linestyle='-')
  plt.xlabel('Run (n)')
  plt.ylabel('Total Running Time (seconds)')
  plt.title(f'Total Running Time Across {num_runs} Runs for {interpolationMethod}')
  plt.grid(True)
  plt.savefig('total_running_time.png', dpi=240, bbox_inches='tight', pad_inches=0)
  plt.show()

  # Print mean and standard deviation and max/min of running main()
  print(f"Mean time for main(): {cp.mean(total_time) / 1000} seconds")
  print(f"Standard deviation of time for main(): {cp.std(total_time) / 1000} seconds")
  print(f"Max time for main(): {cp.max(total_time) / 1000} seconds")
  print(f"Min time for main(): {cp.min(total_time) / 1000} seconds")

  plt.figure(figsize=(10, 6))
  for i, scene_times in enumerate(scene_times_all_runs):
      plt.plot(np.arange(1, num_runs + 1), scene_times, label=f'Scene {i+1}')
  plt.xlabel('Run (n)')
  plt.ylabel('Rendering Time (miliseconds)')
  plt.title(f'Running Time for Each Scene Across {num_runs} Runs for {interpolationMethod}')
  plt.legend()
  plt.grid(True)
  plt.savefig('scene_rendering_times_across_runs.png', dpi=240, bbox_inches='tight', pad_inches=0)
  plt.show()

# Profile the main function using the LineProfiler
def profile_line_profiler():
	"""
    @brief Profile the main function using the LineProfiler.
    @details Adds the main function to the LineProfiler, runs it with predefined parameters,
    and prints the profiling statistics.
    """
    profiler = LineProfiler()
    profiler.add_function(main)
    profiler.run('main(10, 1, "nearest")')
    profiler.print_stats()

if __name__== "__main__":
	interpolation_method = ["nearest", "linear"]
	for method in interpolation_method:
		main(10, 10, method)
		
	# Uncomment the line below to enable profiling
	# profile_line_profiler()
