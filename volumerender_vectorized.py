# Original with Vectorization
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import h5py as h5
from scipy.interpolate import interpn
from line_profiler import LineProfiler

"""
Create Your Own Volume Rendering (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the Schrodinger-Poisson system with the Spectral method
"""


## @brief Transforms intensity values into RGBA color space.
#  @param x The input intensity value.
#  @return Tuple of RGBA values.
#  This function applies a custom transfer function to map intensity values
#  to RGBA color space, facilitating volume rendering visualization.
def transferFunction(x):

	r = 1.0*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) +  0.1*np.exp( -(x - -3.0)**2/0.5 )
	g = 1.0*np.exp( -(x - 9.0)**2/1.0 ) +  1.0*np.exp( -(x - 3.0)**2/0.1 ) +  0.1*np.exp( -(x - -3.0)**2/0.5 )
	b = 0.1*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) +  1.0*np.exp( -(x - -3.0)**2/0.5 )
	a = 0.6*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) + 0.01*np.exp( -(x - -3.0)**2/0.5 )

	return r,g,b,a

## @brief Main function to perform volume rendering.
#  @param Nangles Number of angles to render.
#  @param num_runs Number of runs for each angle.
#  @param interpolationMethod Method used for interpolation.
#  This function loads a 3D data cube, performs volume rendering from different angles,
#  and visualizes the results. It utilizes numpy and matplotlib for computation and visualization.
def main(Nangles, num_runs, interpolationMethod):
	""" 
    Main function for volume rendering.
    
    @param Nangles Number of angles for rendering.
    @param num_runs Number of runs for rendering.
    @param interpolationMethod Method used for interpolation.
    """
  """ Volume Rendering """
  
  # Load Datacube
  with h5.File('datacube.hdf5', 'r') as f:
    datacube = np.array(f['density'])

  # Datacube Grid
  Nx, Ny, Nz = datacube.shape
  x = np.linspace(-Nx/2, Nx/2, Nx)
  y = np.linspace(-Ny/2, Ny/2, Ny)
  z = np.linspace(-Nz/2, Nz/2, Nz)
  points = (x, y, z)

  average_times = []
  total_time = np.zeros(num_runs)
  scene_times_all_runs = [[] for _ in range(Nangles)]

  for run in range(num_runs):
    print(f"Run {run+1} of {num_runs}")

    # Intialise 1D empty array of size Nangles
    average = np.zeros(Nangles)

    for i in range(Nangles):
      print('Rendering Scene ' + str(i+1) + ' of ' + str(Nangles) + '.\n')
      start = timer()

      # Camera Grid / Query Points -- rotate camera view
      angle = np.pi/2 * i / Nangles
      N = 180
      c = np.linspace(-N/2, N/2, N)
      qx, qy, qz = np.meshgrid(c,c,c)
      qxR = qx
      qyR = qy * np.cos(angle) - qz * np.sin(angle)
      qzR = qy * np.sin(angle) + qz * np.cos(angle)
      qi = np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T

      # Interpolate onto Camera Grid
      camera_grid = interpn(points, datacube, qi, method=interpolationMethod).reshape((N,N,N))
    
      # Do Volume Rendering
      image = np.zeros((camera_grid.shape[1],camera_grid.shape[2],3))

      r, g, b, a = transferFunction(np.log(camera_grid))
    
      # Apply transfer function to each slice of camera_grid
      image[:, :, 0] = np.sum(a * r, axis=0)
      image[:, :, 1] = np.sum(a * g, axis=0)
      image[:, :, 2] = np.sum(a * b, axis=0)

      end = timer()
      print(f"Time to render scene {i+1}: {end - start} seconds")
      # Add to average
      average[i] = end - start
      scene_times_all_runs[i].append(end - start)

      image = np.clip(image,0.0,1.0)

      # Plot Volume Rendering
      plt.figure(figsize=(4,4), dpi=80)

      plt.imshow(image)
      plt.axis('off')

      # Save figure
      plt.savefig('volumerender' + str(i) + '.png',dpi=240,  bbox_inches='tight', pad_inches = 0)
      
    average_times.append(average)

    total_time[run] = np.sum(average)

    # Print mean and standard deviation and max/min of rendering times
    print(f"Mean rendering time: {np.mean(average)} seconds")
    print(f"Standard deviation of rendering time: {np.std(average)} seconds")
    print(f"Max rendering time: {np.max(average)} seconds")
    print(f"Min rendering time: {np.min(average)} seconds")

    # Plot Simple Projection -- for Comparison
    plt.figure(figsize=(4,4), dpi=80)

    plt.imshow(np.log(np.mean(datacube,0)), cmap = 'viridis')
    plt.clim(-5, 5)
    plt.axis('off')

    # Save figure
    plt.savefig('projection.png',dpi=240,  bbox_inches='tight', pad_inches = 0)
    plt.show()

  plt.figure()
  plt.plot(np.arange(1, num_runs + 1), [np.sum(times) for times in average_times], marker='o', linestyle='-')
  plt.xlabel('Run (n)')
  plt.ylabel('Total Running Time (seconds)')
  plt.title(f'Total Running Time Across {num_runs} Runs for {interpolationMethod}')
  plt.grid(True)
  plt.savefig('total_running_time.png', dpi=240, bbox_inches='tight', pad_inches=0)
  plt.show()

  # Print mean and standard deviation and max/min of running main()
  print(f"Mean time for main(): {np.mean(total_time)} seconds")
  print(f"Standard deviation of time for main(): {np.std(total_time)} seconds")
  print(f"Max time for main(): {np.max(total_time)} seconds")
  print(f"Min time for main(): {np.min(total_time)} seconds")

  plt.figure(figsize=(10, 6))
  for i, scene_times in enumerate(scene_times_all_runs):
      plt.plot(np.arange(1, num_runs + 1), scene_times, label=f'Scene {i+1}')
  plt.xlabel('Run (n)')
  plt.ylabel('Rendering Time (seconds)')
  plt.title(f'Running Time for Each Scene Across {num_runs} Runs for {interpolationMethod}')
  plt.legend()
  plt.grid(True)
  plt.savefig('scene_rendering_times_across_runs.png', dpi=240, bbox_inches='tight', pad_inches=0)
  plt.show()

## @brief Profiles the `main` function using LineProfiler.
#  This function is intended for performance analysis and optimization.
def profile_line_profiler():
    profiler = LineProfiler()
    profiler.add_function(main)
    profiler.run('main(10, 1, "nearest")')
    profiler.print_stats()

if __name__== "__main__":
	interpolation_method = ["nearest", "linear"]
	for method in interpolation_method:
		main(10, 10, method)

	# Uncomment the line below to profile the main function.
	# profile_line_profiler()
