import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import h5py as h5
from scipy.interpolate import interpn
from line_profiler import LineProfiler
from numba import jit

"""
Create Your Own Volume Rendering (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the Schrodinger-Poisson system with the Spectral method
"""
@jit
def transferFunction(x):
	r = 1.0*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) +  0.1*np.exp( -(x - -3.0)**2/0.5 )
	g = 1.0*np.exp( -(x - 9.0)**2/1.0 ) +  1.0*np.exp( -(x - 3.0)**2/0.1 ) +  0.1*np.exp( -(x - -3.0)**2/0.5 )
	b = 0.1*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) +  1.0*np.exp( -(x - -3.0)**2/0.5 )
	a = 0.6*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) + 0.01*np.exp( -(x - -3.0)**2/0.5 )
	
	return r,g,b,a

def main():
	""" Volume Rendering """
	
	# Load Datacube
	f = h5.File('datacube.hdf5', 'r')
	datacube = np.array(f['density'])
	
	# Datacube Grid
	Nx, Ny, Nz = datacube.shape
	x = np.linspace(-Nx/2, Nx/2, Nx)
	y = np.linspace(-Ny/2, Ny/2, Ny)
	z = np.linspace(-Nz/2, Nz/2, Nz)
	points = (x, y, z)
	
	# Do Volume Rendering at Different Veiwing Angles
	Nangles = 10
	# Intialise 1D empty array of size Nangles
	average = np.zeros(Nangles)

	for i in range(Nangles):
		start = timer()
		print('Rendering Scene ' + str(i+1) + ' of ' + str(Nangles) + '.\n')
	
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
		camera_grid = interpn(points, datacube, qi, method='linear').reshape((N,N,N))
		
		# Do Volume Rendering
		image = np.zeros((camera_grid.shape[1],camera_grid.shape[2],3))
	
		for dataslice in camera_grid:
			r,g,b,a = transferFunction(np.log(dataslice))
			image[:,:,0] = a*r + (1-a)*image[:,:,0]
			image[:,:,1] = a*g + (1-a)*image[:,:,1]
			image[:,:,2] = a*b + (1-a)*image[:,:,2]

		end = timer()
		print(f"Time to render scene {i+1}: {end - start} seconds")
		# Add to average
		average[i] = end - start

		image = np.clip(image,0.0,1.0)
		
		# Plot Volume Rendering
		plt.figure(figsize=(4,4), dpi=80)
		
		plt.imshow(image)
		plt.axis('off')
		
		# Save figure
		plt.savefig('volumerender' + str(i) + '.png',dpi=240,  bbox_inches='tight', pad_inches = 0)
	
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

	return np.sum(average)

# Profile the main function using the LineProfiler
def profile_line_profiler():
    profiler = LineProfiler()
    profiler.add_function(main)
    profiler.run('main()')
    profiler.print_stats()

if __name__== "__main__":
	n_tests = 1
	total_time = np.zeros(n_tests)
	for i in range(n_tests):
		print(f"Running main() test {i+1} of {n_tests}")
		total_time[i] = main()

	# Print mean and standard deviation and max/min of running main()
	print(f"Mean time for main(): {np.mean(total_time)} seconds")
	print(f"Standard deviation of time for main(): {np.std(total_time)} seconds")
	print(f"Max time for main(): {np.max(total_time)} seconds")
	print(f"Min time for main(): {np.min(total_time)} seconds")

	profile_line_profiler()
