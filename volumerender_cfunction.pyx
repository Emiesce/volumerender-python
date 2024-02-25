cimport cython
import numpy as np
cimport numpy as np
import h5py as h5
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import scipy.interpolate 
cimport scipy.interpolate

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef ctransferFunction(np.ndarray[np.float64_t, ndim=2] x):
  cdef np.ndarray[np.float64_t, ndim=2] r, g, b, a
  r = 1.0*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) +  0.1*np.exp( -(x - -3.0)**2/0.5 )
  g = 1.0*np.exp( -(x - 9.0)**2/1.0 ) +  1.0*np.exp( -(x - 3.0)**2/0.1 ) +  0.1*np.exp( -(x - -3.0)**2/0.5 )
  b = 0.1*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) +  1.0*np.exp( -(x - -3.0)**2/0.5 )
  a = 0.6*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) + 0.01*np.exp( -(x - -3.0)**2/0.5 )
  return r,g,b,a


# Define the volume rendering function in Cython
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef volume_rendering():
# Load Datacube
    f = h5.File('datacube.hdf5', 'r')
    cdef np.ndarray[np.float64_t, ndim=3] datacube = np.array(f['density'], dtype=np.float64)
    
    # Datacube Grid
    cdef int Nx, Ny, Nz
    Nx = datacube.shape[0]
    Ny = datacube.shape[1]
    Nz = datacube.shape[2]
    cdef np.ndarray[np.float64_t, ndim=1] x = np.linspace(-Nx/2, Nx/2, Nx)
    cdef np.ndarray[np.float64_t, ndim=1] y = np.linspace(-Ny/2, Ny/2, Ny)
    cdef np.ndarray[np.float64_t, ndim=1] z = np.linspace(-Nz/2, Nz/2, Nz)
    cdef np.ndarray[double, ndim=1] points_x = x
    cdef np.ndarray[double, ndim=1] points_y = y
    cdef np.ndarray[double, ndim=1] points_z = z
    cdef tuple points = (points_x, points_y, points_z)
    
    # Do Volume Rendering at Different Viewing Angles
    cdef int Nangles = 10
    cdef int i, N
    cdef double angle
    cdef np.ndarray[np.float64_t, ndim=1] c
    cdef np.ndarray[np.float64_t, ndim=3] camera_grid
    cdef np.ndarray[np.float64_t, ndim=3] image
    cdef np.ndarray[np.float64_t, ndim=1] average = np.zeros((Nangles))

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
        camera_grid = scipy.interpolate.interpn(points, datacube, qi, method='linear').reshape((N,N,N))
        camera_grid = np.ascontiguousarray(camera_grid, dtype=np.float64)

        # Do Volume Rendering
        image = np.zeros((camera_grid.shape[1],camera_grid.shape[2],3))
        image = np.ascontiguousarray(image)

        for dataslice in camera_grid:
          r,g,b,a = ctransferFunction(np.log(dataslice))
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