import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from scipy.interpolate import interpn

def transferFunction(x):
	calculation_1 = cp.exp( -(x - 9.0)**2/1.0 )
	calculation_2 = cp.exp( -(x - 3.0)**2/0.1 )
	calculation_3 = cp.exp( -(x - -3.0)**2/0.5 )
	
	
    r = 1.0*calculation_1 +  0.1* calculation_2 +  0.1*calculation_3
    g = 1.0*calculation_1 +  1.0* calculation_2 +  0.1* calculation_3
    b = 0.1*calculation_1 +  0.1*calculation_2 +  1.0* calculation_3
    a = 0.6*calculation_1 +  0.1*calculation_2 + 0.01* calculation_3
    return r, g, b, a

def main():
    """ Volume Rendering """
    
    # Load Datacube
    f = h5.File('datacube.hdf5', 'r')
    datacube = cp.array(f['density'])
    
    # Datacube Grid
    Nx, Ny, Nz = datacube.shape
    x = cp.linspace(-Nx/2, Nx/2, Nx)
    y = cp.linspace(-Ny/2, Ny/2, Ny)
    z = cp.linspace(-Nz/2, Nz/2, Nz)
    points = (x.get(), y.get(), z.get())
    
    # Do Volume Rendering at Different Viewing Angles
    Nangles = 10
    for i in range(Nangles):
        
        print('Rendering Scene ' + str(i+1) + ' of ' + str(Nangles) + '.\n')
    
        # Camera Grid / Query Points -- rotate camera view
        angle = cp.pi/2 * i / Nangles
        N = 180
        c = cp.linspace(-N/2, N/2, N)
        qx, qy, qz = cp.meshgrid(c,c,c)
        qxR = qx
        qyR = qy * cp.cos(angle) - qz * cp.sin(angle) 
        qzR = qy * cp.sin(angle) + qz * cp.cos(angle)
        qi = cp.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T
        
        # Interpolate onto Camera Grid
        camera_grid = interpn(points, datacube.get(), qi.get(), method='linear').reshape((N,N,N))
        
        # Do Volume Rendering
        image = cp.zeros((camera_grid.shape[1], camera_grid.shape[2], 3))
    
        for dataslice in camera_grid:
            dataslice_cp = cp.array(dataslice)  # Convert dataslice to CuPy array
            r, g, b, a = transferFunction(cp.log(dataslice_cp))
            image[:,:,0] = a*r + (1-a)*image[:,:,0]
            image[:,:,1] = a*g + (1-a)*image[:,:,1]
            image[:,:,2] = a*b + (1-a)*image[:,:,2]
        
        image = cp.clip(image, 0.0, 1.0)
        
        # Plot Volume Rendering
        plt.figure(figsize=(4,4), dpi=80)
        
        plt.imshow(cp.asnumpy(image))
        plt.axis('off')
        
        # Save figure
        plt.savefig('volumerender' + str(i) + '.png',dpi=240,  bbox_inches='tight', pad_inches = 0)
    
    
    
    # Plot Simple Projection -- for Comparison
    plt.figure(figsize=(4,4), dpi=80)
    
    plt.imshow(cp.asnumpy(cp.log(cp.mean(datacube,0))), cmap = 'viridis')
    plt.clim(-5, 5)
    plt.axis('off')
    
    # Save figure
    plt.savefig('projection.png',dpi=240,  bbox_inches='tight', pad_inches = 0)
    plt.show()
    
    return 0

if __name__== "__main__":
    main()
