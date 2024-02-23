import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from scipy.interpolate import interpn

def transferFunction(x):
    calculation_1 = torch.exp( -(x - 9.0)**2/1.0 )
    calculation_2 = torch.exp( -(x - 3.0)**2/0.1 )
    calculation_3 = torch.exp( -(x + 3.0)**2/0.5 )
    
    r = 1.0*(calculation_1)+0.1*(calculation_2)+0.1*(calculation_3)
    g = 1.0*(calculation_1)+1.0*(calculation_2)+0.1*(calculation_3)
    b = 0.1*(calculation_1)+0.1*(calculation_2)+1.0*(calculation_3)
    a = 0.6*(calculation_1)+0.1*(calculation_2)+0.01*(calculation_3)
    return r, g, b, a

def main():
    """ Volume Rendering """
    
    # Load Datacube
    f = h5.File('datacube.hdf5', 'r')
    datacube = torch.tensor(f['density'], dtype=torch.float32)
    
    # Datacube Grid
    Nx, Ny, Nz = datacube.shape
    x = torch.linspace(-Nx/2, Nx/2, Nx)
    y = torch.linspace(-Ny/2, Ny/2, Ny)
    z = torch.linspace(-Nz/2, Nz/2, Nz)
    points = (x.numpy(), y.numpy(), z.numpy())
    
    # Do Volume Rendering at Different Viewing Angles
    Nangles = 10
    for i in range(Nangles):
        
        print('Rendering Scene ' + str(i+1) + ' of ' + str(Nangles) + '.\n')
    
        # Camera Grid / Query Points -- rotate camera view
        angle = torch.tensor(np.pi/2 * i / Nangles, dtype=torch.float32)
        N = 180
        c = torch.linspace(-N/2, N/2, N)
        qx, qy, qz = torch.meshgrid(c,c,c)
        qxR = qx
        qyR = qy * torch.cos(angle) - qz * torch.sin(angle) 
        qzR = qy * torch.sin(angle) + qz * torch.cos(angle)
        qi = torch.stack((qxR.flatten(), qyR.flatten(), qzR.flatten()), dim=1)
        
        # Interpolate onto Camera Grid
        camera_grid = interpn(points, datacube.numpy(), qi.numpy(), method='linear').reshape((N,N,N))
        camera_grid = torch.tensor(camera_grid, dtype=torch.float32)
        
        # Do Volume Rendering
        image = torch.zeros((camera_grid.shape[1], camera_grid.shape[2], 3))
    
        for dataslice in camera_grid:
            dataslice_torch = torch.tensor(dataslice)  # Convert dataslice to PyTorch tensor
            r, g, b, a = transferFunction(torch.log(dataslice_torch))
            image[:,:,0] = a*r + (1-a)*image[:,:,0]
            image[:,:,1] = a*g + (1-a)*image[:,:,1]
            image[:,:,2] = a*b + (1-a)*image[:,:,2]
        
        image = torch.clamp(image, 0.0, 1.0)
        
        # Plot Volume Rendering
        plt.figure(figsize=(4,4), dpi=80)
        
        plt.imshow(image.numpy())
        plt.axis('off')
        
        # Save figure
        plt.savefig('volumerender' + str(i) + '.png',dpi=240,  bbox_inches='tight', pad_inches = 0)
    
    
    
    # Plot Simple Projection -- for Comparison
    plt.figure(figsize=(4,4), dpi=80)
    
    plt.imshow(torch.log(torch.mean(datacube,0)).numpy(), cmap = 'viridis')
    plt.clim(-5, 5)
    plt.axis('off')
    
    # Save figure
    plt.savefig('projection.png',dpi=240,  bbox_inches='tight', pad_inches = 0)
    plt.show()
    
    return 0

if __name__== "__main__":
    main()
