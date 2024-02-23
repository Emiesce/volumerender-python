import cupy as cp
import matplotlib.pyplot as plt
import h5py as h5
from timeit import default_timer as timer
from scipy.interpolate import interpn

# Define the transfer function to adjust color intensity based on density values
def transferFunction(x):
    # Apply exponential functions to simulate different material densities
    calculation_1 = cp.exp(-(x - 9.0) ** 2 / 1.0)
    calculation_2 = cp.exp(-(x - 3.0) ** 2 / 0.1)
    calculation_3 = cp.exp(-(x - -3.0) ** 2 / 0.5)

    # Calculate RGBA values based on the density calculations
    r = 1.0 * calculation_1 + 0.1 * calculation_2 + 0.1 * calculation_3
    g = 1.0 * calculation_1 + 1.0 * calculation_2 + 0.1 * calculation_3
    b = 0.1 * calculation_1 + 0.1 * calculation_2 + 1.0 * calculation_3
    a = 0.6 * calculation_1 + 0.1 * calculation_2 + 0.01 * calculation_3
    return r, g, b, a

def main():
    # Load density data from an HDF5 file
    f = h5.File('datacube.hdf5', 'r')
    datacube = cp.array(f['density'])

    # Determine the grid size based on data dimensions
    Nx, Ny, Nz = datacube.shape
    x = cp.linspace(-Nx / 2, Nx / 2, Nx)
    y = cp.linspace(-Ny / 2, Ny / 2, Ny)
    z = cp.linspace(-Nz / 2, Nz / 2, Nz)
    points = (x.get(), y.get(), z.get())  # Convert to NumPy for scipy function compatibility

    # Set up parameters for the camera grid and initialize image array
    N = 180
    camera_grid_shape = (N, N, N)
    camera_grid = cp.empty(camera_grid_shape, dtype=cp.float32)
    image = cp.zeros((N, N, 3), dtype=cp.float32)

    # Render scenes from different angles
    Nangles = 10
    for i in range(Nangles):
        start = timer()
        print(f'Rendering Scene {i + 1} of {Nangles}.\n')

        # Calculate rotation angle and apply rotation to the camera grid
        angle = cp.pi / 2 * i / Nangles
        c = cp.linspace(-N / 2, N / 2, N)
        qx, qy, qz = cp.meshgrid(c, c, c, indexing='ij')
        qxR = qx
        qyR = qy * cp.cos(angle) - qz * cp.sin(angle)
        qzR = qy * cp.sin(angle) + qz * cp.cos(angle)
        qi = cp.stack([qxR.ravel(), qyR.ravel(), qzR.ravel()], axis=1)

        # Interpolate the rotated grid back onto the original datacube grid
        camera_grid[:] = cp.array(interpn(points, datacube.get(), qi.get(), method='linear').reshape(camera_grid_shape))

        # Apply transfer function to each slice of the camera grid and compose the final image
        for dataslice in camera_grid:
            r, g, b, a = transferFunction(cp.log(dataslice + 1e-8))
            image[:, :, 0] = a * r + (1 - a) * image[:, :, 0]
            image[:, :, 1] = a * g + (1 - a) * image[:, :, 1]
            image[:, :, 2] = a * b + (1 - a) * image[:, :, 2]
        end = timer()
        print('Time to render scene: ' + str(end - start) + ' seconds.\n')

        # Clip the image values to be between 0 and 1 before displaying
        image = cp.clip(image, 0.0, 1.0)

        # Display and save the rendered image
        plt.figure(figsize=(4, 4), dpi=80)
        plt.imshow(cp.asnumpy(image))
        plt.axis('off')
        plt.savefig(f'volumerender{i}.png', dpi=240, bbox_inches='tight', pad_inches=0)

    # Generate and save a projection of the datacube along an axis
    plt.figure(figsize=(4, 4), dpi=80)
    projection = cp.log(cp.mean(datacube, axis=0) + 1e-8)
    plt.imshow(cp.asnumpy(projection), cmap='viridis')
    plt.clim(-5, 5)
    plt.axis('off')
    plt.savefig('projection.png', dpi=240, bbox_inches='tight', pad_inches=0)
    plt.show()

    return 0

if __name__ == "__main__":
    main()
