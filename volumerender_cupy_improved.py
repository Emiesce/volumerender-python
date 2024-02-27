import cupy as cp
import matplotlib.pyplot as plt
import h5py as h5
from timeit import default_timer as timer
from cupyx.scipy.interpolate import interpn as interpn_cupy
from cupyx.profiler import benchmark
from line_profiler import LineProfiler

def transferFunction(x):
    calculation_1 = cp.exp(-(x - 9.0) ** 2 / 1.0)
    calculation_2 = cp.exp(-(x - 3.0) ** 2 / 0.1)
    calculation_3 = cp.exp(-(x + 3.0) ** 2 / 0.5)

    r = 1.0 * calculation_1 + 0.1 * calculation_2 + 0.1 * calculation_3
    g = 1.0 * calculation_1 + 1.0 * calculation_2 + 0.1 * calculation_3
    b = 0.1 * calculation_1 + 0.1 * calculation_2 + 1.0 * calculation_3
    a = 0.6 * calculation_1 + 0.1 * calculation_2 + 0.01 * calculation_3
    return r, g, b, a

def main():
    Nangles = 10
    rendering_times_gpu = cp.zeros(Nangles)

    f = h5.File('datacube.hdf5', 'r')
    datacube = cp.array(f['density'])

    Nx, Ny, Nz = datacube.shape
    x = cp.linspace(-Nx / 2, Nx / 2, Nx)
    y = cp.linspace(-Ny / 2, Ny / 2, Ny)
    z = cp.linspace(-Nz / 2, Nz / 2, Nz)
    points = (x, y, z)  # Kept as CuPy arrays for GPU acceleration

    N = 180
    camera_grid_shape = (N, N, N)
    camera_grid = cp.empty(camera_grid_shape, dtype=cp.float32)
    image = cp.zeros((N, N, 3), dtype=cp.float32)

    for i in range(Nangles):
        start_gpu = cp.cuda.Event()
        end_gpu = cp.cuda.Event()

        start_gpu.record()
        print(f'Rendering Scene {i + 1} of {Nangles}.\n')

        angle = cp.pi / 2 * i / Nangles
        c = cp.linspace(-N / 2, N / 2, N)
        qx, qy, qz = cp.meshgrid(c, c, c, indexing='ij')
        qxR = qx
        qyR = qy * cp.cos(angle) - qz * cp.sin(angle)
        qzR = qy * cp.sin(angle) + qz * cp.cos(angle)
        qi = cp.stack([qxR.ravel(), qyR.ravel(), qzR.ravel()], axis=1)

        camera_grid[:] = interpn_cupy(points, datacube, qi, method='linear').reshape(camera_grid_shape)

        for dataslice in camera_grid:
            r, g, b, a = transferFunction(cp.log(dataslice + 1e-8))
            image[:, :, 0] = a * r + (1 - a) * image[:, :, 0]
            image[:, :, 1] = a * g + (1 - a) * image[:, :, 1]
            image[:, :, 2] = a * b + (1 - a) * image[:, :, 2]

        end_gpu.record()
        print('Time to render scene (GPU): ' + str(cp.cuda.get_elapsed_time(start_gpu, end_gpu)) + ' milliseconds.\n')
        rendering_times_gpu[i] = cp.cuda.get_elapsed_time(start_gpu, end_gpu)

        image = cp.clip(image, 0.0, 1.0)
        plt.figure(figsize=(4, 4), dpi=80)
        plt.imshow(cp.asnumpy(image))
        plt.axis('off')
        plt.savefig(f'volumerender{i}.png', dpi=240, bbox_inches='tight', pad_inches=0)

    print('Mean rendering time (GPU): ' + str(cp.mean(rendering_times_gpu)) + ' milliseconds.')
    print('Standard deviation of rendering times (GPU): ' + str(cp.std(rendering_times_gpu)) + ' milliseconds.')
    print('Max and min rendering times (GPU): ' + str(cp.max(rendering_times_gpu)) + ' milliseconds, ' + str(cp.min(rendering_times_gpu)) + ' milliseconds.\n')

    plt.figure(figsize=(4, 4), dpi=80)
    projection = cp.log(cp.mean(datacube, axis=0) + 1e-8)
    plt.imshow(cp.asnumpy(projection), cmap='viridis')
    plt.clim(-5, 5)
    plt.axis('off')
    plt.savefig('projection.png', dpi=240, bbox_inches='tight', pad_inches=0)
    plt.show()

    return 0

def profile_line_profiler():
    profiler = LineProfiler()
    profiler.add_function(main)
    profiler.run('main()')
    profiler.print_stats()

if __name__ == "__main__":
    print(benchmark(main, n_repeat=20))  # Benchmark using CuPy profiler
    profile_line_profiler()  # Profile using LineProfiler
