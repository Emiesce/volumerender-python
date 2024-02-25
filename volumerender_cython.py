import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import h5py as h5
from volumerender_cfunction import volume_rendering # With Type Annotations in Cython


"""
Create Your Own Volume Rendering (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the Schrodinger-Poisson system with the Spectral method
"""
if __name__== "__main__":
    n_tests = 1
    total_time = np.zeros(n_tests)
    for i in range(n_tests):
        print(f"Running volume_rendering() test {i+1} of {n_tests}")
        total_time[i] = volume_rendering()

    # Print mean and standard deviation and max/min of running main()
    print(f"Mean time for main(): {np.mean(total_time)} seconds")
    print(f"Standard deviation of time for main(): {np.std(total_time)} seconds")
    print(f"Max time for main(): {np.max(total_time)} seconds")
    print(f"Min time for main(): {np.min(total_time)} seconds")
