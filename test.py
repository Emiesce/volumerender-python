import pytest
import numpy as np
import h5py
from volumerender import transferFunction, main

def test_main():
    try:
        main(180)
    except Exception:
        pytest.fail("main(180) raised an exception")

def test_datacube_file():
    try:
        with h5py.File('datacube.hdf5', 'r') as f:
            pass
    except Exception:
        pytest.fail("datacube.hdf5 file not found or not accessible")

def test_transferFunction():
    r, g, b, a = transferFunction(np.array([1, 2, 3]))
    assert 0 <= r.all() <= 1
    assert 0 <= g.all() <= 1
    assert 0 <= b.all() <= 1
    assert 0 <= a.all() <= 1