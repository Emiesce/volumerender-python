# Setup File for compiling Cython File

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("volumerender_cfunction.pyx"),
    compiler_directives={'language_level' : "3"}
)
