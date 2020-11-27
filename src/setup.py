from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

# Setup file to build the Cython implementation of seeded 3d region growing
# command: python setup.py build_ext --inplace
setup(ext_modules=cythonize("RegionGrowing3D.pyx"), include_dirs=[np.get_include()])
