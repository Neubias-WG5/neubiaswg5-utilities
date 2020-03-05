import sys

from setuptools import setup

import neubiaswg5

with open("README.md", "r") as fh:
    long_description = fh.read()


if sys.version_info[0] == 3:
    packages = [
        'rasterio>=1.1.0', 'scipy>=1.0,<=1.2', 'tifffile==0.15.1', 'scikit-image>=0.14.0,<=0.14.2', 'scikit-learn>=0.17,<=0.20.2',
        'pandas>=0.20,<=0.24.1', 'numpy>=0.15.4', 'opencv-python-headless>=4,<=4.0.0.21',
        'shapely>=1.6,<=1.7a1', 'skan>=0.7,<=0.7.1', 'numba>=0.40,<=0.42.0',
        'imageio>=2,<=2.4.1', 'sldc>=1.1.2'
    ]
else:
    # TODO make a version for python 2.7
    packages = [
        'scipy', 'tifffile', 'scikit-image', 'scikit-learn', 'pandas',
        'numpy', 'opencv-python-headless', 'shapely', 'skan', 'numba',
        'imageio'
    ]


setup(
    name='Neubias-WG5 Utilities',
    version=neubiaswg5.__version__,
    description='A set of utilities for implementing neubias-wg5 softwares '
                '(metrics, annotation exporters, cytomine utilities,...)',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['neubiaswg5', 'neubiaswg5.exporter', 'neubiaswg5.metrics', 'neubiaswg5.helpers'],
    url='https://github.com/Neubias-WG5',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    install_requires=packages,
    license='LICENSE'
)

