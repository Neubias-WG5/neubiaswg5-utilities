from setuptools import setup

import neubiaswg5

with open("README.md", "r") as fh:
    long_description = fh.read()

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
    install_requires=['scipy', 'tifffile', 'scikit-image', 'scikit-learn', 'pandas',
                      'numpy', 'opencv-python-headless', 'shapely', 'skan', 'numba', 'imageio'],
    license='LICENSE'
)

