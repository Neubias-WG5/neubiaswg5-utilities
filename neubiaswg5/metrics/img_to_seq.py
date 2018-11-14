# A script for converting OME-TIFF labeled masks to the Cell Tracking Challenge format
# Author: Martin Maska <xmaska@fi.muni.cz>, 2018

import tifffile as tiff
import os

# Convert the tracking results saved in an OME-TIFF image to a sequence of images
def img_to_seq(fname, out_dir, template):
    # Should be adapted to also work with regular multi-tiff files
    img = tiff.TiffFile(fname)
    T = img.ome_metadata.get('Image').get('Pixels').get('SizeT')
    Z = img.ome_metadata.get('Image').get('Pixels').get('SizeZ')
    Y = img.ome_metadata.get('Image').get('Pixels').get('SizeY')
    X = img.ome_metadata.get('Image').get('Pixels').get('SizeX')
    img_data = img.asarray().ravel()
    index = 0
    offset = Z*Y*X
    for t in range(T):
        with tiff.TiffWriter(os.path.join(out_dir, template + '{0:03d}.tif'.format(t))) as frame:
            frame.save(img_data[index:index+offset].reshape((Z,Y,X)))
        index += offset
