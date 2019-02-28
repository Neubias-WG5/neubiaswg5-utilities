from skimage import morphology
import numpy as np
from neubiaswg5.exporter import mask_to_objects_2d, mask_to_objects_3d


def skeleton_mask_to_objects_2d(mask, background=0, offset=None):
    """Process a 2d skeleton mask

    Parameters
    ----------
    mask: ndqrrqy
    background: int
    offset: tuple

    """
    dilated = morphology.dilation(mask, selem=morphology.selem.disk(2))
    return mask_to_objects_2d(dilated, background=background, offset=offset)


def skeleton_mask_to_objects_3d(mask, background=0, offset=None, assume_unique_labels=False, time=False, projection=0):
    """Process a 3d skeleton mask

    Parameters:
    -----------
    mask: ndarray
    background: int
    offset: tuple
    assume_unique_labels: bool
    time: bool
    projection: int
        Number of nearby frames to project in the current one. -1 for projecting from the whole volume.
        0 for no projection at all (default)
    """
    dilated = morphology.dilation(mask, selem=morphology.ball(1))

    # projection of skeleton from nearby frames
    if projection != 0:
        projected = np.zeros(dilated.shape, dtype=dilated.dtype)
        z_dims = dilated.shape[2]
        for z in range(z_dims):
            z_start = 0 if projection == -1 else max(0, z - projection)
            z_end = z_dims if projection == -1 else min(z_dims, z + projection + 1)
            projected[:, :, z] = np.max(dilated[:, :, z_start:z_end], axis=2)
        dilated = projected

    return mask_to_objects_3d(
        dilated, background=background,
        offset=offset, time=time,
        assume_unique_labels=assume_unique_labels
    )