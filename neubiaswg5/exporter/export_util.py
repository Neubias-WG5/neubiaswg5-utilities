import numpy as np
from PIL.Image import fromarray
from PIL.ImageDraw import ImageDraw
from shapely.geometry import Point


def draw_linestring(image, linestring, color=255):
    """Draw a linestring in the given color at the given location"""
    pil_image = fromarray(image)
    validated_color = color
    draw = ImageDraw(pil_image)
    if len(image.shape) > 2 and image.shape[2] > 1:
        validated_color = tuple(color)
    draw.line(linestring.coords, fill=validated_color)
    return np.asarray(pil_image)


def draw_poly(image, polygon, color=255):
    """Draw a polygon in the given color at the given location"""
    pil_image = fromarray(image)
    validated_color = color
    draw = ImageDraw(pil_image)
    if len(image.shape) > 2 and image.shape[2] > 1:
        validated_color = tuple(color)
    draw.polygon(polygon.boundary.coords, fill=validated_color, outline=validated_color)
    return np.asarray(pil_image)


def draw_slice_2d(_slice, mask):
    """Draw a slice in a 2D mask (ignore possible t and z coordinates)

    Parameters
    ----------
    _slice: AnnotationSlice
        A slice
    mask: ndarray
        2D mask
    """
    color = 255 if _slice.label is None else _slice.label
    if _slice.polygon.area > 0:
        draw_poly(mask, _slice.polygon, color=color)
    elif isinstance(_slice.polygon, Point):
        mask[int(_slice.polygon.y), int(_slice.polygon.x)] = color
    else:
        raise NotImplementedError("Does not support drawing such polygon {}.".format(type(_slice.polygon)))
    return mask


def draw_slice(_slice, mask):
    """Draw a slice in mask

    Parameters
    ----------
    _slice: AnnotationSlice
    mask: ndarray
        A mask. Its dimensions should be such that the slice can be drawn in it.

    Returns
    -------
    mask: ndarray
        The updated mask.
    """
    has_z = _slice.depth is not None
    has_t = _slice.time is not None

    if has_z and has_t:
        z_index = _slice.depth
        t_index = _slice.time
        mask[:, :, z_index, t_index] = draw_slice_2d(_slice, mask[:, :, z_index, t_index])
        return mask
    elif has_z or has_t:
        z_index = _slice.depth if has_z else _slice.time
        mask[:, :, z_index] = draw_slice_2d(_slice, mask[:, :, z_index])
        return mask
    else:
        return draw_slice_2d(_slice, mask)
