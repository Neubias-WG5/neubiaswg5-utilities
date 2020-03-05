# -*- coding: utf-8 -*-
from collections import defaultdict

import numpy as np
from affine import Affine
from rasterio.features import shapes
from shapely.geometry import shape, box, Polygon, MultiPolygon
from skimage.measure import label as label_fn


class AnnotationSlice(object):
    """Represent a 2D slice of an object:
        - polygon encoding the shape
        - label of the object
        - time index (if relevant)
        - depth index (if relevant)
    """
    def __init__(self, polygon, label, time=None, depth=None):
        self._polygon = polygon
        self._label = label
        self._time = time
        self._depth = depth

    @property
    def polygon(self):
        return self._polygon

    @property
    def label(self):
        return self._label

    @property
    def time(self):
        return self._time

    @property
    def depth(self):
        return self._depth


def clamp(x, l, h):
    return max(l, min(h, x))


def geom_as_list(geometry):
    """Return the list of sub-polygon a polygon is made up of"""
    if geometry.geom_type == "Polygon":
        return [geometry]
    elif geometry.geom_type == "MultiPolygon":
        return geometry.geoms


def linear_ring_is_valid(ring):
    points = set([(x, y) for x, y in ring.coords])
    return len(points) >= 3


def fix_geometry(geometry):
    """Attempts to fix an invalid geometry (from https://goo.gl/nfivMh)"""
    try:
        return geometry.buffer(0)
    except ValueError:
        pass

    polygons = geom_as_list(geometry)

    fixed_polygons = list()
    for i, polygon in enumerate(polygons):
        if not linear_ring_is_valid(polygon.exterior):
            continue

        interiors = []
        for ring in polygon.interiors:
            if linear_ring_is_valid(ring):
                interiors.append(ring)

        fixed_polygon = Polygon(polygon.exterior, interiors)

        try:
            fixed_polygon = fixed_polygon.buffer(0)
        except ValueError:
            continue

        fixed_polygons.extend(geom_as_list(fixed_polygon))

    if len(fixed_polygons) > 0:
        return MultiPolygon(fixed_polygons)
    else:
        return None


def representative_point(polygon, mask, label, offset=None):
    """ Extract a representative point with integer coordinates from the given polygon and the label image.
    Parameters
    ----------
    polygon: Polygon
        A polygon
    mask: ndarray
        The label mask from which the polygon was generated
    label: int
        The label associated with the polygon
    offset: tuple
        An (x, y) offset that was applied to polygon

    Returns
    -------
    point: tuple
        The representative point (x, y)
    """
    if offset is None:
        offset = (0, 0)
    rpoint = polygon.representative_point()
    h, w = mask.shape[:2]
    x = clamp(int(rpoint.x) - offset[0], 0, w - 1)
    y = clamp(int(rpoint.y) - offset[1], 0, h - 1)

    # check if start point is withing polygon
    if mask[y, y] == label:
        return x, y

    # circle around central pixel with at most 9 pixels radius
    direction = 1
    for i in range(1, 10):
        # -> x
        for j in range(0, i):
            x += direction
            if 0 <= x < w and mask[y, x] == label:
                return x, y

        # -> y
        for j in range(0, i):
            y += direction
            if 0 <= y < h and mask[y, x] == label:
                return x, y

        direction *= -1

    raise ValueError("could not find a representative point for pol")


def mask_to_objects_2d(mask, background=0, offset=None):
    """Convert 2D (binary or label) mask to polygons. Generates borders fitting in the objects.

    Parameters
    ----------
    mask: ndarray
        2D mask array. Expected shape: (height, width).
    background: int
        Value used for encoding background pixels.
    offset: tuple (optional, default: None)
        (x, y) coordinate offset to apply to all the extracted polygons.

    Returns
    -------
    extracted: list of AnnotationSlice
        Each object slice represent an object from the image. Fields time and depth of AnnotationSlice are set to None.
    """
    if mask.ndim != 2:
        raise ValueError("Cannot handle image with ndim different from 2 ({} dim. given).".format(mask.ndim))
    if offset is None:
        offset = (0, 0)
    exclusion = np.logical_not(mask == background)
    affine = Affine(1, 0, offset[0], 0, 1, offset[1])
    slices = list()
    for gjson, label in shapes(mask.copy(), mask=exclusion, transform=affine):
        polygon = shape(gjson)
        if not polygon.is_valid:  # attempt to fix
            polygon = fix_geometry(polygon)
        if not polygon.is_valid:  # could not be fixed
            continue
        slices.append(AnnotationSlice(polygon=polygon, label=int(label)))
    return slices


def mask_to_objects_3d(mask, background=0, offset=None, assume_unique_labels=False, time=False):
    """Convert a 3D or 2D+t (binary or label) mask to polygon slices.

    Parameters
    ----------
    mask: ndarray
        3D mask array. Expected shape: (width, height, depth|time).
    background: int
        Value used for encoding background pixels.
    offset: tuple (optional, default (0, 0, 0))
        A (x, y, z) offset to apply to all the detected objects.
    assume_unique_labels: bool
        True if each objects is encoded with a unique label in the mask.
    time: bool
        True if the image is a 2D+t volume, false if it is a 3D volume.
    Returns
    -------
    objects: list
        Lists all the extracted objects. Each object is provided as a list of slices (i.e. python object of type
        AnnotationSlice). A slice consists of a 2D polygon, a label and the depth or time at which it was taken.
        If time is True, the field depth of AnnotationSlice is set to None. Otherwise, the field time is set to None.
    """
    if mask.ndim != 3:
        raise ValueError("Cannot handle image with ndim different from 3 ({} dim. given).".format(mask.ndim))
    if offset is None:
        offset = (0, 0, 0)

    label_img = mask
    if not assume_unique_labels:
        label_img = label_fn(mask, connectivity=2, background=background)

    # extract slice per slice
    depth = mask.shape[-1]
    offset_xy = offset[:2]
    offset_z = offset[-1]
    objects = defaultdict(list)  # maps object label with list of slices (as object_3d_type objects)
    image_box = box(offset[0], offset[1], offset[0] + mask.shape[1] - 1, offset[1] + mask.shape[1] - 1)
    for d in range(depth):
        slice_objects = mask_to_objects_2d(label_img[:, :, d].copy(), background, offset=offset_xy)
        for slice_object in slice_objects:
            label = slice_object.label
            if not assume_unique_labels:
                x, y = representative_point(slice_object.polygon, label_img, slice_object.label, offset)
                label = mask[y, x]
            objects[label].append(AnnotationSlice(
                polygon=image_box.intersection(slice_object.polygon),  # to filter part of annot. outside of the mask
                label=label,
                depth=d + offset_z if not time else None,
                time=d + offset_z if time else None
            ))
    return list(objects.values())


def mask_to_objects_3dt(mask, background=0, offset=None):
    """Convert a 3D+t label mask to polygon slices.

    Parameters
    ----------
    mask: ndarray
        4D mask array. Expected shape: (time, height, width, depth).
    background: int
        Value used for encoding background pixels.
    offset: tuple (optional, default (0, 0, 0, 0))
        A (t, x, y, z) offset to apply to all the detected objects.

    Returns
    -------
    objects: list
        Lists all the extracted objects in a three-level list. Third level lists the slices of an object at a given
        timestep. The second level lists all the timesteps at which an object appears and the first level lists all the
        objects.
        E.g.:
        - P_i_j_k denotes the polygon in the kth slice at the jth time for object i
        - t_i_j denotes the jth timestep for object i
        - s_i_j_k denotes the kth slice of object i at the jth time
        - label_i denotes the label of the object i

        [ # objects
            [ # timesteps (object 1)
                [ # slices (object 1, timestep 1)
                    AnnotationSlice(P_1_1_1, t_1_1, s_1_1_1, label_1),
                    AnnotationSlice(P_1_1_2, t_1_1, s_1_1_2, label_1),
                 ...],
                [ # slices (object 1, timestep 2)
                    AnnotationSlice(P_1_2_1, t_1_2, s_1_2_1, label_1),
                    AnnotationSlice(P_1_2_2, t_1_2, s_1_2_2, label_1),
                 ...],
                ...
            ],
            [ # timesteps (object 2)
                [ # slices (object 2, timestep 1)
                    AnnotationSlice(P_2_1_1, t_2_1, s_2_1_1, label_2),
                    AnnotationSlice(P_2_1_2, t_2_1, s_2_1_2, label_2),
                 ...],
                [ # slices (object 2, timestep 2)
                    AnnotationSlice(P_2_2_1, t_2_2, s_2_2_1, label_2),
                    AnnotationSlice(P_2_2_2, t_2_2, s_2_2_2, label_2),
                 ...],
                ...
            ]
        ]

    Notes
    -----
    Each object should be encoded with the same label over time.
    """
    if mask.ndim != 4:
        raise ValueError("Cannot handle image with ndim different from 4 ({} dim. given).".format(mask.ndim))
    duration = mask.shape[0]
    offset_xyz = offset[1:]
    offset_t = offset[0]
    objects = defaultdict(list)
    for t in range(duration):
        time_objects = mask_to_objects_3d(
            mask,
            background=background,
            offset=offset_xyz,
            assume_unique_labels=True,
            time=False
        )
        for time_slices in time_objects:
            label = time_slices[0].label
            objects[label].append(AnnotationSlice(
                polygon=s.polygon,
                label=s.label,
                depth=s.depth,
                time=t + offset_t
            ) for s in time_slices)
    return objects.values()
