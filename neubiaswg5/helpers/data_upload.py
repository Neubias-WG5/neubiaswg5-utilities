import os
import sys

import numpy as np
import imageio
from cytomine import CytomineJob
from cytomine.models import Annotation, ImageInstance, ImageSequenceCollection, AnnotationCollection
from shapely.geometry import LineString

from neubiaswg5.exporter.mask_to_points import mask_to_points_3d
from neubiaswg5.problemclass import *
from neubiaswg5.exporter import mask_to_objects_2d, mask_to_objects_3d, AnnotationSlice, csv_to_points, \
    slices_to_mask, mask_to_points_2d, skeleton_mask_to_objects_2d, skeleton_mask_to_objects_3d
from shapely.affinity import affine_transform


def imread(path, is_2d=True, **kwargs):
    """wrapper for imageio.imread or imageio.volread"""
    if is_2d:
        return imageio.imread(path, **kwargs)
    else:
        return imageio.volread(path, **kwargs)


def imwrite(path, image, is_2d=True, **kwargs):
    """wrapper for imageio.imwrite or imageio.volwrite"""
    if is_2d:
        return imageio.imwrite(path, image, **kwargs)
    else:
        return imageio.volwrite(path, image, **kwargs)


def change_referential(p, height):
    return affine_transform(p, [1, 0, 0, -1, 0, height])


def get_group_id_dict(label):
    return {"key": "ANNOTATION_GROUP_ID", "value": label}


def annotation_from_slice(slice: AnnotationSlice, id_image, image_height, id_project, label=None, upload_group_id=False):
    """
    slice: AnnotationSlice
    id_image: int
    image_height: int
    id_project: int
    label: int
    upload_group_id: bool
    """
    parameters = {
        "location": change_referential(slice.polygon, image_height).wkt,
        "id_image": id_image,
        "id_project": id_project
    }
    if upload_group_id:
        parameters["property"] = [get_group_id_dict(slice.label if label is None else label)]
    return Annotation(**parameters)


def get_image_seq_info(image_group, time=False):
    image_sequences = ImageSequenceCollection().fetch_with_filter("imagegroup", image_group.id)
    height = ImageInstance().fetch(image_sequences[0].image).height
    return {iseq.zStack if not time else iseq.time: iseq.image for iseq in image_sequences}, height


def mask_convert(mask, in_image, project_id, mask_2d_fn, mask_3d_fn, upload_group_id=False):
    """Generic function to convert a mask into an annotation collection

    Parameters
    ----------
    mask: ndarray
    in_image: ImageInstance|ImageGroup
    project_id: int
    mask_2d_fn: callable
    mask_3d_fn: callable
    upload_group_id: bool

    Returns
    -------
    collection: AnnotationCollection
    """
    collection = AnnotationCollection()
    if mask.ndim == 2:
        slices = mask_2d_fn(mask)
        collection.extend([annotation_from_slice(s, in_image.id, in_image.height, project_id) for s in slices])
    elif mask.ndim == 3:
        slices = mask_3d_fn(mask)
        depth_to_image, height = get_image_seq_info(in_image)

        collection.extend([
            annotation_from_slice(
                slice=s, id_image=depth_to_image[s.depth],
                image_height=height, id_project=project_id,
                label=obj_id, upload_group_id=upload_group_id
            ) for obj_id, obj in enumerate(slices) for s in obj
        ])
    else:
        raise ValueError("Only supports 2D or 3D output images...")
    return collection


def extract_annotations_objseg(out_path, in_image, project_id, upload_group_id=False, is_2d=True, **kwargs):
    """
    Parameters
    ----------
    out_path: str
    in_image: ImageInstance|ImageGroup
    project_id: int
    upload_group_id: bool
        True for uploading annotation group id
    is_2d: bool
    kwargs: dict
    """
    file = "{}.tif".format(in_image.id)
    path = os.path.join(out_path, file)
    data = imread(path, is_2d=is_2d)

    return mask_convert(
        data, in_image, project_id,
        mask_2d_fn=mask_to_objects_2d,
        mask_3d_fn=lambda m: mask_to_objects_3d(np.moveaxis(m, 0, 2), background=0, assume_unique_labels=True),
        upload_group_id=upload_group_id
    )


def extract_annotations_objdet(out_path, in_image, project_id, is_csv=True, generate_mask=False, in_path=None,
                               result_file_suffix="_results.txt", has_headers=False, parse_fn=None, upload_group_id=False, is_2d=True, **kwargs):
    """
    Parameters:
    -----------
    out_path: str
    in_image: ImageInstance|ImageGroup
    project_id: int
    is_csv: bool
        True if the output data are stored in a csv file
    generate_mask: bool
        If result is in a CSV, True for generating a mask based on the points in the csv. Ignored if is_csv is False.
        The mask file is generated in out_path with the name "{in_image.id}.png".
    in_path: str
        The path where to find input images. Required when generate mask is present.
    result_file_suffix: str
        Suffix of the result filename (prefix being the image id).
    has_headers: bool
        True if the csv contains some headers (ignored if is_csv is False)
    parse_fn: callable
        A function for extracting coordinates from the csv file (already separated) line.
    upload_group_id: bool
        True for uploading annotation group id
    is_2d: bool
    kwargs: dict
    """
    file = str(in_image.id) + result_file_suffix
    path = os.path.join(out_path, file)

    collection = AnnotationCollection()
    if not os.path.isfile(path):
        print("No output file at '{}' for image with id:{}.".format(path, in_image.id), file=sys.stderr)
        return collection

    # whether the points are stored in a csv or a mask
    if is_csv:
        if parse_fn is None:
            raise ValueError("parse_fn shouldn't be 'None' when result file is a CSV.")
        points = csv_to_points(path, has_headers=has_headers, parse_fn=parse_fn)
        collection.extend([
            annotation_from_slice(slice, in_image.id, in_image.height, project_id)
            for slice in points
        ])

        if generate_mask:
            input_path = os.path.join(in_path, str(in_image.id) + "." + in_image.originalFilename.rsplit(".", 1)[1])
            mask = slices_to_mask(points, imread(input_path, is_2d=is_2d).shape)
            imwrite(os.path.join(out_path, "{}.tif".format(in_image.id)), mask, is_2d=is_2d)
    else:
        # points stored in a mask
        collection = mask_convert(
            imread(path, is_2d=is_2d), in_image, project_id,
            mask_2d_fn=mask_to_points_2d,
            mask_3d_fn=lambda m: mask_to_points_3d(np.moveaxis(m, 0, 2), time=False, assume_unique_labels=False),
            upload_group_id=upload_group_id
        )

    return collection


def extract_annotations_prttrk(out_path, in_image, project_id, upload_group_id=False, is_2d=False, **kwargs):
    """
    Parameters:
    -----------
    out_path: str
    in_image: ImageInstance
    project_id: int
    upload_group_id: bool
    is_2d: bool
    kwargs: dict
    """
    if is_2d:
        raise ValueError("Annotation extraction function called with is_2d=True: object tracking should be at least 3D")

    file = "{}.tif".format(in_image.id)
    path = os.path.join(out_path, file)
    data = imread(path, is_2d=is_2d)

    if data.ndim != 3:
        raise ValueError("Annotation extraction for object tracking does not support masks with more than 3 dims...")

    slices = mask_to_points_3d(np.moveaxis(data, 0, 2), time=True, assume_unique_labels=True)
    time_to_image, height = get_image_seq_info(in_image, time=True)

    collection = AnnotationCollection()
    for slice_group in slices:
        sorted_group = sorted(slice_group, key=lambda s: s.time)
        prev_line = []
        for _slice in sorted_group:
            if len(prev_line) == 0 or not prev_line[-1].equals(_slice.polygon):
                prev_line.append(_slice.polygon)

            if len(prev_line) == 1:
                polygon = _slice.polygon
            else:
                polygon = LineString(prev_line)

            annotation_params = {
                "location": change_referential(polygon, height).wkt,
                "id_image": time_to_image[_slice.time],
                "id_project": project_id
            }
            if upload_group_id:
                annotation_params["property"] = [get_group_id_dict(_slice.label)]
            collection.append(Annotation(**annotation_params))

    return collection


def extract_annotations_lootrc(out_path, in_image, project_id, upload_group_id=False, is_2d=True, **kwargs):
    """
    Parameters
    ----------
    out_path: str
    in_image: ImageInstance|ImageGroup
    project_id: int
    upload_group_id: bool
    is_2d: bool
    kwargs: dict
    """
    file = "{}.tif".format(in_image.id)
    path = os.path.join(out_path, file)
    data = imread(path, is_2d=is_2d)

    collection = mask_convert(
        data, in_image, project_id,
        mask_2d_fn=skeleton_mask_to_objects_2d,
        mask_3d_fn=lambda m: skeleton_mask_to_objects_3d(np.moveaxis(m, 0, 2), background=0, assume_unique_labels=True),
        upload_group_id=upload_group_id
    )
    return collection


def upload_data(problemclass, nj, inputs, out_path, monitor_params=None, do_download=False, do_export=False, is_2d=True, **kwargs):
    """Upload annotations or any other related results to the server.

    Parameters
    ----------
    problemclass: str
        The problem class
    nj: CytomineJob|NeubiasJob
        The CytomineJob or NeubiasJob object. Ignored if do_export is True.
    inputs: list
        Input data as returned by the prepare_data
    out_path: str
        Output path
    monitor_params: dict|None
        A dictionnary of parameters to be passed to the data upload loop monitor.
    do_download: bool
        True if data was downloaded
    do_export: bool
        True if results should be exported
    is_2d: bool
        True for 2D image, False for more than two dimensions.
    kwargs: dict
        Additional parameters for:
        * ObjDet/SptCnt: see function 'extract_annotations_objdet'
        * ObjSeg: see function 'extract_annotations_objseg'
    """
    if not do_export or not do_download:
        return
    if monitor_params is None:
        monitor_params = dict()

    if problemclass == CLASS_OBJSEG:
        extract_fn = extract_annotations_objseg
    elif problemclass == CLASS_OBJDET or problemclass == CLASS_SPTCNT:
        extract_fn = extract_annotations_objdet
    elif problemclass == CLASS_LOOTRC:
        extract_fn = extract_annotations_lootrc
    elif problemclass == CLASS_PRTTRK:
        extract_fn = extract_annotations_prttrk
    else:
        raise NotImplementedError("Upload data does not support problem class '{}' yet.".format(problemclass))

    # whether or not to upload a unique identifier as a property with each detected object
    upload_group_id = not is_2d or problemclass in {CLASS_OBJTRK, CLASS_PRTTRK}

    collection = AnnotationCollection()
    monitor_params["prefix"] = "Extract masks/points/... from output data"
    for image in nj.monitor(inputs, **monitor_params):
        collection.extend(extract_fn(out_path, image, nj.project.id, upload_group_id=upload_group_id, is_2d=is_2d, **kwargs))

    nj.job.update(statusComment="Upload extracted annotations (total: {})".format(len(collection)))
    collection.save()


