import os
import sys

import numpy as np
import imageio
from cytomine import CytomineJob
from cytomine.models import Annotation, ImageInstance, ImageSequenceCollection, AnnotationCollection

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


def annotation_from_slice(slice: AnnotationSlice, id_image, image_height, id_project, label=None, upload_group_id=False):
    parameters = {
        "location": affine_transform(slice.polygon, [1, 0, 0, -1, 0, image_height]).wkt,
        "id_image": id_image,
        "id_project": id_project
    }
    if upload_group_id:
        parameters["property"] = [{"key": "ANNOTATION_GROUP_ID", "value": slice.label if label is None else label}]
    return Annotation(**parameters)


def get_image_seq_info(image_group):
    image_sequences = ImageSequenceCollection().fetch_with_filter("imagegroup", image_group.id)
    height = ImageInstance().fetch(image_sequences[0].image).height
    return {iseq.zStack: iseq.image for iseq in image_sequences}, height


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

    collection = AnnotationCollection()
    if data.ndim == 2:
        slices = mask_to_objects_2d(data)
        collection.extend([annotation_from_slice(s, in_image.id, in_image.height, project_id) for s in slices])
    elif data.ndim == 3:
        # in this case `in_image` is actually an ImageGroup
        slices = mask_to_objects_3d(np.moveaxis(data, 0, 2), background=0, assume_unique_labels=True)
        depth_to_image, height = get_image_seq_info(in_image)

        collection.extend([
            annotation_from_slice(
                slice=s, id_image=depth_to_image[s.depth],
                image_height=height, id_project=project_id,
                label=obj_id, upload_group_id=upload_group_id
            ) for obj_id, obj in enumerate(slices) for s in obj
        ])
    return collection


def extract_annotations_objdet(out_path, in_image, project_id, is_csv=False, generate_mask=False, in_path=None,
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
        mask = imread(path, is_2d=is_2d)

        if mask.ndim == 2:
            points = mask_to_points_2d(mask)
            collection.extend([annotation_from_slice(s, in_image.id, in_image.height, project_id) for s in points])
        elif mask.ndim == 3:
            points = mask_to_objects_3d(mask, time=False)
            depth_to_image, height = get_image_seq_info(in_image)

            collection.extend([
                annotation_from_slice(
                    slice=s, id_image=depth_to_image[s.depth],
                    image_height=height, id_project=project_id,
                    label=obj_id, upload_group_id=upload_group_id
                ) for obj_id, obj in enumerate(points) for s in obj
            ])

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

    collection = AnnotationCollection()
    if data.ndim == 2:
        slices = skeleton_mask_to_objects_2d(data)
        collection.extend([annotation_from_slice(s, in_image.id, in_image.height, project_id) for s in slices])
    elif data.ndim == 3:
        # in this case `in_image` is actually an ImageGroup
        slices = skeleton_mask_to_objects_3d(np.moveaxis(data, 0, 2), background=0, assume_unique_labels=True)
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


