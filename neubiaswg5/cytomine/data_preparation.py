import os
from cytomine import CytomineJob
from cytomine.models import ImageInstanceCollection


def prepare_objseg_data(cj: CytomineJob, in_path, gt_path, gt_suffix="_lbl"):
    """Prepare data for ObjSeg problemclass
    Download input and ground truth images
    """
    cj.job.update(progress=1, statusComment="Downloading images (to {})...".format(in_path))
    image_instances = ImageInstanceCollection().fetch_with_filter("project", cj.parameters.cytomine_id_project)
    in_images = [i for i in image_instances if gt_suffix not in i.originalFilename]
    gt_images = [i for i in image_instances if gt_suffix in i.originalFilename]

    for input_image in in_images:
        input_image.download(os.path.join(in_path, "{id}.tif"))

    for gt_image in gt_images:
        related_name = gt_image.originalFilename.replace(gt_suffix, '')
        related_image = [i for i in in_images if related_name == i.originalFilename]
        if len(related_image) == 1:
            gt_image.download(os.path.join(gt_path, "{}.tif".format(related_image[0].id)))

    return in_images, gt_images


def prepare_data(problemclass, cj: CytomineJob, gt_suffix="_lbl", base_path=None, in_folder="in", out_folder="out",
                 gt_folder="ground_truth", tmp_folder="tmp"):
    """Prepare data from parameters.
    Creates four folders in `base_path`:
        - `base_path`/`in_folder`: input data & images
        - `base_path`/`gt_folder`: ground truth data & images
        - `base_path`/`out_folder`: output data & images
        - `base_path`/`tmp_folder`: tmp data

    Parameters
    ----------
    problemclass: str
        One of the problemclass
    cj: CytomineJob
        The cytomine job instance (including parameters)
    gt_suffix: str
        Ground truth images suffix
    base_path: str
        Base path for data download
    in_folder: str
        Name of folder for input data
    out_folder: str
        Name of folder for output data
    gt_folder: str
        Name of folder for ground truth data
    tmp_folder: str
        Name for temporary data folder

    Returns
    -------
    in_data: list
        List of input data. Can be a list of ImageInstance, ImageGroup,...
    gt_images: list
        List of input data. Can be a list of ImageInstance, ImageGroup,...
    in_path: str
        Full path to input data folder
    gt_path: str
        Full path to ground truth data folder
    out_path: str
        Full path to output data folder
    tmp_path: str
        Full path to tmp data folder
    """
    if base_path is None:
        base_path = "{}".format(os.getenv("HOME"))
    working_path = os.path.join(base_path, str(cj.job.id))
    in_path = os.path.join(working_path, in_folder)
    out_path = os.path.join(working_path, out_folder)
    gt_path = os.path.join(working_path, gt_folder)
    tmp_path = os.path.join(working_path, tmp_folder)

    if not os.path.exists(working_path):
        os.makedirs(working_path)
        os.makedirs(in_path)
        os.makedirs(out_path)
        os.makedirs(gt_path)
        os.makedirs(tmp_path)

    if problemclass == "ObjSeg":
        in_data, gt_data = prepare_objseg_data(cj, in_path, gt_path, gt_suffix=gt_suffix)
    else:
        raise ValueError("Unknown problemclass '{}'.".format(problemclass))

    return in_data, gt_data, in_path, gt_path, out_path, tmp_path
