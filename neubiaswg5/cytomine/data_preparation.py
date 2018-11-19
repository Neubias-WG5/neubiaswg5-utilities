import os
from cytomine import CytomineJob
from cytomine.models import ImageInstanceCollection, ImageGroupCollection

from neubiaswg5 import CLASS_OBJSEG


def makedirs_ifnotexists(folder):
    """Create folder if not exists"""
    if not os.path.exists(folder):
        os.makedirs(folder)


def prepare_objseg_data(cj: CytomineJob, in_path, gt_path, gt_suffix="_lbl", nodownload=False, is_2d=True):
    """Prepare data for ObjSeg problemclass
    Download input and ground truth images (if nodownload is false)
    """
    if nodownload:
        in_images = [os.path.join(in_path, f) for f in os.listdir(in_path)]
        gt_images = [os.path.join(gt_path, f) for f in os.listdir(gt_path)]
        return in_images, gt_images

    collection_class = ImageInstanceCollection if is_2d else ImageGroupCollection

    cj.job.update(progress=1, statusComment="Downloading images (to {})...".format(in_path))
    images = collection_class().fetch_with_filter("project", cj.parameters.cytomine_id_project)
    in_images = [i for i in images if gt_suffix not in i.originalFilename]
    gt_images = [i for i in images if gt_suffix in i.originalFilename]

    for input_image in in_images:
        input_image.download(os.path.join(in_path, "{id}.tif"))

    for gt_image in gt_images:
        related_name = gt_image.originalFilename.replace(gt_suffix, '')
        related_image = [i for i in in_images if related_name == i.originalFilename]
        if len(related_image) == 1:
            gt_image.download(os.path.join(gt_path, "{}.tif".format(related_image[0].id)))

    return in_images, gt_images


def prepare_data(problemclass, cj: CytomineJob, gt_suffix="_lbl", base_path=None, nodownload=False, infolder="in",
                 outfolder="out", gtfolder="ground_truth", tmp_folder="tmp", is_2d=True, **kwargs):
    """Prepare data from parameters.

    If nodownload is false, creates four folders in `base_path`:
        - `base_path`/`infolder`: input data & images
        - `base_path`/`gtfolder`: ground truth data & images
        - `base_path`/`outfolder`: output data & images
        - `base_path`/`tmp_folder`: tmp data

    If nodownload is true, working folders (except tmp) are considered existing and are:
        - `infolder`: input data & images
        - `gtfolder`: ground truth data & images
        - `outfolder`: output data & images
        - `base_path`/`tmp_folder`: tmp data (this one is created whatever the value of nodownload)

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
    nodownload: bool
        True if data shouldn't be downloaded
    infolder: str
        Name of folder for input data
    outfolder: str
        Name of folder for output data
    gtfolder: str
        Name of folder for ground truth data
    tmp_folder: str
        Name for temporary data folder
    is_2d: bool
        True if the problem is a 2d one, False otherwise (3D, 4D, 3D+t).

    Returns
    -------
    in_data: list
        List of input data. Can be a list of ImageInstance, ImageGroup, strings...
        If nodownload is true, simply a list of absolute path to the input images.
    gt_images: list
        List of input data. Can be a list of ImageInstance, ImageGroup,...
        If nodownload is true, simply a list of absolute path to the ground truth images (in the same order as
        in_data).
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
    tmp_path = os.path.join(working_path, tmp_folder)
    in_path = infolder if nodownload else os.path.join(working_path, infolder)
    out_path = outfolder if nodownload else os.path.join(working_path, outfolder)
    gt_path = gtfolder if nodownload else os.path.join(working_path, gtfolder)

    # create directories
    if not nodownload:
        makedirs_ifnotexists(in_path)
        makedirs_ifnotexists(out_path)
        makedirs_ifnotexists(gt_path)

    makedirs_ifnotexists(tmp_path)

    if problemclass == CLASS_OBJSEG:
        in_data, gt_data = prepare_objseg_data(cj, in_path, gt_path, is_2d=is_2d, gt_suffix=gt_suffix, nodownload=nodownload)
    else:
        raise ValueError("Unknown problemclass '{}'.".format(problemclass))

    return in_data, gt_data, in_path, gt_path, out_path, tmp_path
