import os
import shutil
from pathlib import Path

from cytomine import CytomineJob
from cytomine.models import ImageInstanceCollection, ImageGroupCollection, AttachedFileCollection, AttachedFile

from neubiaswg5 import CLASS_OBJTRK, CLASS_LOOTRC, CLASS_TRETRC
from neubiaswg5.helpers.util import default_value, makedirs_ifnotexists


def get_image_name(image, is_2d=True):
    """
    Parameters
    ----------
    image: ImageInstance|ImageGroup
        An image group
    is_2d: bool
        True if 2d then image is an ImageInstance. Otherwise 3d, then image is a ImageGroup.
    Returns
    -------
    """
    if is_2d:
        return image.originalFilename
    else:
        return image.name


def get_file_extension(path):
    """Return the extension of the file (with .) for the given path. If there is no extension returns an empty string"""
    filename = os.path.basename(path)
    if "." not in filename:
        return ""
    else:
        return ".{}".format(filename.rsplit(".", 1)[-1])


def get_file_name(path):
    """Return the filename without extension"""
    return os.path.basename(path).rsplit(".", 1)[0]


def download_images(nj, in_path, gt_path, gt_suffix="_lbl", do_download=False, is_2d=True):
    """
    If do_download is true: download input and ground truth images to in_path and gt_path respectively, and return the
    corresponding ImageInstance or ImageGroup objects.
    If do_download is false: list and return images from folders in_path and gt_path

    Parameters
    ----------
    nj: NeubiasJob
        A neubias job
    in_path: str
        Path for input images
    gt_path: str
        Path for ground truth images
    gt_suffix: str
        A suffix for ground truth images filename
    do_download: bool
        True for actually downloading the image, False for getting them from in_path and gt_path
    is_2d: bool
        True for 2d images

    Returns
    -------
    in_images: iterable
        subtype: ImageInstance|ImageGroup|str
        Input images
    gt_images: iterable
        subtype: ImageInstance|ImageGroup|str
        Ground truth images
    """
    if not do_download:
        in_images = [os.path.join(in_path, f) for f in os.listdir(in_path)]
        gt_images = [os.path.join(gt_path, f) for f in os.listdir(gt_path)]
        return in_images, gt_images

    collection_class = ImageInstanceCollection if is_2d else ImageGroupCollection

    nj.job.update(progress=1, statusComment="Downloading images (to {})...".format(in_path))
    images = collection_class().fetch_with_filter("project", nj.parameters.cytomine_id_project)
    in_images = [i for i in images if gt_suffix not in get_image_name(i, is_2d=is_2d)]
    gt_images = [i for i in images if gt_suffix in get_image_name(i, is_2d=is_2d)]

    for input_image in in_images:
        input_image.download(os.path.join(in_path, "{id}.tif"))

    for gt_image in gt_images:
        related_name = get_image_name(gt_image, is_2d=is_2d).replace(gt_suffix, '')
        related_image = [i for i in in_images if related_name == get_image_name(i, is_2d=is_2d)]
        if len(related_image) == 1:
            gt_image.download(os.path.join(gt_path, "{}.tif".format(related_image[0].id)))

    return in_images, gt_images


def download_attached(inputs, path, suffix="_attached", do_download=False):
    """
    Download the most recent attached file for each input.
    If do_download is False, then the attached file must have the same name (without extension) as the corresponding
    input file plus the suffix.
    """
    new_inputs = list()

    # if do_download is False, need to scan for existing attached files
    existing_files = {get_file_name(f) for f in os.listdir(path) if suffix in f}
    existing_extensions = {get_file_name(f): get_file_extension(f) for f in os.listdir(path) if suffix in f}

    # get path for all inputs
    for image in inputs:
        if do_download:
            # extract most recent file
            files = AttachedFileCollection(image).fetch()
            current_file = sorted(files.data(), key=lambda f: int(f.created), reverse=True)[0]

            # download the last file
            extension = get_file_extension(current_file.filename)
            attached_path = os.path.join(path, "{}{}{}".format(image.id, suffix, extension))
            current_file.download(attached_path)
        else:
            image_name = os.path.basename(image).rsplit(".")[0]
            attached_name = "{}".format(image_name, suffix)
            if attached_name not in existing_files:
                raise FileNotFoundError("Missing attached file for input image '{}'.".format(image))
            attached_path = os.path.join(path, "{}{}".format(attached_name, existing_extensions[attached_name]))
        new_inputs.append((image, attached_path))
    return new_inputs


def prepare_data(problemclass, nj, gt_suffix="_lbl", base_path=None, do_download=False, infolder=None,
                 outfolder=None, gtfolder=None, tmp_folder="tmp", is_2d=True, **kwargs):
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
    nj: CytomineJob|NeubiasJob
        A CytomineJob or NeubiasJob instance.
    gt_suffix: str
        Ground truth images suffix
    base_path: str
        Base path for data download. Defaults to the '$HOME/{nj.job.id}/'.
    do_download: bool
        True if data should be downloaded.
    infolder: str|None
        Full path of the folder for input data. If None, defaults to '`base_path`/in'.
    outfolder: str|None
        Full path of the folder for output data. If None, defaults to '`base_path`/out'.
    gtfolder: str|None
        Full path of the folder for ground truth data. If None, defaults to '`base_path`/ground_truth'.
    tmp_folder: str
        Name (not the path) for temporary data folder.
    is_2d: bool
        True if the problem is a 2d one, False otherwise (3D, 4D, 3D+t).
    kwargs: dict
        For CLASS_TRETRC:
            - suffix: suffix in the filename for attached files (by default "_attached")
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
    # get path
    base_path = default_value(base_path, Path.home())
    working_path = os.path.join(base_path, str(nj.job.id))
    in_path = default_value(infolder, os.path.join(working_path, "in"))
    out_path = default_value(outfolder, os.path.join(working_path, "out"))
    gt_path = default_value(gtfolder, os.path.join(working_path, "ground_truth"))
    tmp_path = os.path.join(working_path, tmp_folder)

    # create directories
    makedirs_ifnotexists(in_path)
    makedirs_ifnotexists(out_path)
    makedirs_ifnotexists(gt_path)
    makedirs_ifnotexists(tmp_path)

    # in all cases download input and gt
    in_data, gt_data = download_images(nj, in_path, gt_path, is_2d=is_2d, gt_suffix=gt_suffix, do_download=do_download)

    # download additional data
    if problemclass == CLASS_TRETRC:
        suffix = kwargs.get("suffix", "_attached")
        in_data = download_attached(in_data, in_path, suffix=suffix, do_download=do_download)
        gt_data = download_attached(gt_data, gt_path, suffix=suffix, do_download=do_download)
    elif problemclass == CLASS_OBJTRK:
        raise NotImplementedError("Problemclass '{}' needs additional data. Download of this "
                                  "data hasn't been implemented yet".format(problemclass))

    return in_data, gt_data, in_path, gt_path, out_path, tmp_path
