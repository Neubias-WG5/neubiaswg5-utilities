import os
from pathlib import Path

from cytomine import CytomineJob
from cytomine.models import ImageInstanceCollection, ImageGroupCollection, AttachedFileCollection

from neubiaswg5 import CLASS_OBJTRK, CLASS_TRETRC
from neubiaswg5.helpers.util import default_value, makedirs_ifnotexists, NeubiasImageInstance, NeubiasImageGroup, \
    NeubiasFilepath, NeubiasAttachedFile, split_filename


SUPPORTED_MULTI_EXTENSION = ["ome.tif"]


def download_images(nj, in_path, gt_path, gt_suffix="_lbl", do_download=False, is_2d=True, ignore_missing_gt=False):
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
    ignore_missing_gt: bool
        If False, an exception is raised when a ground truth images is missing. Otherwise, just skip download

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
        in_images = [NeubiasFilepath(os.path.join(in_path, f)) for f in os.listdir(in_path)]
        gt_images = [NeubiasFilepath(os.path.join(gt_path, f)) for f in os.listdir(gt_path)]
        return in_images, gt_images

    collection_class = ImageInstanceCollection if is_2d else ImageGroupCollection
    input_class = NeubiasImageInstance if is_2d else NeubiasImageGroup

    filename_pattern = "{id}.tif"
    nj.job.update(progress=1, statusComment="Downloading images (to {})...".format(in_path))
    images = collection_class().fetch_with_filter("project", nj.parameters.cytomine_id_project)
    images = [input_class(i, name_pattern=filename_pattern) for i in images]
    gt_images_to_process = [input_class(i.object) for i in images if gt_suffix in i.original_filename]
    in_images_to_process = {i.original_filename: i for i in images if gt_suffix not in i.original_filename}

    in_images = list()
    gt_images = list()
    for gt in gt_images_to_process:
        name, ext = gt.original_filename.rsplit(gt_suffix + ".", 1)
        in_filename = name + "." + ext
        if in_filename not in in_images_to_process:
            raise ValueError("Missing input image '{}' for ground truth image '{}' (id:{}).".format(in_filename, gt.original_filename, gt.object.id))
        in_object = in_images_to_process[in_filename].object
        in_images.append(input_class(in_object, in_path, filename_pattern))
        gt_images.append(input_class(gt.object, gt_path, filename_pattern.format(id=in_object.id)))

    # check that all input images have a ground truth
    if len(in_images_to_process) != len(in_images):
        diff = set(in_images_to_process.keys()).difference([i.original_filename for i in in_images])
        for in_image in diff:
            if not ignore_missing_gt:
                raise ValueError("Missing ground truth for input image '{}' (id:{}).".format(in_image.original_filename, in_image.object.id))
            in_images.append(input_class(in_image.object, in_path, filename_pattern))

    for img in (in_images + gt_images):
        img.object.download(img.filepath, parent=True, override=False)

    return in_images, gt_images


def download_attached(inputs, path, suffix="_attached", do_download=False):
    """
    Download the most recent attached file for each input.
    If do_download is False, then the attached file must have the same name (without extension) as the corresponding
    input file plus the suffix.
    """
    # if do_download is False, need to scan for existing attached files
    existing_files, existing_extensions = set(), dict()
    for f in os.listdir(path):
        if suffix not in f:
            continue
        name, ext = split_filename(f, multi_extension=SUPPORTED_MULTI_EXTENSION)
        existing_files.add(name)
        existing_extensions[name] = ext

    # get path for all inputs
    for in_image in inputs:
        image = in_image.object
        if do_download:
            # extract most recent file
            files = AttachedFileCollection(image).fetch()
            most_recent = sorted(files.data(), key=lambda f: int(f.created), reverse=True)[0]

            # download the last file
            attached_file = NeubiasAttachedFile(most_recent, path, name_pattern="{filename}")
            most_recent.download(attached_file.filepath)
        else:
            image_name = os.path.basename(image).rsplit(".")[0]
            attached_name = "{}".format(image_name, suffix)
            if attached_name not in existing_files:
                raise FileNotFoundError("Missing attached file for input image '{}'.".format(image))
            attached_file = NeubiasFilepath(os.path.join(
                path, "{}{}".format(attached_name, existing_extensions[attached_name])
            ))
        in_image.attached.append(attached_file)


def prepare_data(problemclass, nj, gt_suffix="_lbl", base_path=None, do_download=False, infolder=None,
                 outfolder=None, gtfolder=None, tmp_folder="tmp", is_2d=True, ignore_missing_gt=False, **kwargs):
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
    ignore_missing_gt: bool
        If False, an exception is raised when a ground truth images is missing. Otherwise, just skip download
    kwargs: dict
        For CLASS_TRETRC:
            - suffix: suffix in the filename for attached files (by default "_attached")
    Returns
    -------
    in_data: list
        List of input data. If `is_2d` then usually a list of `ImageInstance`, otherwise a list of `ImageGroup`.
        If `--nodownload` (i.e. `do_download` is True) was used, then usually a list of absolute path to the input
        images. For CLASS_TRETRC, a list of tuple containing the input as first item and attached file path as second
        item.
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
    in_data, gt_data = download_images(nj, in_path, gt_path, is_2d=is_2d, gt_suffix=gt_suffix,
                                       do_download=do_download, ignore_missing_gt=ignore_missing_gt)

    # download additional data
    if problemclass == CLASS_TRETRC:
        suffix = kwargs.get("suffix", "_attached")
        download_attached(in_data, gt_path, suffix=suffix, do_download=do_download)
    elif problemclass == CLASS_OBJTRK:
        raise NotImplementedError("Problemclass '{}' needs additional data. Download of this "
                                  "data hasn't been implemented yet".format(problemclass))

    return in_data, gt_data, in_path, gt_path, out_path, tmp_path
