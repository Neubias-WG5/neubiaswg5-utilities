import math
import os
import warnings
from pathlib import Path

from cytomine import CytomineJob
from cytomine.models import ImageInstanceCollection, AttachedFileCollection
from sldc import TileTopology, DefaultTileBuilder

from neubiaswg5 import CLASS_OBJTRK, CLASS_TRETRC
from neubiaswg5.helpers.data_upload import imwrite
from neubiaswg5.helpers.util import default_value, makedirs_ifnotexists, NeubiasCytomineInput, \
    NeubiasFilepath, NeubiasAttachedFile, split_filename, NeubiasSldcImage, NeubiasTile

SUPPORTED_MULTI_EXTENSION = ["ome.tif"]


def extract_batch(data, key=None, batch_size=0, batch_id=0):
    """Extract a batch from a list of data

    Parameters
    ----------
    data: list
        List of data from which to extract the batch
    key: callable
        A function transforming an element of data into a sortable key.
    batch_size: int
        Size of a batch
    batch_id: int
        Batch identifier
    """
    if batch_size == 0:
        return data
    if key is None:
        key = lambda i: i
    start = batch_id * batch_size
    end = start + batch_size
    return list(sorted(data, key=key))[start:end]


def download_images(nj, in_path, gt_path, gt_suffix="_lbl", ignore_missing_gt=False, batch_id=0, batch_size=0):
    """
    If do_download is true: download input and ground truth images to in_path and gt_path respectively, and return the
    corresponding ImageInstance objects.
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
    ignore_missing_gt: bool
        If False, an exception is raised when a ground truth images is missing. Otherwise, just skip download
    batch_size: int
        The size of a batch (default value 0 means no batching)
    batch_id: int
        The index of the batch of input images to download

    Returns
    -------
    in_images: iterable
        Input images
    gt_images: iterable
        Ground truth images
    """
    if not nj.flags["do_download"]:
        # NOTE: there might be a mismatch between batches resulting from local or BIAFLOWS-connected execution
        in_images = [NeubiasFilepath(os.path.join(in_path, f)) for f in os.listdir(in_path)]
        gt_images = [NeubiasFilepath(os.path.join(gt_path, f)) for f in os.listdir(gt_path)]
        in_images = extract_batch(in_images, key=lambda i: i.filename, batch_size=batch_size, batch_id=batch_id)
        gt_images = extract_batch(gt_images, key=lambda i: i.filename, batch_size=batch_size, batch_id=batch_id)
        return in_images, gt_images

    filename_pattern = "{id}.tif"
    nj.job.update(progress=1, statusComment="Downloading images (to {})...".format(in_path))
    images = ImageInstanceCollection().fetch_with_filter("project", nj.parameters.cytomine_id_project)
    images = [NeubiasCytomineInput(i, name_pattern=filename_pattern) for i in images]
    gt_images_to_process = [NeubiasCytomineInput(i.object) for i in images if gt_suffix in i.original_filename]
    in_images_to_process = {i.original_filename: i for i in images if gt_suffix not in i.original_filename}

    if batch_size > 0:
        # use lexical order of ground truth files to have consistent batches
        num_samples = len(gt_images_to_process)

        gt_images_to_process = extract_batch(
            gt_images_to_process,
            key=lambda l: l.original_filename,
            batch_size=batch_size,
            batch_id=batch_id
        )

        if num_samples > 0 and len(gt_images_to_process) == 0:
            warnings.warn("Empty batch ! You have activated batch processing but provided batch size ({}) and id ({}) "
                          "yield an empty batch (starting at index {} over {} image(s) total)".format(batch_size, batch_id, batch_id * batch_size, num_samples))

    in_images = list()
    gt_images = list()
    for gt in gt_images_to_process:
        name, ext = gt.original_filename.rsplit(gt_suffix + ".", 1)
        in_filename = name + "." + ext
        if in_filename not in in_images_to_process:
            raise ValueError("Missing input image '{}' for ground truth image '{}' (id:{}).".format(in_filename, gt.original_filename, gt.object.id))
        in_object = in_images_to_process[in_filename].object
        in_images.append(NeubiasCytomineInput(in_object, in_path, filename_pattern))
        gt_images.append(NeubiasCytomineInput(gt.object, gt_path, filename_pattern.format(id=in_object.id)))

    # check that all input images have a ground truth
    if (batch_size <= 0 and len(in_images_to_process) != len(in_images)) or (batch_size > 0 and batch_size != len(in_images)):
        to_process = {i.original_filename for i in in_images}
        for in_filename, in_image in in_images_to_process.items():
            if in_filename in to_process:
                continue
            if not ignore_missing_gt:
                raise ValueError("Missing ground truth for input image '{}' (id:{}).".format(in_image.original_filename, in_image.object.id))
            in_images.append(NeubiasCytomineInput(in_image.object, in_path, filename_pattern))

    for img in (in_images + gt_images):
        img.object.download(img.filepath, override=False)

    return in_images, gt_images


def download_attached(inputs, path, suffix="_attached", do_download=False, ignore_missing_gt=False):
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
        name, ext = f.rsplit(suffix + ".")
        existing_files.add(name)
        existing_extensions[name] = ext

    # get path for all inputs
    for in_image in inputs:
        image = in_image.object
        if do_download:
            # extract most recent file
            files = AttachedFileCollection(image).fetch()
            if len(files) == 0:
                if ignore_missing_gt:
                    continue
                raise ValueError("Missing ground truth attached file for input image '{}'.".format(in_image.filename))
            most_recent = sorted(files.data(), key=lambda f: int(f.created), reverse=True)[0]

            # download the last file
            in_name, _ = split_filename(in_image.filename)
            _, ext = split_filename(most_recent.filename)
            attached_file = NeubiasAttachedFile(most_recent, path, name_pattern=in_name + suffix + "." + ext)
            most_recent.download(attached_file.filepath)
        else:
            image_name = os.path.basename(image).rsplit(".")[0]
            attached_name = "{}".format(image_name, suffix)
            if attached_name not in existing_files:
                if ignore_missing_gt:
                    continue
                raise FileNotFoundError("Missing attached file for input image '{}'.".format(image))
            attached_file = NeubiasFilepath(os.path.join(
                path, "{}{}".format(attached_name, existing_extensions[attached_name])
            ))
        in_image.attached.append(attached_file)


def make_tiles(in_data, tile_path, tile_width=256, tile_height=256, tile_overlap=32):
    """Split input images into tiles

    Parameters
    ----------
    in_data: iterable
        The input images, to split into tiles
    tile_path: str
        Filepath where the tiles should be saved
    tile_width: int
        Tile width
    tile_height: int
        Tile height
    tile_overlap: int
        Tile overlap

    Returns
    -------
    tiles: iterable
        A list of all generated tiles
    """
    tiles = list()
    for in_image in in_data:
        sldc_image = NeubiasSldcImage(in_image, is_2d=True)  # support 2D-only for now
        tile_builder = DefaultTileBuilder()
        topology = sldc_image.tile_topology(tile_builder, max_width=tile_width, max_height=tile_height, overlap=tile_overlap)
        for tile in topology:
            neubias_tile = NeubiasTile(in_image, tile_path, tile)
            imwrite(neubias_tile.filepath, tile.np_image, is_2d=True)
            tiles.append(neubias_tile)
    return tiles


def prepare_data(problemclass, nj, gt_suffix="_lbl", base_path=None, do_download=False, infolder=None,
                 outfolder=None, gtfolder=None, tmp_folder="tmp", ignore_missing_gt=False, **kwargs):
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
    ignore_missing_gt: bool
        If False, an exception is raised when a ground truth images is missing. Otherwise, just skip download
    kwargs: dict
        For CLASS_TRETRC:
            - suffix: suffix in the filename for attached files (by default "_attached")
    Returns
    -------
    in_data: list
        List of input data. If `is_2d` then usually a list of `ImageInstance`.
        If `--nodownload` (i.e. `do_download` is True) was used, then usually a list of absolute path to the input
        images. For CLASS_TRETRC, a list of tuple containing the input as first item and attached file path as second
        item.
    gt_images: list
        List of input data. Can be a list of ImageInstance,...
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

    # when metrics are ignored, we can ignore ground truth download
    ignore_missing_gt = ignore_missing_gt or not kwargs.get("do_compute_metrics", True)

    # in all cases download input and gt
    in_data, gt_data = download_images(nj, in_path, gt_path, gt_suffix=gt_suffix, ignore_missing_gt=ignore_missing_gt,
                                       batch_size=nj.flags['batch_size'], batch_id=nj.flags['batch_id'])

    # download additional data
    if problemclass == CLASS_TRETRC or problemclass == CLASS_OBJTRK:
        suffix = kwargs.get("suffix", "_attached")
        download_attached(in_data, gt_path, suffix=suffix, do_download=do_download, ignore_missing_gt=ignore_missing_gt)

    # if tiling enabled, replace in_data with list of tiles
    if nj.flags["tiling"]:
        tile_path = default_value(nj.flags["tilefolder"], os.path.join(base_path, "tiles"))
        makedirs_ifnotexists(tile_path)
        # replace in_data with tiles
        in_path = tile_path
        in_data = make_tiles(
            in_data, tile_path,
            tile_height=nj.flags["tile_height"],
            tile_width=nj.flags["tile_width"],
            tile_overlap=nj.flags["tile_overlap"]
        )

    return in_data, gt_data, in_path, gt_path, out_path, tmp_path
