import json
import logging
import sys
import warnings
from argparse import ArgumentParser, Namespace

from cytomine.cytomine_job import _software_params_to_argparse, CytomineJob

from neubiaswg5.helpers.util import check_field


def get_discipline(nj, default):
    if hasattr(nj, "project"):
        return nj.project.disciplineShortName
    else:
        return default


class NeubiasParameter(object):
    """Partially compatible with SoftwareParameter in order work with _software_params_to_argparse"""
    def __init__(self, **parameters):
        """Input names of properties should match descriptor file"""
        self._name = check_field(parameters, "id", target="parameter descriptor.")
        target = "parameter '{}' descriptor.".format(parameters["id"])
        self._required = not check_field(parameters, "optional", target=target)
        self._defaultParamValue = parameters.get("default-value", None)
        self._type = check_field(parameters, "type", target=target)

    @property
    def name(self):
        return self._name

    @property
    def required(self):
        return self._required

    @property
    def type(self):
        return self._type

    @property
    def defaultParamValue(self):
        return self._defaultParamValue


class FakeUpdatableJob(object):
    @property
    def id(self):
        return "standalone"

    """A fake job that can be updated."""
    def update(self, progress=0, status="", statusComment="", **kwargs):
        print("Progress: {: <3d}% ... Status: {: >1} - '{}'".format(progress, str(status), statusComment))


class NeubiasJob(object):
    """Equivalent of CytomineJob but that can run without a Cytomine server. Can be used with a context manager"""
    def __init__(self, flags, parameters):
        self._parameters = parameters
        self._flags = flags

    @property
    def parameters(self):
        return self._parameters

    @property
    def flags(self):
        return self._flags

    @property
    def job(self):
        return FakeUpdatableJob()

    @staticmethod
    def from_cli(argv, **kwargs):
        """
        Returns a CytomineJob if interaction with Cytomine are needed. Otherwise, returns a minimal NeubiasJob
        that emulates a CytomineJob (e.g. parameters access).
        """
        # Parse CytomineJob constructor parameters
        flags_ap = ArgumentParser()
        flags_ap.add_argument("--nodownload", dest="old_do_download", action="store_false", required=False, help="DEPRECATED")
        flags_ap.add_argument("--noexport", dest="old_do_export", action="store_false", required=False, help="DEPRECATED")
        flags_ap.add_argument("--nometrics", dest="old_do_compute_metrics", action="store_false", required=False, help="DEPRECATED")
        flags_ap.add_argument("-l", "--local", dest="local", action="store_true", required=False,
                              help="Local execution, the workflow will never try to contact the server.")
        flags_ap.add_argument("-nd", "--no_download", dest="no_download", action="store_true", required=False,
                              help="Whether or not to download data from BIAFLOWS/local server. If raised, the absolute"
                                   " path path to the folder containing the input data should be provided through "
                                   "--infolder. Ignored if --local flag is raised.")
        flags_ap.add_argument("-nmu", "--no_metrics_upload", dest="no_metrics_upload", action="store_true", required=False,
                              help="Whether or not to upload the metrics to the BIAFLOWS/local server. If --no_download"
                                   " is raised but this flag is not, then the absolute path to the folder containing "
                                   "the ground truth data should be provided through --gtfolder. This flag is set to "
                                   "true if --no_metrics_computation is raised. Ignored if --local flag is raised.")
        flags_ap.add_argument("-nau", "--no_annotations_upload", dest="no_annotations_upload", action="store_true", required=False,
                              help="Whether or not to upload results annotations to the BIAFLOWS/local server. Ignored "
                                   "if --local flag is raised.")
        flags_ap.add_argument("-nmc", "--no_metrics_computation", dest="no_metrics_computation", action="store_true", required=False,
                              help="Whether or not to compute and display the metrics.")
        flags_ap.add_argument("--descriptor", dest="descriptor", default="/app/descriptor.json", required=False,
                              help="A path to a descriptor.json file. This file will be used to check parameters if the"
                                   " three 'no' flags are raised.")
        flags_ap.add_argument("--infolder", dest="infolder", default=None, required=False,
                              help="Absolute path to the container folder where the input data to be processed is "
                                   "stored. If not specified, a custom folder is created and used by the workflow.")
        flags_ap.add_argument("--outfolder", dest="outfolder", default=None, required=False,
                              help="Absolute path to the container folder where the output data will be generated."
                                   "If not specified, a custom folder is created and used by the workflow.")
        flags_ap.add_argument("--gtfolder", dest="gtfolder", default=None, required=False,
                              help="Absolute path to the container folder where the ground truth to be processed is "
                                   "stored. If not specified, a custom folder is created and used by the workflow.")
        flags_ap.add_argument("--batch_size", dest="batch_size", type=int, default=0, required=False,
                              help="The size of the batch of image to process. To use in conjunction with batch_id to "
                                   "process only a subset of images. Default value (=0) indicates that batching is "
                                   "disabled.")
        flags_ap.add_argument("--batch_id", dest="batch_id", type=int, default=0, required=False,
                              help="The index of the batch of input images to process. Value should be in range [0, B[ "
                                   "where B is the number of batches given the number of images in the project. Ignored"
                                   " if batching is disabled.")
        flags_ap.add_argument("-t", "--tiling", dest="tiling", action="store_true", required=False,
                              help="Specify to enable tiling (supported by ObjSeg 2D problems only)")
        flags_ap.add_argument("-tw", "--tile_width", dest="tile_width", type=int, default=256, required=False,
                              help="The maximum width of a tile (ignored if tiling is disabled, see --tiling).")
        flags_ap.add_argument("-th", "--tile_height", dest="tile_height", type=int, default=256, required=False,
                              help="The maximum height of a tile (ignored if tiling is disabled, see --tiling).")
        flags_ap.add_argument("-to", "--tile_overlap", dest="tile_overlap", type=int, default=32, required=False,
                              help="The overlap between two adjacent tiles (ignored if tiling is disabled, "
                                   "see --tiling).")
        flags_ap.add_argument("--tilefolder", dest="tilefolder", default=None, required=False,
                              help="Absolute path to the container folder where the tiled images is stored. If not "
                                   "specified, a custom folder is created and used by the workflow.")
        flags_ap.set_defaults(local=False, no_download=False, no_metrics_upload=False, no_annotations_upload=False,
                              no_metrics_computation=False, old_do_download=None, old_do_export=None, tiling=False,
                              old_do_compute_metrics=None)
        cli_flags, _ = flags_ap.parse_known_args(argv)

        def check_deprecated_flag(value, name, new):
            if value is not None:
                warnings.warn("Flag '{}' is deprecated, use '{}' instead.".format(name, new))

        check_deprecated_flag(cli_flags.old_do_download, "--nodownload", "--no_download")
        check_deprecated_flag(cli_flags.old_do_export, "--noexport", "--no_annotations_upload")
        check_deprecated_flag(cli_flags.old_do_compute_metrics, "--nometrics", "--no_metrics_upload")

        # rewrite flags for local switches
        flags = Namespace()
        flags.local = cli_flags.local or (cli_flags.no_annotations_upload and cli_flags.no_metrics_upload and cli_flags.no_download)
        flags.do_download = not (flags.local or cli_flags.no_download)
        flags.do_upload_annotations = not (flags.local or cli_flags.no_annotations_upload)
        flags.do_upload_metrics = not (flags.local or cli_flags.no_metrics_upload or cli_flags.no_metrics_computation)
        flags.do_compute_metrics = not cli_flags.no_metrics_computation
        flags.descriptor = cli_flags.descriptor
        flags.infolder = cli_flags.infolder
        flags.outfolder = cli_flags.outfolder
        flags.gtfolder = cli_flags.gtfolder
        flags.batch_size = cli_flags.batch_size
        flags.batch_id = cli_flags.batch_id
        flags.tiling = cli_flags.tiling
        flags.tile_width = cli_flags.tile_width
        flags.tile_height = cli_flags.tile_height
        flags.tile_overlap = cli_flags.tile_overlap
        flags.tilefolder = cli_flags.tilefolder

        if not flags.do_download and flags.infolder is None:
            raise ValueError("When --no_download is raised, an --infolder should be specified.")
        if flags.do_compute_metrics and not flags.do_download and flags.gtfolder is None:
            raise ValueError("When --no_download is raised but metrics have to be computed, a --gtfolder should be "
                             "specified.")

        # Cytomine is needed if at least one flag is not raised.
        if not flags.local:
            cj = CytomineJob.from_cli(argv, **kwargs)
            cj.flags = vars(flags)  # append the flags
            return cj

        with open(flags.descriptor, "r") as file:
            software_desc = json.load(file)

            if software_desc is None or "inputs" not in software_desc:
                raise ValueError("Cannot read 'descriptor.json' or missing 'inputs' in JSON")

            parameters = [NeubiasParameter(**param) for param in software_desc["inputs"]]

            # exclude all cytomine parameters
            # hypothesis: such parameters always start with 'cytomine' prefix
            job_parameters = [p for p in parameters if not p.name.startswith("cytomine")]
            argparse = _software_params_to_argparse(job_parameters)
            base_params, _ = argparse.parse_known_args(args=argv)
            return NeubiasJob(vars(flags), base_params)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


if __name__ == "__main__":
    with NeubiasJob.from_cli(sys.argv[1:]) as job:
        print(job.parameters)
