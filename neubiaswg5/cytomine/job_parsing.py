import json
import sys
from argparse import ArgumentParser


from cytomine.cytomine_job import _software_params_to_argparse, CytomineJob


def check_field(d, f, target="dictionary"):
    if f not in d:
        raise ValueError("Missing field '{}' in {}".format(f, target))
    return d[f]


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


class NeubiasJob(object):
    """Equivalent of CytomineJob but that can run without a Cytomine server. Can be used with a context manager"""
    def __init__(self, parameters):
        self._parameters = parameters

    @property
    def parameters(self):
        return self._parameters

    @staticmethod
    def from_cli(argv, **kwargs):
        """
        Returns a CytomineJob if interaction with Cytomine are needed. Otherwise, returns a minimal NeubiasJob
        that emulates a CytomineJob (e.g. parameters access).
        """
        # Parse CytomineJob constructor parameters
        flags_ap = ArgumentParser()
        flags_ap.add_argument("--nodownload", dest="do_download", action="store_false", required=False,
                              help="Whether or not to download data from BIAFLOWS/local server. If raised, the absolute"
                                   " path path to the folder containing the input data should be provided through "
                                   "--infolder.")
        flags_ap.add_argument("--noexport", dest="do_export", action="store_false", required=False,
                              help="Whether or not to upload results annotations to the BIAFLOWS/local server.")
        flags_ap.add_argument("--nometrics", dest="do_compute_metrics", action="store_false", required=False,
                              help="Whether or not to compute and upload the metrics to the BIAFLOWS/local server. If "
                                   "--nodownload is raised but --nometrics is not, then the absolute path to the folder"
                                   " containing the ground truth data should be provided through --gtfolder.")
        flags_ap.set_defaults(do_download=True, do_export=True, do_compute_metrics=True)
        flags, _ = flags_ap.parse_known_args(argv)

        # Cytomine is needed if at least one flag is not raised.
        if flags.do_download or flags.do_export or flags.do_compute_metrics:
            return CytomineJob.from_cli(argv, **kwargs)

        with open("descriptor.json", "r") as file:
            software_desc = json.load(file)

            if software_desc is None or "inputs" not in software_desc:
                raise ValueError("Cannot read 'descriptor.json' or missing 'inputs' in JSON")

            parameters = [NeubiasParameter(**param) for param in software_desc["inputs"]]

            # exclude all cytomine parameters
            # hypothesis: such parameters always start with 'cytomine' prefix
            job_parameters = [p for p in parameters if not p.name.startswith("cytomine")]
            argparse = _software_params_to_argparse(job_parameters)
            base_params, _ = argparse.parse_known_args(args=argv)
            return NeubiasJob(base_params)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


if __name__ == "__main__":
    with NeubiasJob.from_cli(sys.argv[1:]) as job:
        print(job.parameters)