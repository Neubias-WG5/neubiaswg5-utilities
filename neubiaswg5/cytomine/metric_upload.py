import os

from cytomine import CytomineJob
from cytomine.models import Property

from neubiaswg5 import CLASS_OBJSEG
from neubiaswg5.metrics import computemetrics_batch


def upload_metrics(problemclass, nj, inputs, gt_path, out_path, tmp_path, metric_params=None,
                   do_download=False, do_compute_metrics=False, **kwargs):
    """Upload each sample will get a value for each metrics of the given problemclass
    Parameters
    ----------
    problemclass: str
        Problem class
    nj: CytomineJob|NeubiasJob
        A CytomineJob or NeubiasJob instance.
    inputs: iterable
        A list of input data for which the metrics must be computed
    gt_path: str
        Absolute path to the ground truth data
    out_path: str
        Absolute path to the output data
    tmp_path: str
        Absolute path to a temporary folder
    metric_params: dict
        Additional parameters for metric computation (forwarded to computemetrics or computemetrics_batch directly)
    do_download: bool
        Whether or not the images were downloaded from the server.
    do_compute_metrics: bool
        Whether or not to compute and upload the metrics.
    """
    if not do_compute_metrics:
        return
    if metric_params is None:
        metric_params = dict()
    additional_properties = dict() # properties to be added in addition to the metrics and their parameters
    if problemclass == CLASS_OBJSEG:
        # get image names
        filenames = ["{}.tif".format(image.id) if do_download else os.path.basename(image) for image in inputs]
        outfiles, reffiles = zip(*[
            (os.path.join(out_path, filename),
             os.path.join(gt_path, filename))
            for filename in filenames
        ])
        additional_properties["IMAGES"] = str([im.id if do_download else os.path.basename(im) for im in inputs])
    else:
        raise ValueError("Unknown problemclass '{}'.".format(problemclass))

    results = computemetrics_batch(outfiles, reffiles, problemclass, tmp_path, **metric_params)

    for key, value in results.items():
        Property(nj.job, key=key, value=str(value)).save()
    for key, value in additional_properties.items():
        Property(nj.job, key=key, value=str(value)).save()