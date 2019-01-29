import logging
import os

from cytomine import CytomineJob
from cytomine.models import Project

from neubiaswg5.helpers.cytomine_metrics import MetricCollection, get_metric_result_collection, get_metric_result
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

    # get image names
    filenames = ["{}.tif".format(image.id) if do_download else os.path.basename(image) for image in inputs]
    outfiles, reffiles = zip(*[
        (os.path.join(out_path, filename),
         os.path.join(gt_path, filename))
        for filename in filenames
    ])
    results, params = computemetrics_batch(outfiles, reffiles, problemclass, tmp_path, **metric_params)

    # effectively upload metrics
    project = Project().fetch(nj.project.id)
    metrics = MetricCollection().fetch_with_filter("discipline", project.discipline)

    metric_collection = get_metric_result_collection(inputs)
    per_input_metrics = dict()
    for metric_name, values in results.items():
        # check if metric is supposed to be computed for this problem class
        metric = metrics.find_by_attribute("shortName", metric_name)
        if metric is None:
            nj.logger.info("Skip metric '{}' because not listed as a metric of the problem class '{}'.".format(metric_name, problemclass))
            continue
        # create metric results
        for i, _input in enumerate(inputs):
            metric_result = get_metric_result(_input, id_metric=metric.id, id_job=nj.job.id, value=values[i])
            metric_collection.append(metric_result)

            # for logging
            image_dict = per_input_metrics.get(_input.id, {})
            image_dict[metric.shortName] = image_dict.get(metric.shortName, []) + [values[i]]
            per_input_metrics[_input.id] = image_dict

    metric_collection.save()

    nj.logger.log(logging.DEBUG, "Metrics:")
    for _name, _metrics in per_input_metrics.items():
        nj.logger.log(logging.DEBUG, "> {}: {}".format(
            _name,
            ", ".join(["{}:{}".format(m, v) for m, v in _metrics.items()])
        ))
