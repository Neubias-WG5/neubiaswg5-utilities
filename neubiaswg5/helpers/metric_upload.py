import logging
import os

from cytomine import CytomineJob
from cytomine.models import Project

from neubiaswg5.helpers.cytomine_metrics import MetricCollection, get_metric_result_collection, get_metric_result
from neubiaswg5.metrics import computemetrics_batch


def upload_metrics(problemclass, nj, inputs, gt_path, out_path, tmp_path, metric_params=None,
                   do_compute_metrics=False, gt_inputs=None, **kwargs):
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
    do_compute_metrics: bool
        Whether or not to compute and upload the metrics.
    gt_inputs: list|None
        If there is a mismatch between file format of input and ground truth, the ground truth inputs should be passed
        through this parameter.
    """
    if not do_compute_metrics:
        return
    if len(inputs) == 0:
        nj.logger.info("Skipping metric computation because there is no images to process.")
        return
    if metric_params is None:
        metric_params = dict()

    # get image names
    outfiles, reffiles = zip(*[
        (os.path.join(out_path, in_image.filename),
         os.path.join(gt_path, in_image.filename))
        for in_image in (inputs if gt_inputs is None else gt_inputs)
    ])

    # check that output files exist
    for outfile, reffile in zip(outfiles, reffiles):
        if not os.path.isfile(outfile):
            raise FileNotFoundError("There should be one output file for each ground truth file."
                                    "Output file '{}' for reference file '{}' does not exist.".format(outfile, reffiles))

    results, params = computemetrics_batch(outfiles, reffiles, problemclass, tmp_path, **metric_params)

    # effectively upload metrics
    project = Project().fetch(nj.project.id)
    metrics = MetricCollection().fetch_with_filter("discipline", project.discipline)

    metric_collection = get_metric_result_collection(inputs[0].object)
    per_input_metrics = dict()
    for metric_name, values in results.items():
        # check if metric is supposed to be computed for this problem class
        metric = metrics.find_by_attribute("shortName", metric_name)
        if metric is None:
            nj.logger.info("Skip metric '{}' because not listed as a metric of the problem class '{}'.".format(metric_name, problemclass))
            continue
        # create metric results
        for i, in_image in enumerate(inputs):
            image = in_image.object
            metric_result = get_metric_result(image, id_metric=metric.id, id_job=nj.job.id, value=values[i])
            metric_collection.append(metric_result)

            # for logging
            image_dict = per_input_metrics.get(image.id, {})
            image_dict[metric.shortName] = image_dict.get(metric.shortName, []) + [values[i]]
            per_input_metrics[image.id] = image_dict

    nj.logger.log(logging.DEBUG, "Metrics:")
    for _name, _metrics in per_input_metrics.items():
        nj.logger.log(logging.DEBUG, "> {}: {}".format(
            _name,
            ", ".join(["{}:{}".format(m, v) for m, v in _metrics.items()])
        ))

    metric_collection.save()
