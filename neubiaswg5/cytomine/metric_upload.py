import os

from cytomine import CytomineJob
from cytomine.models import Property

from neubiaswg5.metrics import computemetrics_batch


def upload_metrics(problemclass, cj: CytomineJob, inputs, gt_path, out_path, tmp_path, **metric_params):
    """Upload each sample will get a value for each metrics of the given problemclass"""
    additional_properties = dict() # properties to be added in addition to the metrics and their parameters
    if problemclass == "ObjSeg":
        # samples are either images or image groups
        outfiles, reffiles = zip(*[
            (os.path.join(out_path, "{}.tif".format(image.id)),
             os.path.join(gt_path, "{}.tif".format(image.id)))
            for image in inputs
        ])
        additional_properties["IMAGES"] = str([im.id for im in inputs])
    else:
        raise ValueError("Unknown problemclass '{}'.".format(problemclass))

    results = computemetrics_batch(outfiles, reffiles, problemclass, tmp_path, **metric_params)

    for key, value in results.items():
        Property(cj.job, key=key, value=str(value)).save()
    for key, value in additional_properties.items():
        Property(cj.job, key=key, value=str(value)).save()