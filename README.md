# Neubias-WG5 utilities

Utilities for Neubias WG5 softwares

## Running workflows locally

You can run workflows without communicating with neubias

Requirements:
- in the workflow Dockerfile: `ADD descriptor.json /app/descriptor.json`

Then run:
```bash
docker run -v $LOCAL_INPUT:/data/in $LOCAL_OUTPUT:/data/out $LOCAL_GROUND_TRUTH:/data/gt -it DOCKER_TAG \
    (workflow parameters ...)
    --nodownload --noexport --nometrics \
    --infolder /data/in \
    --outfolder /data/out \
    --gtfolder /data/gt
```

where `$LOCAL_INPUT`, `$LOCAL_OUTPUT` and `$LOCAL_GROUND_TRUTH` are the local data folders (i.e. on your host).

## Install

Some installation steps have to be performed manually:
- installation of [Cytomine-Python-Client](https://github.com/Cytomine-ULiege/Cytomine-python-client)
- installation of binaries for metrics computation

```bash
# install cytomine python client
pip install https://github.com/Cytomine-ULiege/Cytomine-python-client/archive/master.zip

# install utilities
git clone https://github.com/Neubias-WG5/neubiaswg5-utilities.git
cd neubiaswg5-utilities
pip install .

# manually place binaries in path (required for using neubiaswg5.metrics subpackage)
chmod +x bin/*
cp bin/* /usr/bin/
```

## `neubias.helpers`

This sections documents the three helper functions `prepare_data`, `upload_data` and `upload_metrics`.

In general, the parameter `problemclass` should one of the following constants (see `neubiaswg5/problemclass.py`):

* `CLASS_OBJSEG`: object segmentation
* `CLASS_SPTCNT`: spot count
* `CLASS_PIXCLA`: pixel classification
* `CLASS_TRETRC`: tree network tracing
* `CLASS_LOOTRC`: loopy network tracing
* `CLASS_OBJDET`: object detection
* `CLASS_PRTTRK`: particle tracking
* `CLASS_OBJTRK`: object tracking
* `CLASS_LNDDET`: landmark detection

### Prepare data

This functions sets up the execution environment of a workflow by creating necessary folders and by downloading the
input data, or simply checking that this data is already present in the expected folder.

Parameters:

* `problemclass` (type: `str`): the problem class of the workflow for which the env must be setup.
* `nj` (type: `CytomineJob|NeubiasJob`): a CytomineJob or NeubiasJob instance.
* `gt_suffix` (type: `str`): ground truth images suffix in the Neubias project.
* `base_path` (type: `str|None`): base path for data download. Defaults to the `$HOME/{nj.job.id}/`.
* `do_download` (type: `bool`): true if data should be downloaded from a BIAFLOWS instance.
* `infolder` (type: `str|None`): full path of the folder for input data. If None, defaults to `{base_path}/in`.
* `outfolder` (type: `str|None`): full path of the folder for output data. If None, defaults to `{base_path}/out`.
* `gtfolder` (type: `str|None`): full path of the folder for ground truth data. If None, defaults to `{base_path}/ground_truth`.
* `tmp_folder` (type: `str`): name (not the path) for temporary data folder.
* `is_2d` (type: `bool`): True if the problem is a 2d one, False otherwise (3D, 4D, 3D+t).
* `kwargs` (type: `dict`): additional problem class-specific parameters (see sections below).

For `CLASS_TRETRC`:

* `suffix` (type: `str`): suffix in the filename for attached files (by default `_attached`).

### Upload data



## `neubiaswg5.exporter`

Annotation export tools.

### Annotation slice

The `AnnotationSlice` class represents a 2D annotation with additional metadata when relevant:

- _shape_: encoded as a polygon
- _label_: label associated to the annotation
- _index_: for 3D volumes, the depth index
- _time_: for time volumes, the time index

### From masks to objects

See file `mask_to_objects.py`. All function take a multi-dimensional array as input and output AnnotationSlice
- 2D: `mask_to_objects_2d `
- 3D/2D+t: `mask_to_objects_3d`
- 3D+t: `mask_to_objects_3dt`

### From masks to points

See file `mask_to_points.py`:

- 2D: `mask_to_points_2d`


## `neubiaswg5.metrics`


test/metrics/test_compute_metrics.py is a sample wrapper script calling the benchmarking module (ComputeMetrics), some calls and sample images are provided.

