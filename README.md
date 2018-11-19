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

