import os
import numpy as np
from skimage import io
from cytomine import CytomineJob
from cytomine.models import Annotation, ImageInstance, ImageInstanceCollection, ImageSequenceCollection, AnnotationCollection, Property

from neubiaswg5.exporter import mask_to_objects_2d, mask_to_objects_3d, AnnotationSlice
from shapely.affinity import affine_transform


def annotation_from_slice(slice: AnnotationSlice, id_image, image_height, id_project):
    polygon = affine_transform(slice.polygon, [1, 0, 0, -1, 0, image_height]).wkt
    return Annotation(
        location=polygon, id_image=id_image, id_project=id_project,
        property=[{"key": "index", "value": str(slice.label)}]
    )


def upload_data_objseg(cj: CytomineJob, in_images, out_path, **monitor_params):
    if "prefix" not in monitor_params:
        monitor_params["prefix"] = "Extracting and uploading polygons from masks"

    for image in cj.monitor(in_images, **monitor_params):
        file = "{}.tif".format(image.id)
        path = os.path.join(out_path, file)
        data = io.imread(path)

        collection = AnnotationCollection()
        # extract objects
        if data.ndim == 2:
            slices = mask_to_objects_2d(data)
            collection.extend([annotation_from_slice(s, image.id, image.height, cj.project.id) for s in slices])
        elif data.ndim == 3:
            # in this case `image` is actually an ImageGroup
            image_sequences = ImageSequenceCollection().fetch_with_filter("imagegroup", image.id)
            depth_to_image = {iseq.zStack: iseq.image for iseq in image_sequences}
            height = ImageInstance().fetch(image_sequences[0].image).height

            slices = mask_to_objects_3d(np.moveaxis(data, 0, 2), background=0, assume_unique_labels=True)
            collection.extend([
                annotation_from_slice(
                    slice=s, id_image=depth_to_image[s.depth],
                    image_height=height, id_project=cj.parameters.cytomine_id_project
                ) for obj in slices for s in obj
            ])
        else:
            raise ValueError("Only supports 2D or 3D output images...")

        print("Found {} polygons in this image {}.".format(len(slices), image.id))
        collection.save()


def upload_data(problemclass, cj: CytomineJob, inputs, out_path, **monitor_params):
    if problemclass == "ObjSeg":
        upload_data_objseg(cj, inputs, out_path=out_path, **monitor_params)
    else:
        raise ValueError("Unknown problemclass '{}'.".format(problemclass))
