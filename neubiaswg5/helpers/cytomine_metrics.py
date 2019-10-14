from cytomine.models import Model, Collection, ImageInstance, ImageInstanceCollection


class Metric(Model):
    def __init__(self, name=None, shortName=None, **attributes):
        super(Metric, self).__init__()
        self.name = name
        self.shortName = shortName
        self.populate(attributes)

    def __str__(self):
        return self.name + " (" + self.shortName + ")"


class MetricCollection(Collection):
    def __init__(self, filters=None, max=0, offset=0, **parameters):
        super(MetricCollection, self).__init__(Metric, filters, max, offset)
        self._allowed_filters = [None, "discipline"]
        self.set_parameters(parameters)


class ImageInstanceMetricResult(Model):
    def __init__(self, id_metric=None, id_job=None, id_image=None, value=None, **attributes):
        super(ImageInstanceMetricResult, self).__init__()
        self.metric = id_metric
        self.job = id_job
        self.image = id_image
        self.value = value
        self.populate(attributes)

    def __str__(self):
        return "metric[{}] = {}".format(self.metric, self.value)


class ImageInstanceMetricResultCollection(Collection):
    def __init__(self, filters=None, max=0, offset=0, **parameters):
        super(ImageInstanceMetricResultCollection, self).__init__(ImageInstanceMetricResult, filters, max, offset)
        self._allowed_filters = [None]
        self.set_parameters(parameters)


def _check_type(obj, model_type, collection_type):
    return isinstance(obj, model_type) or isinstance(obj, collection_type)\
            or (isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], model_type))


def get_metric_result_collection(image_or_collection):
    if _check_type(image_or_collection, ImageInstance, ImageInstanceCollection):
        return ImageInstanceMetricResultCollection()
    else:
        raise ValueError("Unknown type for object (t:{}).".format(type(image_or_collection)))


def get_metric_result(image, id_metric=None, id_job=None, value=None, **attributes):
    if _check_type(image, ImageInstance, ImageInstanceCollection):
        return ImageInstanceMetricResult(id_metric=id_metric, id_job=id_job, id_image=image.id, value=value, **attributes)
    else:
        raise ValueError("Unknown type for object (t:{}).".format(type(image)))