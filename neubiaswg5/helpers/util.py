import os
from abc import ABCMeta, abstractmethod, abstractproperty

from cytomine.models._utilities import resolve_pattern


def default_value(v, default):
    return default if v is None else v


def makedirs_ifnotexists(folder):
    """Create folder if not exists"""
    if not os.path.exists(folder):
        os.makedirs(folder)


def check_field(d, f, target="dictionary"):
    if f not in d:
        raise ValueError("Missing field '{}' in {}".format(f, target))
    return d[f]


def split_filename(filename):
    return filename.rsplit(".", 1)


class NeubiasInput(metaclass=ABCMeta):
    """A neubias input is a file that is used as input of a workflow (either ground truth or actual input).
    This class provides utilities methods for manipulating the input images
    instance.object allows to get the underlying input object
    instance.attached allows to get the list of files (as their filepath) attached to the input
    """
    def __init__(self, obj, attached=None):
        self._obj = obj
        self._attached = list() if attached is None else attached

    @property
    @abstractmethod
    def filepath(self):
        pass

    @property
    @abstractmethod
    def filename(self):
        """
        Returns
        -------
        str
        """
        pass

    @property
    def extension(self):
        return split_filename(self.filename)[1]

    @property
    def filename_no_extension(self):
        return split_filename(self.filename)[0]

    @property
    def object(self):
        return self._obj

    @property
    def attached(self):
        return self._attached


class NeubiasCytomineInput(NeubiasInput):
    def __init__(self, cytomine_model, in_path="", name_pattern="{id}.tif"):
        super().__init__(cytomine_model)
        self._in_path = in_path
        self._name_pattern = name_pattern

    @property
    def filename(self):
        return resolve_pattern(self._name_pattern, self.object)[0]

    @property
    def filepath(self):
        return os.path.join(self._in_path, self.filename)

    @property
    def original_filename(self):
        return getattr(self.object, self.filename_attribute)

    @property
    @abstractmethod
    def filename_attribute(self):
        """
        Returns
        -------
        attr: str
        """
        pass


class NeubiasImageInstance(NeubiasCytomineInput):
    @property
    def filename_attribute(self):
        return "originalFilename"


class NeubiasImageGroup(NeubiasCytomineInput):
    @property
    def filename_attribute(self):
        return "name"


class NeubiasFilepath(NeubiasInput):
    def __init__(self, filepath):
        super().__init__(filepath)

    @property
    def filepath(self):
        return self.object

    @property
    def filename(self):
        return os.path.basename(self.filepath)


class NeubiasAttachedFile(NeubiasCytomineInput):
    def __init__(self, attached_file, in_path="", name_pattern="{filename}"):  # change default pattern
        super().__init__(attached_file, in_path, name_pattern)

    @property
    def filename_attribute(self):
        return "filename"
