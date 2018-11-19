import os


def default_value(v, default):
    return default if v is None else v


def makedirs_ifnotexists(folder):
    """Create folder if not exists"""
    if not os.path.exists(folder):
        os.makedirs(folder)
