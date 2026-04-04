import os
import shutil
from functools import wraps

from ..utils import is_folder


def _verify_is_folder(func):

    @wraps(func)
    def wrapper(path):
        path = str(path)
        if not is_folder(path):
            raise TypeError(
                f"'path' argument must be a folder, got: {path!r}"
                )
        return func(path)

    return wrapper


@_verify_is_folder
def create_folder(path):
    ''' create folder if it does not already exist '''
    if not os.path.exists(path):
        os.mkdir(path)


@_verify_is_folder
def delete_folder(path):
    ''' delete folder if it exists '''
    if os.path.exists(path):
        shutil.rmtree(path)