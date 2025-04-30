import os
import shutil

from ..utils import is_folder


def _verify_is_folder(func):

    def wrapper(f):
        f = str(f)
        if not is_folder(f):
            raise TypeError(f"{f} is not a folder")
        return func(f)

    return wrapper


@_verify_is_folder
def create_folder(f):
    ''' create folder if it does not already exist '''
    if not os.path.exists(f):
        os.mkdir(f)


@_verify_is_folder
def delete_folder(f):
    ''' delete folder if it exists '''
    if os.path.exists(f):
        shutil.rmtree(f)

