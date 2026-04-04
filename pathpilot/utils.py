import os
import datetime
from functools import wraps

import clockwork as cw


def split_extension(x):
    ''' separates file extention from the rest of the string '''
    dot_index = x.rfind('.')

    # period not found
    if dot_index == -1:
        rest, ext = x, ''
    else:
        rest = x[:dot_index]
        ext = x[dot_index + 1:]

    return rest, ext


def is_file(path):
    ''' returns True if the path is a file '''
    return bool(trifurcate(path)[-1])


def is_folder(path):
    ''' returns True if the path is a folder '''
    parts = trifurcate(path)
    return parts[0] and not any(parts[1:])


def trifurcate_and_join(path):
    ''' split path into its components (folder inferred if absent) and
        combine them into one string '''
    folder, name, ext = trifurcate(path)
    return f'{folder}{name}.{ext}' if ext else folder


def trifurcate(path, default_folder=True):
    ''' split path into folder, file name, and file extension components '''
    sep = '/'
    path = str(path).replace('\\', sep).strip()

    if not path:
        raise ValueError(
            "'path' argument cannot be empty."
            )

    is_dir = path[-1] == sep
    path = sep.join([x for x in path.split(sep) if x])

    if is_dir or '.' not in path.split(sep)[-1]:
        path += sep

    if path.split(sep)[0][-4:].lower() == '.com':
        path = sep * 2 + path

    path = path.rsplit(sep, 1)

    folder = (
        path[0] + sep
        if len(path) == 2
        else (get_cwd() if default_folder else '')
        )

    name, ext = split_extension(path[-1])
    return folder, name, ext.lower()


def get_cwd():
    pypath = os.getenv('PYTHONPATH')

    path = str(
        os.getcwd()
        if pypath is None
        else pypath.split(os.pathsep)[0]
        )

    path = path.replace('\\','/') + '/'
    return path


def _to_timestamp(func):

    @wraps(func)
    def wrapper(path):
        letter = func.__name__.split('_')[1][0]
        ts = getattr(os.path, f'get{letter}time')(path)

        # Conversion to datetime is redundant since cw.Timestamp
        # constructor accepts timestamps, but explicitly using
        # datetime library ensures time zone information is
        # respected in case constructor behavior ever changes.
        dt = datetime.datetime.fromtimestamp(ts)

        return cw.Timestamp(dt)

    return wrapper


@_to_timestamp
def get_created_date():
    pass


@_to_timestamp
def get_modified_date():
    pass