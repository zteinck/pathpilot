import os
import datetime
from clockwork import Date


def split_extension(x):
    ''' separates file extention from rest of string '''
    dot_index = x.rfind('.')
    if dot_index == -1: # period not found
        return x, ''
    else:
        ext = x[dot_index + 1:]
        rest = x[:dot_index]
        return rest, ext


def is_file(f):
    ''' returns True if the argument is a file '''
    return bool(trifurcate(f)[-1])


def is_folder(f):
    ''' returns True if the argument is a folder '''
    f = trifurcate(f)
    return f[0] and not any(f[1:])


def trifurcate_and_join(f):
    ''' split argument into its components (folder inferred if absent) and
        combine them into one string '''
    folder, name, ext = trifurcate(f)
    return (folder + name + '.' + ext) if ext else folder


def trifurcate(f, default_folder=True):
    ''' split argument into folder, file name, and file extension components '''
    f = str(f).replace('\\','/').strip()
    if not f: raise ValueError("'f' argument cannot be empty")
    explicitly_folder = f[-1] == '/'
    f = '/'.join([x for x in f.split('/') if x])
    f = f + '/' if explicitly_folder or '.' not in f.split('/')[-1] else f
    if f.split('/')[0][-4:].lower() == '.com': f = '//' + f
    f = f.rsplit('/', 1)
    folder = f[0] + '/' if len(f) == 2 else (get_cwd() if default_folder else '')
    name, ext = split_extension(f[-1])
    return folder, name, ext.lower()


def get_cwd():
    pypath = os.getenv('PYTHONPATH')
    f = os.getcwd() if pypath is None else pypath.split(os.pathsep)[0]
    f = str(f).replace('\\','/') + '/'
    return f


def _timestamp_to_date(func):

    def wrapper(path):
        return Date(datetime.datetime.fromtimestamp(func(path)))

    return wrapper


@_timestamp_to_date
def get_created_date(path):
    return os.path.getctime(path)


@_timestamp_to_date
def get_modified_date(path):
    return os.path.getmtime(path)