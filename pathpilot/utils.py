import shutil
import filecmp
import math
import os
import datetime
import inspect
import numpy as np
from clockwork import Date



#╭-------------------------------------------------------------------------╮
#| Classes                                                                 |
#╰-------------------------------------------------------------------------╯

class ReadOnlyError(Exception):

    def __init__(self, message=None):
        super().__init__(message or 'cannot perform this action in read-only mode')



#+---------------------------------------------------------------------------+
# Functions
#+---------------------------------------------------------------------------+

def check_read_only(func):

    def wrapper(self, *args, **kwargs):
        if self.read_only: raise ReadOnlyError
        return func(self, *args, **kwargs)

    return wrapper


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
    ''' split argument into its components (folder inferred if absent) and combine them into one string '''
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


def get_size_label(size_in_bytes, decimal_places=2):

    units = ('','K','M','G','T','P','E','Z','Y')
    conversion_factor = 1024

    if size_in_bytes == 0:
        index, size = 0, 0
    else:
        index = int(math.floor(math.log(size_in_bytes, conversion_factor)))
        size = size_in_bytes / math.pow(conversion_factor, index)

    return f'{size:,.{decimal_places}f} {units[index]}B'


def backup_folder(origin, destination, overwrite=True, shallow=True, verbose=True):
    '''
    Description
    ------------
    Backs up the the folders and files in the 'origin' folder to the 'destination' folder.
    Files in 'destination' are overwritten if they are different than files of the same name in
    'origin' according to filecmp.cmp (e.g. doc.xlsx exists in both directories but the version
    in 'origin' was updated since the last time a backup was performed.

    Parameters
    ------------
    origin : str | Folder
        folder to backup
    destination : str | Folder
        backup folder
    overwrite : bool
        if True, if the destination file already exists and it is different than the origin file,
        it will be overwritten.
        If False, overlapping files are ignored.
    shallow : bool
        filecmp.cmp(f1, f2, shallow=True) shallow argument.
        "If shallow is true and the os.stat() signatures (file type, size, and modification time) of both
        files are identical, the files are taken to be equal."
        https://docs.python.org/3/library/filecmp.html
    verbose : bool
        if True, all folders and files that were backed up or overwritten are printed.

    Returns
    ------------
    None
    '''

    def format_path(path):
        path = str(path).replace('\\','/')
        if path[-1] != '/': path = path + '/'
        return path

    origin = format_path(origin)
    destination = format_path(destination)
    if not os.path.exists(destination): os.mkdir(destination)

    for path, folders, files in os.walk(origin):
        from_path = format_path(path)
        to_path = from_path.replace(origin, destination)

        for file in files:
            copy_file = False
            from_file = from_path + file
            to_file = to_path + file
            text = to_file.replace(destination, '/')

            if os.path.exists(to_file):
                if not filecmp.cmp(from_file, to_file, shallow=shallow) and overwrite:
                    action = 'Overwrite'
                    copy_file = True
            else:
                action = 'BackingUp'
                copy_file = True

            if copy_file:
                try:
                    shutil.copyfile(from_file, to_file)
                    if verbose: print(f'{action}: {text}')
                except Exception as e:
                    if verbose: print(e)

        for folder in folders:
            to_folder = to_path + folder
            text = to_folder.replace(destination, '/') + '/'
            if not os.path.exists(to_folder):
                os.mkdir(to_folder)
                if verbose: print(f'BackingUp: {text}')


def purge_whitespace(func):
    ''' wrapper function that purges unwanted whitespace from a DataFrame '''

    def wrapper(*args, **kwargs):

        def strip_or_skip(x):
            try:
                x = x.strip()
                if x == '': return np.nan
            except:
                pass
            return x

        df = func(*args, **kwargs)

        # clean column names by trimming leading/trailing whitespace and removing new lines and consecutive spaces
        df.rename(columns={k: ' '.join(k.split()) for k in df.columns if isinstance(k, str)}, inplace=True)

        # replace None with np.nan
        for k in df.columns:
            try:
                df[k] = df[k].fillna(np.nan)
            except:
                pass

        # trim leading/trailing whitespace and replace whitespace-only values with NaN
        for k in df.select_dtypes(include=['object']).columns:
            # df[k] = df[k].replace(to_replace=r'^\s*$', value=np.nan, regex=True)

            # using the vectorized string method str.strip() is faster but object-type columns can have mixed data types
            df[k] = df[k].apply(strip_or_skip) #.str.strip()

        # df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

        return df

    return wrapper


def timestamp_to_date(func):

    def wrapper(path):
        return Date(datetime.datetime.fromtimestamp(func(path)))

    return wrapper


@timestamp_to_date
def get_created_date(path):
    return os.path.getctime(path)


@timestamp_to_date
def get_modified_date(path):
    return os.path.getmtime(path)


def verify_is_folder(func):

    def wrapper(f):
        f = str(f)
        if not is_folder(f):
            raise TypeError(f"{f} is not a folder")
        return func(f)

    return wrapper


@verify_is_folder
def create_folder(f):
    ''' create folder if it does not already exist '''
    if not os.path.exists(f):
        os.mkdir(f)


@verify_is_folder
def delete_folder(f):
    ''' delete folder if it exists '''
    if os.path.exists(f):
        shutil.rmtree(f)


def get_object_folder(obj):
    ''' return file folder of Python object '''
    return os.path.absfolder(inspect.getfile(obj))