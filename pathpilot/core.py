from ._file import *
from ._folder import *

from .utils import (
    trifurcate,
    get_cwd
    )


#╭-------------------------------------------------------------------------╮
#| Variables                                                               |
#╰-------------------------------------------------------------------------╯

extension_mapping = {
    'xlsx': ExcelFile,
    'xls': ExcelFile,
    'csv': CSVFile,
    'pickle': PickleFile,
    'py': TextFile,
    'txt': TextFile,
    'zip': ZipFile,
    'sqlite': SQLiteFile,
    }


#╭-------------------------------------------------------------------------╮
#| Functions                                                               |
#╰-------------------------------------------------------------------------╯

def file_factory(f, **kwargs):
    ''' assigns new file instances to the correct subclass '''
    extension = trifurcate(f)[-1]
    out = extension_mapping.get(extension, FileBase)(f, **kwargs)
    return out


def get_python_path(*args, **kwargs):
    ''' return python path environmental variable as a Folder object '''
    return Folder(get_cwd(), *args, **kwargs)


def get_data_path():
    return get_python_path().parent.join('Data', read_only=False)


#╭-------------------------------------------------------------------------╮
#| Assign Class Attributes                                                 |
#╰-------------------------------------------------------------------------╯

FileBase.file_factory = file_factory
Folder.file_factory = file_factory