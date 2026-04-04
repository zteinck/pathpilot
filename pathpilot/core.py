from ._file import *
from ._folder import Folder
from .path import Path
from .config import config

from .utils import (
    trifurcate,
    get_cwd
    )


#╭-------------------------------------------------------------------------╮
#| Configuration Variables                                                 |
#╰-------------------------------------------------------------------------╯

config.extension_map = {
    'csv': CsvFile,
    'parquet': ParquetFile,
    'pickle': PickleFile,
    'py': TextFile,
    'sqlite': SQLiteFile,
    'txt': TextFile,
    'xls': ExcelFile,
    'xlsx': ExcelFile,
    'zip': ZipFile,
    }


#╭-------------------------------------------------------------------------╮
#| Functions                                                               |
#╰-------------------------------------------------------------------------╯

def file_factory(path, **kwargs):
    ''' assigns new file instances to the correct subclass '''
    ext = trifurcate(path)[-1]
    file_cls = config.extension_map.get(ext, File)
    file = file_cls(path, **kwargs)
    return file


def get_python_path(*args, **kwargs):
    ''' return python path environmental variable as a Folder object '''
    return Folder(get_cwd(), *args, **kwargs)


def get_data_path():
    path = (
        get_python_path()
        .parent
        .join('Data', read_only=False)
        )
    return path


#╭-------------------------------------------------------------------------╮
#| Assign Class Attributes                                                 |
#╰-------------------------------------------------------------------------╯

Path._config = config
File.file_factory = Folder.file_factory = file_factory