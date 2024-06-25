from ._file import FileBase
from ._folder import Folder
from ._excel import ExcelFile
from ._csv import CSVFile
from ._pickle import PickleFile
from ._text import TextFile
from ._zip import ZipFile
from ._sqlite import SQLiteFile
from .utils import trifurcate, get_cwd


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

def File(f, **kwargs):
    ''' assigns new file instances to the correct class polymorphism '''
    extension = trifurcate(f)[-1]
    out = extension_mapping.get(extension, FileBase)(f, **kwargs)
    return out


def get_python_path(*args, **kwargs):
    ''' return python path environmental variable as a Folder object '''
    return Folder(get_cwd(), *args, **kwargs)


def get_data_path():
    return get_python_path().parent.join('Data', read_only=False)


#╭-------------------------------------------------------------------------╮
#| Assign Class Attribute                                                  |
#╰-------------------------------------------------------------------------╯

FileBase.factory = File
Folder.factory = File