from .utils import get_size_label

from .base import FileBase

from ._csv import CSVFile
from ._excel import ExcelFile
from ._pickle import PickleFile
from ._sqlite import SQLiteFile
from ._text import TextFile
from ._zip import ZipFile