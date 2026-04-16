from .core import (
    config,
    file_factory,
    get_python_path,
    get_data_path,
    )

from ._folder import *
from ._file import *
from .decorators import check_read_only
from .exceptions import ReadOnlyError

from .utils import (
    split_extension,
    is_file,
    is_folder,
    trifurcate_and_join,
    trifurcate,
    get_cwd,
    get_created_date,
    get_modified_date,
    )

__version__ = '0.4.5'
__author__ = 'Zachary Einck <zacharyeinck@gmail.com>'