import pandas as pd

from ._file import FileBase
from .decorators import purge_whitespace



class CSVFile(FileBase):

    def __init__(self, f, **kwargs):
        super().__init__(f, **kwargs)


    @purge_whitespace
    def read(self, **kwargs):
        kwargs.setdefault('encoding', 'ISO-8859-1')
        kwargs.setdefault('keep_default_na', False)
        return pd.read_csv(self.path, **kwargs)


    def _save(self, obj, **kwargs):
        obj.to_csv(self.path, **kwargs)