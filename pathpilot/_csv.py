import pandas as pd

from ._file import FileBase
from .utils import purge_whitespace



class CSVFile(FileBase):

    def __init__(self, f, **kwargs):
        super().__init__(f, **kwargs)


    @purge_whitespace
    def read(self, **kwargs):
        df = pd.read_csv(
            self.path,
            encoding='ISO-8859-1',
            keep_default_na=False,
            **kwargs
            )
        return df


    def _save(self, obj, **kwargs):
        obj.to_csv(self.path, **kwargs)