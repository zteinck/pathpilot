import pandas as pd
import oddments as odd

from .base import FileBase


class CSVFile(FileBase):

    #╭-------------------------------------------------------------------------╮
    #| Initialize Instance                                                     |
    #╰-------------------------------------------------------------------------╯

    def __init__(self, f, **kwargs):
        super().__init__(f, **kwargs)


    #╭-------------------------------------------------------------------------╮
    #| Instance Methods                                                        |
    #╰-------------------------------------------------------------------------╯

    @odd.purge_whitespace
    def read(self, **kwargs):
        kwargs.setdefault('encoding', 'ISO-8859-1')
        kwargs.setdefault('keep_default_na', False)
        return pd.read_csv(self.path, **kwargs)


    def _save(self, obj, **kwargs):
        obj.to_csv(self.path, **kwargs)