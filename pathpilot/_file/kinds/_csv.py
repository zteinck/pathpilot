import oddments as odd

import pandas as pd
import polars as pl

from ...decorators import check_read_only
from ..base import DfDispatchFile


class CsvFile(DfDispatchFile):

    #╭-------------------------------------------------------------------------╮
    #| Initialize Instance                                                     |
    #╰-------------------------------------------------------------------------╯

    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)


    #╭-------------------------------------------------------------------------╮
    #| Instance Methods                                                        |
    #╰-------------------------------------------------------------------------╯

    def _read_with_pandas(self, **kwargs):
        defaults = {
            'encoding': 'ISO-8859-1',
            'keep_default_na': False,
            'na_values': [''],
            }
        params = defaults | kwargs
        return pd.read_csv(self.path, **params)


    def _save_with_pandas(self, obj, **kwargs):
        params = {'index': odd.has_named_index(obj)} | kwargs
        obj.to_csv(self.path, **params)


    def _read_with_polars(self, **kwargs):
        return pl.read_csv(self.path, **kwargs)


    def _save_with_polars(self, obj, **kwargs):
        obj.write_csv(self.path, **kwargs)


    def scan(self, **kwargs):
        return pl.scan_csv(self.path, **kwargs)


    @check_read_only
    def sink(self, lf, **kwargs):
        lf.sink_csv(self.path, **kwargs)