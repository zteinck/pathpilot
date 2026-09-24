from oddments import Validator
import polars as pl

from ...decorators import (
    assert_exists,
    assert_writable,
    )

from ..base import File


class ParquetFile(File):

    #╭-------------------------------------------------------------------------╮
    #| Initialize Instance                                                     |
    #╰-------------------------------------------------------------------------╯

    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)


    #╭-------------------------------------------------------------------------╮
    #| Instance Methods                                                        |
    #╰-------------------------------------------------------------------------╯

    @assert_exists
    def scan(self, **kwargs):
        lf = pl.scan_parquet(self.path, **kwargs)
        return lf


    @assert_writable
    def sink(self, lf, **kwargs):
        lf.sink_parquet(self.path, **kwargs)


    def read(self, **kwargs):
        df = pl.read_parquet(self.path, **kwargs)
        return df


    @assert_writable
    def write(self, df, **kwargs):
        df.write_parquet(self.path, **kwargs)


    def _save(self, obj, **kwargs):

        func_map = {
            pl.LazyFrame: self.sink,
            pl.DataFrame: self.write,
            }

        Validator(types=list(func_map.keys())).validate(obj=obj)
        func_map[type(obj)](obj, **kwargs)