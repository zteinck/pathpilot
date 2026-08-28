import polars as pl

from ...decorators import (
    assert_exists,
    check_read_only,
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


    @check_read_only
    def sink(self, lf, **kwargs):
        lf.sink_parquet(self.path, **kwargs)


    def read(self, **kwargs):
        df = pl.read_parquet(self.path, **kwargs)
        return df


    def _save(self, df, **kwargs):
        df.write_parquet(self.path, **kwargs)