import polars as pl

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

    def scan(self, **kwargs):
        lf = pl.scan_parquet(self.path, **kwargs)
        return lf


    def read(self, **kwargs):
        df = pl.read_parquet(self.path, **kwargs)
        return df


    def _save(self, df, **kwargs):
        df.write_parquet(self.path, **kwargs)