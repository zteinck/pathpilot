import oddments as odd

from .._utils import _is_frame
from .core import File
from ._decorators import inject_df_backend


class DfDispatchFile(File):

    #╭-------------------------------------------------------------------------╮
    #| Initialize Instance                                                     |
    #╰-------------------------------------------------------------------------╯

    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)


    #╭-------------------------------------------------------------------------╮
    #| Instance Methods                                                        |
    #╰-------------------------------------------------------------------------╯

    @odd.apply_purge_whitespace
    @inject_df_backend
    def read(self, df_backend, **kwargs):
        func = (
            self._read_with_pandas
            if df_backend == 'pandas'
            else self._read_with_polars
            )
        return func(**kwargs)


    @inject_df_backend
    def _save(self, obj, df_backend, **kwargs):

        if df_backend == 'pandas':
            cast_func = odd.to_pandas_frame
            save_func = self._save_with_pandas
        else:
            cast_func = odd.to_polars_frame
            save_func = self._save_with_polars

        if _is_frame(obj):
            obj = cast_func(obj)

        save_func(obj, **kwargs)