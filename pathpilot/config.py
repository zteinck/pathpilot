from oddments import (
    ReprMixin,
    validate_setter,
    )

from ._utils import _validate_df_backend


class PathpilotConfig(ReprMixin):

    #╭-------------------------------------------------------------------------╮
    #| Initialize Instance                                                     |
    #╰-------------------------------------------------------------------------╯

    def __init__(self):
        self.verbose = True
        self.df_backend = 'pandas'


    #╭-------------------------------------------------------------------------╮
    #| Properties                                                              |
    #╰-------------------------------------------------------------------------╯

    @property
    def verbose(self):
        return self._verbose


    @verbose.setter
    @validate_setter(types=bool)
    def verbose():
        pass


    @property
    def df_backend(self):
        return self._df_backend


    @df_backend.setter
    def df_backend(self, value):
        _validate_df_backend(value)
        self._df_backend = value


config = PathpilotConfig()