import pickle

import pandas as pd

from ..base import File


class PickleFile(File):

    #╭-------------------------------------------------------------------------╮
    #| Initialize Instance                                                     |
    #╰-------------------------------------------------------------------------╯

    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)


    #╭-------------------------------------------------------------------------╮
    #| Static Methods                                                          |
    #╰-------------------------------------------------------------------------╯

    @staticmethod
    def _squeeze(x):
        if isinstance(x, tuple) and len(x) == 1:
            return x[0]
        return x


    #╭-------------------------------------------------------------------------╮
    #| Instance Methods                                                        |
    #╰-------------------------------------------------------------------------╯

    def read(self):
        out = pd.read_pickle(self.path)
        return self._squeeze(out)


    def _save(self, args, **kwargs):
        args = self._squeeze(args)

        if hasattr(args, 'to_pickle'):
            args.to_pickle(self.path, **kwargs)
            return

        with open(self.path, 'wb') as file:
            pickle.dump(args, file, **kwargs)