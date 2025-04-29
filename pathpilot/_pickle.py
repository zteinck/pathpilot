import pickle
import pandas as pd

from ._file import FileBase


class PickleFile(FileBase):

    #╭-------------------------------------------------------------------------╮
    #| Initialize Instance                                                     |
    #╰-------------------------------------------------------------------------╯

    def __init__(self, f, **kwargs):
        super().__init__(f, **kwargs)


    #╭-------------------------------------------------------------------------╮
    #| Instance Methods                                                        |
    #╰-------------------------------------------------------------------------╯

    def read(self):
        out = pd.read_pickle(self.path)
        return out[0] if isinstance(out, tuple) and len(out) == 1 else out


    def _save(self, args, **kwargs):
        if isinstance(args, tuple) and len(args) == 1:
            args = args[0]

        if hasattr(args, 'to_pickle'):
            args.to_pickle(self.path, **kwargs)
            return

        with open(self.path, 'wb') as file:
            pickle.dump(args, file, **kwargs)