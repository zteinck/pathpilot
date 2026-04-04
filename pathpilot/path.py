import os

import oddments as odd
from cachegrab import sha256

from .utils import (
    get_created_date,
    get_modified_date,
    )


class Path(object):
    '''
    Description
    --------------------
    Path base class

    Class Attributes
    --------------------
    ...

    Instance Attributes
    --------------------
    _read_only : bool
        If True, creating or deleting paths is disabled.
    '''

    #╭-------------------------------------------------------------------------╮
    #| Initialize Instance                                                     |
    #╰-------------------------------------------------------------------------╯

    def __init__(self, read_only):
        self.read_only = read_only


    #╭-------------------------------------------------------------------------╮
    #| Properties                                                              |
    #╰-------------------------------------------------------------------------╯

    @property
    def read_only(self):
        return self._read_only


    @read_only.setter
    @odd.validate_setter(
        types=bool,
        call_func=True
        )
    def read_only(self, value):
        self._read_only = value
        self._on_read_only_toggle()


    @property
    def verbose(self):
        return self._config.verbose


    @property
    def df_backend(self):
        return self._config.df_backend


    @property
    def exists(self):
        ''' returns True if the file currently exists '''
        return os.path.exists(self.path)


    @property
    def hash_value(self):
        ''' sha256 hash value of path '''
        return sha256(self.path)


    @property
    def created_date(self):
        ''' date the file was created '''
        if self.exists:
            return get_created_date(self.path)


    @property
    def modified_date(self):
        ''' date the file was modified '''
        if self.exists:
            return get_modified_date(self.path)


    @property
    def meta_data(self):

        out = {
            'type': self.__class__.__name__,
            'hash_value': self.hash_value,
            'path': self.path,
            'name': self.name,
            'read_only': self.read_only,
            'exists': self.exists,
            }

        for k in [
            'created_date',
            'modified_date',
            ]:
            out[k] = (
                getattr(self, k).to_datetime()
                if out['exists']
                else None
                )

        return out


    #╭-------------------------------------------------------------------------╮
    #| Instance Methods                                                        |
    #╰-------------------------------------------------------------------------╯

    def _on_read_only_toggle(self):
        pass


    #╭-------------------------------------------------------------------------╮
    #| Magic Methods                                                           |
    #╰-------------------------------------------------------------------------╯

    def __repr__(self):
        return self.path.replace('/','\\')


    def __str__(self):
        return self.path


    def __bool__(self):
        return self.exists