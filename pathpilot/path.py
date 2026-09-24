from copy import deepcopy
import os

import oddments as odd
from cachegrab import sha256

from .utils import (
    get_created_date,
    get_modified_date,
    )


class Path(odd.ReprMixin):
    '''
    Description
    --------------------
    Path base class

    Class Attributes
    --------------------
    None

    Instance Attributes
    --------------------
    _read_only : bool
        If True, creating or deleting paths is disabled.
    '''

    #╭-------------------------------------------------------------------------╮
    #| Class Attributes                                                        |
    #╰-------------------------------------------------------------------------╯

    _repr_attrs = ['path', 'read_only']


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
    @odd.validate_on_set(
        types=bool,
        call_wrapped=True,
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
    def parts(self):
        return self._to_parts(self.path)


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

        result = {
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
            result[k] = (
                getattr(self, k).to_datetime()
                if result['exists']
                else None
                )

        return result


    #╭-------------------------------------------------------------------------╮
    #| Static Methods                                                          |
    #╰-------------------------------------------------------------------------╯

    @staticmethod
    def _to_parts(path):
        return [part for part in path.split('/') if part]


    #╭-------------------------------------------------------------------------╮
    #| Instance Methods                                                        |
    #╰-------------------------------------------------------------------------╯

    def clone(self):
        ''' create a copy of the file object '''
        return deepcopy(self)


    def with_prefix(self, prefix):
        ''' adds a prefix the file name '''
        (
        odd.Validator(
            types=str,
            allow_blank=False,
            )
        .validate(
            prefix=prefix
            )
        )

        return self.with_name(f'{prefix}{self.name}')


    def with_suffix(self, suffix):
        ''' add suffix to file name '''
        (
        odd.Validator(
            types=str,
            allow_blank=False,
            )
        .validate(
            suffix=suffix
            )
        )

        return self.with_name(f'{self.name}{suffix}')


    def with_read_only(self):
        path = self.clone()
        path.read_only = True
        return path


    def with_writable(self):
        path = self.clone()
        path.read_only = False
        return path


    #╭-------------------------------------------------------------------------╮
    #| Magic Methods                                                           |
    #╰-------------------------------------------------------------------------╯

    def __hash__(self):
        return hash(self.path)


    def __str__(self):
        return self.path


    def __bool__(self):
        return self.exists