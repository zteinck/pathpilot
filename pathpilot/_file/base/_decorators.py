from functools import wraps

from ...exceptions import ReadOnlyError
from ..._folder import Folder
from ..._utils import _validate_df_backend


def move_file(func):

    @wraps(func)
    def wrapper(
        self,
        destination,
        overwrite=True,
        raise_on_exist=True,
        raise_on_overwrite=True
        ):
        '''
        Parameters
        ------------
        destination : str | object
            File object or string file path. If Folder, then file is
            moved to that folder.
        overwrite : bool
            if True, if the destination file already exists, it will be
                overwritten otherwise behavior is determined by
                raise_on_overwrite.
            If False, the destination file is returned.
        raise_on_exist : bool
            if True, if the file to be copied does not exist then an
            exception is raised.
        raise_on_overwrite : bool
            if True and overwrite is False, then an exception will be
            raised if the destination file already exists.
        '''

        obj = (
            destination.join(self.full_name)
            if isinstance(destination, Folder)
            else self.trifurcate_and_fill(destination)
            )

        if self.read_only and func.__name__ != 'copy':
            raise ReadOnlyError

        if obj.read_only:
            raise ReadOnlyError(
                f'Destination is in read-only mode: {destination.path}'
                )

        if not overwrite and obj.exists:
            if raise_on_overwrite:
                raise ValueError(
                    'Copy failed. File already exists in destination: '
                    f'{obj.full_name}'
                    )
            else:
                return obj

        # this is below overwite logic so so that self.require() is
        # able to return existing file before raising if exists
        if raise_on_exist and not self.exists:
            raise FileNotFoundError(
                f"Copy failed. File does not exist:\n'{self}'"
                )

        if self.exists:
            func(self, obj)

        return obj

    return wrapper


def add_affix(func):

    @wraps(func)
    def wrapper(self, text, delimiter=' ', encase=False):
        '''
        Parameters
        ------------
        text : str
            text to affix at the beginning or end of the file name
        delimiter : str
            character(s) separating the file name and affix
        encase : bool
            if True, the affix is encased in parenthesis
        '''
        kind = func.__name__.split('_')[-1]
        if encase: text = f'({text})'
        parts = [text, self.name]
        if kind == 'suffix': parts.reverse()
        return self.swap(name=delimiter.join(parts))

    return wrapper


def inject_df_backend(func):

    @wraps(func)
    def wrapper(self, *args,  df_backend=None, **kwargs):
        df_backend = df_backend or self.df_backend
        _validate_df_backend(df_backend)
        return func(self, *args, df_backend=df_backend, **kwargs)

    return wrapper