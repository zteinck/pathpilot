from functools import cached_property
import shutil
import os

import oddments as odd

from ...path import Path
from ...exceptions import ReadOnlyError

from ...decorators import (
    assert_writable,
    inject_read_only,
    )

from ...utils import (
    trifurcate,
    trifurcate_and_join,
    is_file,
    is_folder,
    get_cwd,
    )

from ..utils import (
    create_folder,
    delete_folder,
    )

from ._contents import *


class Folder(Path):
    '''
    Description
    --------------------
    Folder object

    Class Attributes
    --------------------
    file_factory : callable
        Function or class that determines which subclass is returned when
        a new file object is initialized.
    sorter : callable
        Function used to sort folder and file objects.

    Instance Attributes
    --------------------
    _path : str
        Folder's path. Defaults to the return value of get_cwd() if not
        provided.
    '''

    #╭-------------------------------------------------------------------------╮
    #| Class Attributes                                                        |
    #╰-------------------------------------------------------------------------╯

    sorter = odd.natural_sort


    #╭-------------------------------------------------------------------------╮
    #| Initialize Instance                                                     |
    #╰-------------------------------------------------------------------------╯

    def __init__(self, path=None, read_only=True):
        if path is None:
            path = get_cwd()

        self._path = trifurcate(path)[0]
        super().__init__(read_only=read_only)


    #╭-------------------------------------------------------------------------╮
    #| Class Methods                                                           |
    #╰-------------------------------------------------------------------------╯

    @classmethod
    def _spawn_file(cls, *args, **kwargs):
        return cls.file_factory(*args, **kwargs)


    @classmethod
    def sort(cls, x):
        return cls.sorter(x)


    #╭-------------------------------------------------------------------------╮
    #| Properties                                                              |
    #╰-------------------------------------------------------------------------╯

    @property
    def path(self):
        return self._path[:]


    @property
    def empty(self):
        ''' True if folder contains no files or folders otherwise False '''
        return len(self) == 0


    @property
    def name(self):
        ''' name of folder '''
        return self.parts[-1]


    @property
    def full_name(self):
        ''' name alias (for consistency with file properties) '''
        return self.name


    @property
    def parent(self):
        ''' parent folder '''
        return self._get_parent(read_only=self.read_only)


    @property
    def depth(self):
        return len(self.parts)


    @property
    def meta_data(self):
        result = super().meta_data.copy()

        result.update({
            'label': 'folder',
            'folder': self.parent.path,
            'full_name': self.name.join(['/'] * 2),
            'empty': self.empty,
            })

        for k in [
            'file_count',
            'folder_count'
            ]:
            attr = k.split('_')[0] + 's'
            result[k] = len(getattr(self, attr))

        return result


    #╭-------------------------------------------------------------------------╮
    #| Cached Properties                                                       |
    #╰-------------------------------------------------------------------------╯

    @cached_property
    def contents(self):
        return FolderContents(self)


    @cached_property
    def files(self):
        return Files(self)


    @cached_property
    def folders(self):
        return Subfolders(self)


    #╭-------------------------------------------------------------------------╮
    #| Magic Methods                                                           |
    #╰-------------------------------------------------------------------------╯

    def __len__(self):
        return len(os.listdir(self.path)) if self.exists else 0


    def __iter__(self):
        for x in self.sort(os.listdir(self.path)):
            yield self.join(x)


    def __add__(self, other):
        return self.join(other)


    def __getitem__(self, key):
        return self.contents[key]


    #╭-------------------------------------------------------------------------╮
    #| Instance Methods                                                        |
    #╰-------------------------------------------------------------------------╯

    @inject_read_only
    def spawn(self, *args, **kwargs):
        return type(self)(*args, **kwargs)


    @inject_read_only
    def spawn_file(self, *args, **kwargs):
        return self._spawn_file(*args, **kwargs)


    def join_folder(self, *args, **kwargs):
        return self.join(*args, require='folder', **kwargs)


    def join_file(self, *args, **kwargs):
        return self.join(*args, require='file', **kwargs)


    def join(self, *args, require='either', expected_descent=None, **kwargs):
        '''
        Description
        --------------------
        Join one or more subfolders and/or join a file.

        Parameters
        ----------
        args : tuple[str]
            Arbitrary number of strings to join to the folder.

            Behavior Examples:
            ------------------
            Consider the following folder instance:
            folder = Folder('C:/Users/Me/MyFolder')

            • Unadorned strings are treated like subfolder names:

                folder.join('A','B')
                    or
                folder.join('A/B')

                >> C:/Users/Me/MyFolder/A/B/

            • Strings preceded by a period prior to the final argument are
              treated like dot folders:

                folder.join('.A','.B','C')

                >> C:/Users/Me/MyFolder/.A/.B/C/

            • Strings preceded by a period that are also the final argument
              present a special case. It will be treated as a dot file unless
              you add a slash to signal you intend for it to be considered a
              dot folder.

                In this case, .env would be a dot folder subfile in folder B.

                                        note the slash
                                              ↓
                    folder.join('.A','B','.env/')

                Conversely, in this case, .env would be a dot file in folder B.

                    folder.join('.A','B','.env')

            • Strings separated by a period that are also the last argument
              follow the same logic:

                    folder.join('A','B','MyFile.xlsx')
                    >> C:/Users/Me/MyFolder/A/B/MyFile.xlsx

                    folder.join('A','B','C.D/')
                    >> C:/Users/Me/MyFolder/A/B/C.D/

        kwargs : dict
            spawn keyword arguments

        Returns
        ----------
        result : object
            folder or file object
        '''

        def raise_requirement_error():
            raise ValueError(
                f'join() arguments must resolve to a {require} path when '
                f'require={require!r}, got: {path!r}'
                )


        self._validate_join_args(args)

        # validate require
        (
        odd.Validator(
            types=str,
            whitelist=['either','folder','file'],
            )
        .validate(
            require=require
            )
        )

        # validate max_descent
        (
        odd.Validator(
            types=int,
            allow_none=True,
            min_value=0,
            min_inclusive=True,
            )
        .validate(
            expected_descent=expected_descent
            )
        )

        path = trifurcate_and_join(self.path + '/'.join(args))
        descent = len(self._to_parts(path)) - self.depth

        (
        odd.Validator(
            types=int,
            min_value=1,
            min_inclusive=True,
            )
        .validate(
            descent=descent
            )
        )

        path_is_file = is_file(path)

        if path_is_file:
            descent -= 1

        if expected_descent is not None and descent != expected_descent:
            raise ValueError(
                f'Join argument(s) resolve to a path {descent:,} level(s) '
                f'beneath the current path but expected {expected_descent:,}.'
                )

        if path_is_file:

            if require == 'folder':
                raise_requirement_error()

            if not self.read_only:
                # creates the folder(s) in the file path if they do not
                # already exist
                parts = path.replace(self.path, '').split('/')[:-1]

                if parts:
                    self.join_folder(*parts)

            return self.spawn_file(path, **kwargs)

        elif is_folder(path):

            if require == 'file':
                raise_requirement_error()

            return self.spawn(path, **kwargs)

        else:
            raise AssertionError(
                f'Join result is neither file nor folder: {path!r}'
                )


    @assert_writable
    def create(self):
        ''' Creates the folder if it does not already exist. Missing parents
            in the hierarchy are created as well because accessing self.parent
            calls this method as well. '''

        if not self.exists and self.parent.exists:
            create_folder(self.path)


    @assert_writable
    def delete(self):
        ''' delete the instance folder '''
        delete_folder(self.path)


    @assert_writable
    def clear(self):
        ''' deletes folder to clear it and then immediately recreates it '''
        self.delete()
        self.create()


    def copy(self, destination, overwrite=False):
        ''' copy instance folder to another folder '''
        if destination.read_only:
            raise ReadOnlyError

        if destination.exists and not overwrite:
            raise ValueError(
                "Copy failed. Folder already exists "
                f"in destination:\n'{destination}'"
                )

        destination.delete()
        shutil.copytree(self.path, destination.path)


    def walk(self):
        ''' iterates through every file in the folder and subfolders '''
        for dir_folder, dir_names, file_names in os.walk(self.path):
            for file_name in self.sort(file_names):
                path = os.path.join(dir_folder, file_name)
                yield self.spawn_file(path)


    def with_name(self, name):
        return self.parent.join_folder(name, expected_descent=1)


    def _get_parent(self, read_only):
        parts = self.parts[:-1]

        if parts:
            path = '/'.join([*parts, ''])
            return self.spawn(path, read_only=read_only)

        raise ValueError(
            f'Folder does not have a parent folder: {self}'
            )


    def _on_read_only_toggle(self):
        ''' creates the folder if read-only is toggled to False '''
        if not self.read_only:
            self.create()


    #╭-------------------------------------------------------------------------╮
    #| Static Methods                                                          |
    #╰-------------------------------------------------------------------------╯

    @staticmethod
    def _validate_join_args(args):
        ''' verify join args are valid '''

        period_rules = (
            'However, arguments may begin with or contain a single period.'
            )

        vd = odd.Validator(
            types=str,
            allow_blank=False,
            )

        for index, arg in enumerate(args):
            err_msg = f'Invalid join argument detected at index {index}:'

            vd.validate(arg, err_msg)

            if arg == '.':
                raise ValueError(
                    f"{err_msg} Single period arguments ('.') are not "
                    f"allowed. {period_rules}"
                    )

            if arg.strip() == '':
                raise ValueError(
                    f"{err_msg} Empty or whitespace-only arguments are not "
                    "allowed."
                    )

            if '..' in arg:
                raise ValueError(
                    f"{err_msg} Consecutive periods are not allowed. "
                    f"{period_rules}"
                    )

            if arg[-1] == '.':
                raise ValueError(
                    f"{err_msg} Arguments may not end in with a period "
                    f"('.'). {period_rules}"
                    )

            if arg != arg.strip():
                raise ValueError(
                    f'{err_msg} Argument ({arg!r}) contains leading or '
                    'trailing whitespace.'
                    )

            if any(x in arg for x in [':','*','?','"','<','>']):
                raise ValueError(
                    f'{err_msg} Argument ({arg!r}) contains a reserved '
                    'character.'
                    )