from functools import cached_property
import shutil
import os

import oddments as odd

from ...path import Path
from ...exceptions import ReadOnlyError

from ...decorators import (
    check_read_only,
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
    troubleshoot : bool
        If True, some information useful in troubleshooting is printed.

    Instance Attributes
    --------------------
    _path : str
        Folder's path.
    _subfolder_cache : dict
        Cache of subfolders accessed by referencing attributes that do not
        exist. See __getattr__ documentation for more information.
    '''

    #╭-------------------------------------------------------------------------╮
    #| Class Attributes                                                        |
    #╰-------------------------------------------------------------------------╯

    sorter = odd.natural_sort
    troubleshoot = False


    #╭-------------------------------------------------------------------------╮
    #| Initialize Instance                                                     |
    #╰-------------------------------------------------------------------------╯

    def __init__(self, path=None, read_only=True):
        '''
        Parameters
        ------------
        path : str | Folder
            Folder path. if None, path will be the current working directory.
            See 'get_cwd()' documentation for more information.
        '''

        if path is None:
            path = get_cwd()

        self._path = trifurcate(path)[0]
        self._subfolder_cache = {}

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
    def hierarchy(self):
        ''' returns list representing tree hierarchy
            (e.g. 'C:/Python/Projects/' → ['C:','Python','Projects']) '''
        return list(filter(lambda x: x != '', self.path.split('/')))


    @property
    def name(self):
        ''' name of folder '''
        return self.hierarchy[-1]


    @property
    def full_name(self):
        ''' name alias (for consistency with file properties) '''
        return self.name


    @property
    def parent(self):
        ''' parent directory '''
        hierarchy = self.hierarchy

        if len(hierarchy) == 1:
            raise ValueError(
                f'Folder does not have a parent directory:\n{self}'
                )

        return self.spawn('/'.join(hierarchy[:-1]) + '/')


    @property
    def cache_key(self):
        return self._to_cache_key(self.name)


    @property
    def meta_data(self):
        out = super().meta_data.copy()

        out.update({
            'label': 'folder',
            'directory': self.parent.path,
            'full_name': self.name.join(['/'] * 2),
            'empty': self.empty,
            })

        for k in [
            'file_count',
            'folder_count'
            ]:
            attr = k.split('_')[0] + 's'
            out[k] = len(getattr(self, attr))

        return out


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


    def __getattr__(self, name):
        '''
        Description
        ------------
        Called when an attribute is referenced that does not exist.
        This implementation treats every non-existing attribute as a reference
        to a subfolder that may or may not exist. Because attributes can be
        formatted differently than their extant folder counterparts
        (e.g. self.my_folder → 'MY FOLDER/') we first need to iterate through
        the folder's current subfolder names to see if the attribute is a
        referencing a folder that already exists. After this step, if no
        matching extant folder was found and read_only is False, the referenced
        folder will be created automatically.

        Parameters
        ------------
        name : str
            subfolder name

        Returns
        ------------
        folder : Folder
            Folder object
        '''

        if self.troubleshoot:
            print(f'__getattr__(name={name!r}')

        key = self._to_cache_key(name)

        # check if the subfolder being referenced was already cached
        if key in self._subfolder_cache:
            if self.troubleshoot:
                print(f'found key = {key!r} in cache')
            return self._subfolder_cache[key]

        # check if the subfolder being referenced already exists
        for folder in self.folders:
            if key == folder.cache_key:
                if self.troubleshoot:
                    print(f'found existing subfolder = {folder.name}')
                self._subfolder_cache[key] = folder
                return self._subfolder_cache[key]

        # cache subfolder that does not exist yet
        # (it will exist now if read_only=False)
        if self.troubleshoot:
            print('could not find subfolder = {name!r}')

        self._subfolder_cache[key] = self.join(
            name.replace('_', ' ').title()
            )

        return self._subfolder_cache[key]


    #╭-------------------------------------------------------------------------╮
    #| Instance Methods                                                        |
    #╰-------------------------------------------------------------------------╯

    @inject_read_only
    def spawn(self, *args, **kwargs):
        return type(self)(*args, **kwargs)


    @inject_read_only
    def spawn_file(self, *args, **kwargs):
        return self._spawn_file(*args, **kwargs)


    def join(self, *args, **kwargs):
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
        out : object
            folder or file object
        '''
        self._validate_join_args(args)

        path = trifurcate_and_join(self.path + '/'.join(args))

        if is_file(path):
            if not self.read_only:
                # creates the folder(s) in the file path if they do not
                # already exist
                self.join(*path.replace(self.path, '').split('/')[:-1])
            return self.spawn_file(path, **kwargs)

        elif is_folder(path):
            return self.spawn(path, **kwargs)

        else:
            raise TypeError(
                f'Join result is neither file nor folder: {path!r}'
                )


    @check_read_only
    def create(self):
        ''' Creates the folder if it does not already exist. Missing parents
            in the hierarchy are created as well because accessing self.parent
            calls this method as well. '''

        if not self.exists and self.parent.exists:
            create_folder(self.path)


    @check_read_only
    def delete(self):
        ''' delete the instance folder '''
        delete_folder(self.path)
        self._clear_subfolder_cache()


    @check_read_only
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


    def _to_cache_key(self, key):
        cache_key = '_'.join(
            key
            .replace('/', '')
            .replace('\\','')
            .lower()
            .split()
            )

        return cache_key


    def _clear_subfolder_cache(self):
        self._subfolder_cache.clear()


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
            'However, arguments may begin with '
            'or contain a single period.'
            )

        for index, arg in enumerate(args):
            err_msg = f'Invalid join argument detected at index {index}:'

            odd.validate_value(
                value=arg,
                name=err_msg,
                types=str,
                empty_ok=False,
                )

            if arg == '.':
                raise ValueError(
                    f"{err_msg} Single period arguments ('.') "
                    f"are not allowed. {period_rules}"
                    )

            if arg.strip() == '':
                raise ValueError(
                    f"{err_msg} Empty or whitespace-only "
                    "arguments are not allowed."
                    )

            if '..' in arg:
                raise ValueError(
                    f"{err_msg} Consecutive periods are "
                    f"not allowed. {period_rules}"
                    )

            if arg[-1] == '.':
                raise ValueError(
                    f"{err_msg} Arguments may not end in "
                    f"with a period ('.'). {period_rules}"
                    )

            if arg != arg.strip():
                raise ValueError(
                    f'{err_msg} Argument ({arg!r}) contains '
                    'leading or trailing whitespace.'
                    )

            if any(x in arg for x in [':','*','?','"','<','>']):
                raise ValueError(
                    f'{err_msg} Argument ({arg!r}) '
                    'contains a reserved character.'
                    )