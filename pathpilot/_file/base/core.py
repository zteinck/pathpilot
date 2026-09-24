import shutil
import filecmp
import os

import oddments as odd

from ...path import Path
from ..._folder import Folder

from ...decorators import (
    assert_writable,
    inject_read_only,
    )

from ...utils import (
    trifurcate,
    is_file,
    )


from ..utils import get_size_label

from ._decorators import move_file
from ._temp import TempFile


class File(Path):
    '''
    Description
    --------------------
    File object.

    Class Attributes
    --------------------
    file_factory : func | None
        Function that assigns new file instances to the correct subclass.
        If None, new file instances will default to the same type as the
        spawning instance.

    Instance Attributes
    --------------------
    _folder : str
        Name of folder in which the file currently resides.
    _name : str
        File name (does not include the file extension).
    _extension : str
        The file extension (does not include the period).
    '''

    #╭-------------------------------------------------------------------------╮
    #| Initialize Instance                                                     |
    #╰-------------------------------------------------------------------------╯

    def __init__(self, path, read_only=False):

        if not is_file(path):
            raise ValueError(
                f"'path' argument must be a file, got: {path!r}"
                )

        self._folder, self._name, self._extension = trifurcate(path)
        super().__init__(read_only=read_only)


    #╭-------------------------------------------------------------------------╮
    #| Class Methods                                                           |
    #╰-------------------------------------------------------------------------╯

    @classmethod
    def _spawn_file(cls, *args, **kwargs):
        '''
        Description
        ------------
        Ensures that newly created file instances use the appropriate
        subclass. For example, if a CsvFile instance is converted to Excel,
        this method guarantees the return value is an ExcelFile instance.
        If subclass typing must be preserved regardless of changes, the
        subclass can set the factory to None.

        Note: This method must be at the class level since calling
        self.file_factory() at the instance level passes self as the first
        argument.

        Parameters
        ------------
        args : tuple
            Positional arguments passed to factory callable.
        kwargs : dict
            Keyword arguments passed to factory callable.

        Returns
        ------------
        file : File subclass instance
            spawned file instance
        '''
        ff = cls.file_factory
        return (cls if ff is None else ff)(*args, **kwargs)


    #╭-------------------------------------------------------------------------╮
    #| Properties                                                              |
    #╰-------------------------------------------------------------------------╯

    @property
    def path(self):
        ''' string representation of the file including the full folder '''
        return self._folder + self.name_with_ext


    @property
    def folder(self):
        ''' returns folder the file is currently in as a Folder object '''
        return self._get_folder(read_only=self.read_only)


    @property
    def name(self):
        return self._name


    @property
    def name_with_extension(self):
        ''' file name including file extension but exlcuding the folder '''
        return self.name + '.' + self.extension


    @property
    def name_with_ext(self):
        ''' self.name_with_extension alias '''
        return self.name_with_extension


    @property
    def full_name(self):
        ''' self.name_with_extension alias '''
        return self.name_with_extension


    @property
    def extension(self):
        ''' shorthand self.extension alias '''
        return self._extension


    @property
    def ext(self):
        ''' shorthand self.extension alias '''
        return self._extension


    @property
    def size(self):
        ''' the current size of the file expressed in bytes '''
        if self.exists:
            return os.stat(self.path).st_size


    @property
    def size_label(self):
        ''' the current size of the file expressed in bytes '''
        if self.exists:
            return get_size_label(self.size)


    @property
    def meta_data(self):
        result = super().meta_data.copy()

        result.update({
            'label': 'file',
            'folder': self.folder.path,
            'full_name': self.full_name,
            'extension': self.extension,
            'size': self.size,
            'size_label': self.size_label,
            })

        return result


    #╭-------------------------------------------------------------------------╮
    #| Magic Methods                                                           |
    #╰-------------------------------------------------------------------------╯

    def __eq__(self, other):
        a, b = self.path, str(other)
        return a == b and filecmp.cmp(a, b)


    def __ne__(self, other):
        return not self.__eq__(other)


    #╭-------------------------------------------------------------------------╮
    #| Instance Methods                                                        |
    #╰-------------------------------------------------------------------------╯

    @inject_read_only
    def spawn(self, *args, **kwargs):
        return self._spawn_file(*args, **kwargs)


    def read(self, *args, **kwargs):
        raise NotImplementedError(
            f"'{self.__class__.__name__}.read()' is not implemented."
            )


    @assert_writable
    def save(self, *args, **kwargs):
        self._save(*args, **kwargs)


    @assert_writable
    def touch(self):
        ''' create file with no content '''
        with open(self.path, mode='wb') as file:
            pass


    @assert_writable
    def delete(self):
        ''' delete file if it exists '''
        if self.exists:
            os.remove(self.path)


    def open(self):
        ''' open file in default program '''
        os.startfile(self.path)


    def trifurcate_and_fill(self, path):
        ''' trifurcates file and fills gaps with instance attributes '''
        folder, name, extension = trifurcate(path, default_folder=False)

        filled_path = ''.join((
            (folder or self.folder.path),
            (name or self.name),
            '.',
            (extension or self.extension),
            ))

        return self.spawn(filled_path)


    @move_file
    def replace(self, destination):
        '''
        Description
        ------------
        ...

        Parameters
        ------------
        ...

        Returns
        ------------
        None
        '''
        os.replace(self.path, destination.path)
        return destination


    @move_file
    def rename(self, destination):
        '''
        Description
        ------------
        Rename file.

        Parameters
        ------------
        ...

        Returns
        ------------
        None
        '''
        os.rename(self.path, destination.path)
        return destination


    @move_file
    def cut(self, destination):
        ''' cut and paste the file to a new location '''
        shutil.move(self.path, destination.path)


    @move_file
    def copy(self, destination):
        ''' copy the file to a new location '''
        shutil.copyfile(self.path, destination.path)
        return destination


    def require(self, destination):
        ''' special case of self.copy where file is copied to destination ONLY
            if it does not already exist '''
        return self.copy(
            destination,
            overwrite=False,
            raise_on_exist=False,
            raise_on_overwrite=False,
            )


    @assert_writable
    def zip(self, name=None, **kwargs):
        ''' zips a single file '''
        if not self.exists:
            raise FileNotFoundError(
                f'Cannot zip file because it does not exist:\n{self.path}'
                )

        from ..kinds import ZipFile

        zip_file = ZipFile(
            self
            .trifurcate_and_fill(name or self.path)
            .with_extension('zip')
            .path
            )

        zip_file.zip(self.path, **kwargs)

        return zip_file


    def swap(self, folder=None, name=None, extension=None, **kwargs):
        '''
        Description
        ------------
        Returns a new file instance with one or more of the current instance's
        key attributes swapped out and replaced with another value.

        Parameters
        ------------
        folder : None | str | Folder
            The folder to use for the new instance. If None, the current
            instance's folder is left intact. If a string is given, it is
            interpreted as a path.
        name : None | str
            The file name (without extension) to use for the new instance. If
            None, the current instance's name is left intact.
        extension : None | str
            The file extension to use for the new instance, with or without
            a leading period (e.g. 'txt' or '.txt'). If None, the current
            instance's extension is left intact.

        Returns
        ------------
        result : File or subclass
            A new instance of the same type as the current instance, with the
            specified attributes replaced.
        '''

        # validate folder argument
        (
        odd.Validator(
            types=(str, Folder),
            allow_none=True,
            allow_blank=False,
            require_stripped=True,
            )
        .validate(
            folder=folder
            )
        )

        # normalize folder argument
        if folder is None:
            folder = self._folder
        elif isinstance(folder, str):
            if folder[-1] != '/':
                folder += '/'
        elif isinstance(folder, Folder):
            folder = folder.path
        else:
            raise TypeError

        # normalize name argument
        (
        odd.Validator(
            types=str,
            allow_none=True,
            require_stripped=True,
            )
        .validate(
            name=name
            )
        )

        if name is None:
            name = self._name

        # normalize extension argument
        (
        odd.Validator(
            types=str,
            allow_none=True,
            allow_blank=False,
            )
        .validate(
            extension=extension
            )
        )

        extension = (
            self._extension
            if extension is None
            else extension.replace('.', '')
            )

        path = ''.join((folder, name, '.', extension))

        return self.spawn(path, **kwargs)


    @assert_writable
    def to_temp_file(self):
        return TempFile(self)


    def with_folder(self, folder):
        return self.swap(folder=folder)


    def with_name(self, name):
        return self.swap(name=name)


    def with_extension(self, extension):
        return self.swap(extension=extension)


    def with_parent(self):
        return self.folder.parent.join_file(self.name_with_ext)


    def with_sibling(self, *args, **kwargs):
        return self.folder.join_file(*args, expected_descent=0, **kwargs)


    def with_child(self, name):
        return self.with_descendant(name, expected_descent=1)


    def with_subfolder(self, name):
        return self.with_child(name)


    def with_descendant(self, *args, **kwargs):
        file = (
            self.folder
            .join_folder(*args, **kwargs)
            .join_file(self.name_with_ext)
            )

        return file


    @assert_writable
    def nest(self, *args, **kwargs):
        nested_file = self.with_descendant(*args, **kwargs)
        return self.replace(nested_file)


    @assert_writable
    def unnest(self):
        unnested_file = self.with_parent()
        return self.replace(unnested_file)


    def _save(self, *args, **kwargs):
        raise NotImplementedError(
            f"'{self.__class__.__name__}._save()' is not implemented."
            )


    def _get_folder(self, read_only):
        return Folder(self._folder, read_only=read_only)


    def _on_read_only_toggle(self):
        ''' creates the folder if read-only is toggled to False '''
        if not self.read_only:
            self.folder.create()