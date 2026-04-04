import shutil
import filecmp
import os

from ...path import Path
from ..._folder import Folder

from ...decorators import (
    check_read_only,
    inject_read_only,
    )

from ...utils import (
    trifurcate,
    is_file,
    )

from ..utils import get_size_label

from ._decorators import (
    move_file,
    add_affix,
    )


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
    directory : str
        Name of folder in which the file currently resides.
    name : str
        File name (does not include the file extension).
    extension | ext (property) : str
        The file extension (does not include the period).
    '''

    #╭-------------------------------------------------------------------------╮
    #| Initialize Instance                                                     |
    #╰-------------------------------------------------------------------------╯

    def __init__(self, path, read_only=False):
        '''
        Parameters
        ------------
        path : str | File
            File path.
        '''

        if not is_file(path):
            raise ValueError(
                f"'path' argument must be a file, got: {path!r}"
                )

        self.directory, \
        self.name, \
        self.extension = trifurcate(path)

        super().__init__(read_only=read_only)


    #╭-------------------------------------------------------------------------╮
    #| Class Methods                                                           |
    #╰-------------------------------------------------------------------------╯

    @classmethod
    def _spawn(cls, *args, **kwargs):
        '''
        Description
        ------------
        This method ensures that functions returning new instances create
        objects of the correct subclass. For instance, if a CsvFile instance is
        converted to an Excel file, this method guarantees that the returned
        instance is an ExcelFile object. If subclass typing must be preserved
        regardless of changes, the subclass can set the factory to None. For
        example, a CryptoFile should always create instances of CryptoFile even
        if the file extension is changed.

        Note: This must be at the class level since calling self.file_factory()
        at the instance level passes self as the first argument.

        Parameters
        ------------
        args : tuple
            arguments passed to factory
        kwargs : dict
            keyword arguments passed to factory

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
    def ext(self):
        ''' shorthand self.extension alias '''
        return self.extension


    @property
    def path(self):
        ''' string representation of the file including the full folder '''
        return self.directory + self.name_ext


    @property
    def name_ext(self):
        ''' file name including file extension but exlcuding the folder '''
        return self.name + '.' + self.ext


    @property
    def full_name(self):
        ''' self.name_ext alias '''
        return self.name_ext


    @property
    def folder(self):
        ''' returns folder the file is currently in as a Folder object '''
        return Folder(self.directory, read_only=self.read_only)


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
        out = super().meta_data.copy()

        out.update({
            'label': 'file',
            'directory': self.directory,
            'full_name': self.full_name,
            'extension': self.extension,
            'size': self.size,
            'size_label': self.size_label,
            })

        return out


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
        return self._spawn(*args, **kwargs)


    def read(self, *args, **kwargs):
        raise NotImplementedError(
            f"'{self.__class__.__name__}.read()' is not implemented."
            )


    def _save(self, *args, **kwargs):
        raise NotImplementedError(
            f"'{self.__class__.__name__}._save()' is not implemented."
            )


    @check_read_only
    def save(self, *args, **kwargs):
        self._save(*args, **kwargs)


    @check_read_only
    def touch(self):
        ''' create file with no content '''
        with open(self.path, mode='wb') as file:
            pass


    @check_read_only
    def delete(self):
        ''' delete file if it exists '''
        if self.exists:
            os.remove(self.path)


    def open(self, attached=False):
        ''' open file in default program '''
        os.startfile(self.path)


    def trifurcate_and_fill(self, path):
        ''' trifurcates file and fills gaps with instance attributes '''
        folder, name, ext = trifurcate(path, default_folder=False)

        filled_path = (
            (folder or self.directory)
            + (name or self.name)
            + '.'
            + (ext or self.ext)
            )

        return self.spawn(filled_path)


    @move_file
    def rename(self, destination):
        '''
        Description
        ------------
        Renames file. You can essentially use this as a cut and paste if you
        specify the new directory. You can also change the file extention if
        you wish.

        Parameters
        ------------
        ...

        Returns
        ------------
        None
        '''
        os.rename(self.path, destination.path)


    @move_file
    def cut(self, destination, **kwargs):
        ''' cut and paste the file to a new location '''
        return shutil.move(self.path, destination.path)


    @move_file
    def copy(self, destination, **kwargs):
        ''' copy the file to a new location '''
        shutil.copyfile(self.path, destination.path)


    def require(self, destination):
        ''' special case of self.copy where file is copied to destination ONLY
            if it does not already exist '''
        return self.copy(
            destination,
            overwrite=False,
            raise_on_exist=False,
            raise_on_overwrite=False
            )


    @check_read_only
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
            .swap(ext='zip')
            .path
            )

        zip_file.zip(self.path, **kwargs)

        return zip_file


    def swap(self, **kwargs):
        ''' returns a new file instance with different attribute(s) (e.g.
            you have a csv file and want an xlsx file of the same name) '''

        alias_map = {
            'folder': 'directory',
            'ext': 'extension'
            }

        kwargs = {
            alias_map.get(k, k): v
            for k, v in kwargs.items()
            }

        directory, name, extension = [
            str(kwargs.get(attr, getattr(self, attr)))
            for attr in ['directory','name','extension']
            ]

        if directory[-1] != '/':
            directory += '/'

        path = (
            directory
            + name
            + '.'
            + extension.replace('.', '')
            )

        return self.spawn(path)


    def deep_copy(self):
        ''' create a copy of the file object '''
        return self.spawn(self.path)


    @add_affix
    def prefix():
        ''' add prefix to file name '''
        pass


    @add_affix
    def suffix():
        ''' add suffix to file name '''
        pass