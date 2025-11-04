import zipfile
import shutil
import filecmp
import os
import pandas as pd
import numpy as np
import oddments as odd
from cachegrab import sha256

from .._folder import Folder
from ..decorators import check_read_only
from ..exceptions import ReadOnlyError

from ..utils import (
    trifurcate,
    is_file,
    get_created_date,
    get_modified_date,
    )

from .utils import get_size_label


class FileBase(object):
    '''
    Description
    --------------------
    File base class

    Class Attributes
    --------------------
    factory : func
        Function that assigns new file instances to the correct subclass.

    Instance Attributes
    --------------------
    directory : str
        Name of folder in which the file currently resides.
    name : str
        File name (does not include the file extension).
    extension | ext (property) : str
        The file extension (does not include the period).
    _read_only : bool
        if True, creating or deleting files is disabled.
    '''

    #╭-------------------------------------------------------------------------╮
    #| Initialize Instance                                                     |
    #╰-------------------------------------------------------------------------╯

    def __init__(self, f, read_only=False):
        '''
        Parameters
        ------------
        f : str | FileBase instance
            file folder
        '''
        if not is_file(f):
            raise ValueError(f"'f' argument is not a file: {f}")

        self.directory, self.name, self.extension = trifurcate(f)
        self.read_only = read_only


    #╭-------------------------------------------------------------------------╮
    #| Classes                                                                 |
    #╰-------------------------------------------------------------------------╯

    class Decorators(object):

        @staticmethod
        def move_file(func):

            # @check_read_only
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

                f = destination.join(self.full_name) \
                    if isinstance(destination, Folder) \
                    else self.trifurcate_and_fill(destination)

                if self.read_only and func.__name__ != 'copy':
                    raise ReadOnlyError

                if f.read_only:
                    raise ReadOnlyError(
                        f"Destination is in read-only mode: {destination.path}"
                        )

                if not overwrite and f.exists:
                    if raise_on_overwrite:
                        raise ValueError(
                            "Copy failed. File already exists in destination:\n"
                            f"'{f.full_name}'"
                            )
                    else:
                        return f

                # this is below overwite logic so so that self.require() is
                # able to return existing file before raising if exists
                if raise_on_exist and not self.exists:
                    raise FileNotFoundError(
                        f"Copy failed. File does not exist:\n'{self}'"
                        )

                if self.exists:
                    func(self, f)

                return f

            return wrapper


        @staticmethod
        def add_affix(func):
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

            def wrapper(self, text, delimiter=' ', encase=False):
                kind = func.__name__.split('_')[-1]
                if encase: text = f'({text})'
                parts = [text, self.name]
                if kind == 'suffix': parts.reverse()
                return self.swap(name=delimiter.join(parts))

            return wrapper


    #╭-------------------------------------------------------------------------╮
    #| Class Methods                                                           |
    #╰-------------------------------------------------------------------------╯

    @classmethod
    def _spawn(cls, *args, **kwargs):
        '''
        Description
        ------------
        This method ensures that functions returning new instances create
        objects of the correct subclass. For instance, if a CSVFile instance is
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
        file : FileBase subclass instance
            spawned file instance
        '''
        ff = cls.file_factory
        return (cls if ff is None else ff)(*args, **kwargs)


    #╭-------------------------------------------------------------------------╮
    #| Properties                                                              |
    #╰-------------------------------------------------------------------------╯

    @property
    def read_only(self):
        return self._read_only


    @read_only.setter
    @odd.validate_setter(types=bool)
    def read_only():
        pass


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
    def exists(self):
        ''' returns True if the file currently exists '''
        return os.path.exists(self.path)


    @property
    def folder(self):
        ''' returns folder the file is currently in as a Folder object '''
        return Folder(self.directory, read_only=self.read_only)


    @property
    def size(self):
        ''' the current size of the file expressed in bytes '''
        return os.stat(self.path).st_size


    @property
    def size_label(self):
        ''' the current size of the file expressed in bytes '''
        return get_size_label(self.size)


    @property
    def created_date(self):
        ''' date the file was created '''
        return get_created_date(self.path)


    @property
    def modified_date(self):
        ''' date the file was modified '''
        return get_modified_date(self.path)


    @property
    def hash_value(self):
        return sha256(self.path)


    @property
    def meta_data(self):
        s = pd.Series(name='meta_data')

        s.loc['label'] = 'file'
        s.loc['type'] = self.__class__.__name__

        for k in [
            'hash_value',
            'path',
            'directory',
            'full_name',
            'name',
            'extension',
            'read_only',
            'exists',
            ]:
            s.loc[k] = getattr(self, k)

        for k in [
            'created_date',
            'modified_date',
            'size',
            'size_label',
            ]:
            if s.loc['exists']:
                v = getattr(self, k)
                if 'date' in k:
                    v = v.to_pandas_timestamp()
            else:
                v = np.nan

            s.loc[k] = v

        return s


    #╭-------------------------------------------------------------------------╮
    #| Magic Methods                                                           |
    #╰-------------------------------------------------------------------------╯

    def __repr__(self):
        return self.path.replace('/','\\')


    def __str__(self):
        return self.path


    def __bool__(self):
        return self.exists


    def __eq__(self, other):
        a, b = self.path, str(other)
        return a == b and filecmp.cmp(a, b)


    def __ne__(self, other):
        return not self.__eq__(other)


    #╭-------------------------------------------------------------------------╮
    #| Instance Methods                                                        |
    #╰-------------------------------------------------------------------------╯

    def spawn(self, *args, **kwargs):
        kwargs.setdefault('read_only', self.read_only)
        return self._spawn(*args, **kwargs)


    def read(self, *args, **kwargs):
        raise NotImplementedError(
            'read() method is not supported for files with extension: '
            f"'.{self.ext}'"
            )


    def _save(self, *args, **kwargs):
        raise NotImplementedError(
            'subclass must implement _save() method for files with extension: '
            f"'.{self.ext}'"
            )


    @check_read_only
    def save(self, *args, **kwargs):
        self._save(*args, **kwargs)


    @check_read_only
    def touch(self):
        with open(self.path, mode='w') as file:
            pass


    @check_read_only
    def delete(self):
        ''' delete file if it exists '''
        if self.exists:
            os.remove(self.path)


    def open(self):
        ''' open file in default program '''
        os.startfile(self.path)


    def trifurcate_and_fill(self, f):
        ''' trifurcates file and fills gaps with instance attributes '''
        folder, name, ext = trifurcate(f, default_folder=False)
        f = (folder or self.directory) + \
            (name or self.name) + '.' + \
            (ext or self.ext)
        return self.spawn(f)


    @Decorators.move_file
    def rename(self, destination):
        ''' Renames file. You can essentially use this as a cut and paste if you
            specify the new directory. You can also change the file extention if
            you wish '''
        os.rename(self.path, destination.path)


    @Decorators.move_file
    def cut(self, destination, **kwargs):
        ''' cut and paste the file to a new location '''
        return shutil.move(self.path, destination.path)


    @Decorators.move_file
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
    def zip(self, name=None, delete_original=False):
        ''' zips a single file '''
        if not self.exists:
            raise FileNotFoundError(
                f"Cannot zip file because it does not exist: '{self}'"
                )
        f = self.trifurcate_and_fill(name or self.path)
        if f.ext != 'zip': f = f.swap(ext='zip')
        zipfile.ZipFile(f.path, 'w', zipfile.ZIP_DEFLATED)\
            .write(self.path, arcname=self.full_name)
        if delete_original:
            self.delete()
        return f


    @check_read_only
    def unzip(self, folder=None, delete_original=False):
        ''' unzips file '''
        if self.ext != 'zip':
            raise ValueError(f"file '{self.name_ext}' is not a zip file?")
        folder = self.directory if folder is None else folder
        with zipfile.ZipFile(self.path, 'r') as zip_file:
            zip_file.extractall(str(folder))
        if delete_original:
            self.delete()


    def swap(self, **kwargs):
        ''' returns a new file instance with different attribute(s) (e.g.
            you have a csv file and want an xlsx file of the same name) '''
        alias_map = {'folder': 'directory', 'ext': 'extension'}
        kwargs = {alias_map.get(k, k): v for k, v in kwargs.items()}
        directory, name, extension = [
            str(kwargs.get(k, getattr(self, k)))
            for k in ['directory','name','extension']
            ]
        if directory[-1] != '/': directory += '/'
        f = directory + name + '.' + extension.replace('.', '')
        return self.spawn(f)


    def deep_copy(self):
        ''' create a copy of the file object '''
        return self.spawn(self.path)


    @Decorators.add_affix
    def prefix():
        ''' add prefix to file name '''


    @Decorators.add_affix
    def suffix():
        ''' add suffix to file name '''