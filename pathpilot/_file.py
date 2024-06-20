import zipfile
import shutil
import filecmp
import os
import pandas as pd
from clockwork import (
    quarter_end, 
    month_end, 
    day_of_week, 
    year_end, 
    Date,
    )

from ._folder import Folder
from .utils import (
    get_size_label, 
    check_read_only, 
    trifurcate, 
    is_file,
    )


#+---------------------------------------------------------------------------+
# Classes
#+---------------------------------------------------------------------------+

class FileBase(object):
    '''
    Description
    --------------------
    user-friendly object-oriented representation of a file

    Class Attributes
    --------------------
    sorter : func
        function that assigns new file instances to the correct class polymorphism

    Instance Attributes
    --------------------
    directory : str
        name of folder in which the file currently resides
    name : str
        file name (does not include the file extension)
    extension | ext (property) : str
        the file extension (does not include the period)
    '''

    #+---------------------------------------------------------------------------+
    # Initialize Instance
    #+---------------------------------------------------------------------------+

    def __init__(self, f, read_only=False):
        '''
        Parameters
        ----------
        f : str | File obj
            file folder
        '''
        if not is_file(f):
            raise ValueError(f'Passed argument {f} is not a file.')

        self.directory, self.name, self.extension = trifurcate(f)
        self.read_only = read_only


    #+---------------------------------------------------------------------------+
    # Classes
    #+---------------------------------------------------------------------------+

    class Decorators(object):

        @staticmethod
        def move_file(func):

            @check_read_only
            def wrapper(self, name, overwrite=True, raise_on_exist=True, raise_on_overwrite=True):
                '''
                Parameters
                ------------
                name : str | object
                    File object or str representing destination. If Folder then file is moved to that folder with the same name.
                overwrite : bool
                    if True, if the destination file already exists it will be overwritten otherwise behavior is determined by
                    the raise_on_overwrite arg. If False, the destination file is returned.
                raise_on_exist : bool
                    if True, if the file to be copied does not exist then an exception is raised
                raise_on_overwrite : bool
                    if True and overwrite is False then an exception will be raised if the destination file already exists
                '''

                f = name.join(self.fullname) if isinstance(name, Folder) else self.trifurcate_and_fill(name)
                # print(f'copying {self.path} -> {f}...')

                if not overwrite and f.exists:
                    if raise_on_overwrite:
                        raise Exception(f"Copy failed. File already exists in destination:\n'{f.fullname}'")
                    else:
                        return f

                # this is below overwite logic so so that self.require() is able to return existing file before raising if exists
                if raise_on_exist and not self.exists:
                    raise Exception(f"Copy failed. File does not exist:\n'{self}'")

                if self.exists:
                    func(self, f)

                return f

            return wrapper


        @staticmethod
        def add_timestamp(func):
            '''
            Parameters
            ------------
            args : tuple
                arguments are only passed to the respective clockwork function
            kwargs : dict
                keyword arguments are passed to the respective clockwork function with the exception of
                'loc' and 'encase':

                    loc : str
                        the location where you want the timestamp. Supported values are 'prefix' and 'suffix'
                    encase : bool
                        if True, the timestamp will be encased in paranthesis
            '''

            def wrapper(self, *args, **kwargs):
                loc = kwargs.pop('loc', 'prefix')
                encase = kwargs.pop('encase', False)
                timestamp = func(self, *args, **kwargs)
                if encase: timestamp = f'({timestamp})'
                return getattr(self, loc)(timestamp)

            return wrapper


    #+---------------------------------------------------------------------------+
    # Class Methods
    #+---------------------------------------------------------------------------+

    @classmethod
    def spawn(cls, *args, **kwargs):
        '''
        For functions that return new instances, this ensures the new instance
        is of the correct class polymorphism. For example, if a csv file gets
        changed to an excel file the sorting hat function will correctly change
        the type from CSVFile to ExcelFileBase. The class polymorphism can set
        sorter = None if the type needs to be retained no matter what. For
        example, CryptoFile should always spawn more CryptoFiles regardless of
        the file extension.
        '''
        return (cls.sorter if cls.sorter is not None else cls)(*args, **kwargs)


    #+---------------------------------------------------------------------------+
    # Properties
    #+---------------------------------------------------------------------------+

    @property
    def ext(self):
        ''' shorthand self.extension alias '''
        return self.extension


    @property
    def path(self):
        ''' string representation of the file including the full folder '''
        return self.directory + self.nameext


    @property
    def nameext(self):
        ''' file name including file extension but exlcuding the folder in string format '''
        return self.name + '.' + self.ext


    @property
    def fullname(self):
        ''' self.nameext alias '''
        return self.nameext


    @property
    def exists(self):
        ''' returns True if the file currently exists '''
        return os.path.exists(self.path)


    @property
    def folder(self):
        ''' returns folder the file is currently in as a Folder object '''
        return Folder(self.directory)


    @property
    def size(self):
        ''' the current size of the file expressed in bytes '''
        return os.stat(self.path).st_size if self.exists else None


    @property
    def size_label(self):
        ''' the current size of the file expressed in bytes '''
        return get_size_label(self.size) if self.exists else None


    @property
    def created_date(self):
        ''' date the file was created '''
        return Date(pd.to_datetime(os.path.getctime(self.path), unit='s'))


    @property
    def modified_date(self):
        ''' date the file was modified '''
        return Date(pd.to_datetime(os.path.getmtime(self.path), unit='s'))


    #+---------------------------------------------------------------------------+
    # Magic Methods
    #+---------------------------------------------------------------------------+

    def __repr__(self):
        return self.path.replace('/','\\')


    def __str__(self):
        return self.path


    def __bool__(self):
        return self.exists


    def __eq__(self, other):
        f1, f2 = self.path, str(other)
        return f1 == f2 and filecmp.cmp(f1, f2)


    def __ne__(self, other):
        f1, f2 = self.path, str(other)
        return f1 != f2 or not filecmp.cmp(f1, f2)


    #+---------------------------------------------------------------------------+
    # Instance Methods
    #+---------------------------------------------------------------------------+

    def read(self, *args, **kwargs):
        raise NotImplementedError(f"read function is not supported for files with extension '{self.ext}'")


    def _save(self, *args, **kwargs):
        raise NotImplementedError("subclass for '.{self.ext}' files must implement _save method.")


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
        if self.exists: os.remove(self.path)


    def trifurcate_and_fill(self, f):
        ''' trifurcates file and fills gaps with instance attributes '''
        folder, name, ext = trifurcate(f)
        f = (folder or self.directory) + (name or self.name) + '.' + (ext or self.ext)
        return self.spawn(f)


    @Decorators.move_file
    def rename(self, name):
        ''' Renames file. You can essentially use this as a cut and paste if you specify the new directory.
        You can also change the file extention if you wish '''
        os.rename(self.path, name.path)


    @Decorators.move_file
    def cut(self, name, **kwargs):
        ''' cut and paste the file to a new location '''
        return shutil.move(self.path, name.path)


    @Decorators.move_file
    def copy(self, name, **kwargs):
        ''' copy the file to a new location '''
        shutil.copyfile(self.path, name.path)


    def require(self, name):
        ''' special case of self.copy where file is copied to destination ONLY if it does not already exist '''
        return self.copy(name, overwrite=False, raise_on_exist=False, raise_on_overwrite=False)


    @check_read_only
    def zip(self, name=None, delete_original=False):
        ''' zips a single file '''
        f = self.trifurcate_and_fill(name or self.path)
        if f.ext != 'zip': f = self.spawn(f'{f.folder}{f.name}.zip')
        if not self.exists: raise Exception(f"'{self}' cannot be zipped because it does not exist")
        zipfile.ZipFile(f.path, 'w', zipfile.ZIP_DEFLATED).write(self.path, arcname=self.fullname)
        if delete_original: self.delete()
        return f


    @check_read_only
    def unzip(self, folder=None, delete_original=False):
        ''' unzips file '''
        if self.ext != 'zip': raise Exception(f"file '{self.nameext}' is not a zip file")
        folder = self.directory if folder is None else folder
        with zipfile.ZipFile(self.path, 'r') as f:
            f.extractall(str(folder))
        f.close()
        if delete_original: self.delete()


    def swap(self, **kwargs):
        ''' quick way of intitializing a new File with different attribute(s)
        (e.g. you have a csv file and want an xlsx file of the same name) '''
        alias_map = {'folder': 'directory', 'ext': 'extension'}
        kwargs = {alias_map.get(k, k): v for k,v in kwargs.items()}
        directory, name, extension = [str(kwargs.get(k, getattr(self, k))) for k in ['directory','name','extension']]
        if directory[-1] != '/': directory += '/'
        f = directory + name + '.' + extension.replace('.','')
        return self.spawn(f)


    def deep_copy(self):
        ''' create a copy of the File object '''
        return self.trifurcate_and_fill(str(self.path))


    def prefix(self, prefix):
        ''' add prefix to file name '''
        name = f'{prefix} {self.name}'
        return self.swap(name=name)


    def suffix(self, suffix):
        ''' add suffix to file name '''
        name = f'{self.name} {suffix}'
        return self.swap(name=name)


    # helper functions adding timestamps to files
    @Decorators.add_timestamp
    def quarter(self, delta=0):
        return quarter_end(delta=delta).label


    def qtr(self, *args, **kwargs):
        return self.quarter(*args, **kwargs)


    @Decorators.add_timestamp
    def month(self, delta=0):
        return month_end(delta=delta).sqlsvr


    @Decorators.add_timestamp
    def day(self, weekday, delta=0):
        return day_of_week(weekday=weekday, delta=delta).sqlsvr


    @Decorators.add_timestamp
    def year(self, delta=0):
        return str(year_end(delta=delta).year)


    @Decorators.add_timestamp
    def timestamp(self, normalize=False, week_offset=0, fmt=None):
        now = Date(normalize=normalize, week_offset=week_offset)
        now = str(now).replace(':','.') if fmt is None else now.str(fmt)
        return now