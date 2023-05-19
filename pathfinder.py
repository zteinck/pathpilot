from iterkit import natural_sort, iter_get, to_iter, lower_iter
from clockwork import quarter_end, month_end, day_of_week, year_end, Date

from collections import OrderedDict
from copy import deepcopy
import os
import zipfile
import shutil
import pickle
import inspect
import filecmp
import re
import pandas as pd
import numpy as np
import sqlite3
import datetime
import time
import hashlib
import math
import sys



#+---------------------------------------------------------------------------+
# Freestanding functions
#+---------------------------------------------------------------------------+

def File(f, **kwargs):
    ''' assigns new file instances to the correct class polymorphism '''
    extension = FileBase.trifurcate(f)[-1]
    out = extension_mapping.get(extension, FileBase)(f, **kwargs)
    return out



def _get_cwd():
    ''' os.getcwd() alternative since its behavior is inconsistent. In vscode, for example,
        it returns the workspace folder rather than the subfolder '''
    # cwd = os.path.dirname(os.path.abspath(__file__)).split('\\')
    # cwd[-1] = ''
    # return '/'.join(cwd)
    return get_python_path().path



def get_python_path(*args, **kwargs):
    ''' return python path environmental variable as a Folder object '''
    return Folder(os.environ['PYTHONPATH'], *args, **kwargs)



def get_data_path():
    return get_python_path().parent.join('Data')



def get_size_label(size_in_bytes, decimal_places=2):

    units = ('','K','M','G','T','P','E','Z','Y')
    conversion_factor = 1024

    if size_in_bytes == 0:
        index, size = 0, 0
    else:
        index = int(math.floor(math.log(size_in_bytes, conversion_factor)))
        size = size_in_bytes / math.pow(conversion_factor, index)

    return f'{size:,.{decimal_places}f} {units[index]}B'



def backup_folder(origin, destination, overwrite=True, shallow=True, verbose=True):
    '''
    Description
    ------------
    Backs up the the folders and files in the 'origin' folder to the 'destination' folder.
    Files in 'destination' are overwritten if they are different than files of the same name in
    'origin' according to filecmp.cmp (e.g. doc.xlsx exists in both directories but the version
    in 'origin' was updated since the last time a backup was performed.

    Parameters
    ------------
    origin : str | Folder
        folder to backup
    destination : str | Folder
        backup folder
    overwrite : bool
        if True, if the destination file already exists and it is different than the origin file,
        it will be overwritten.
        If False, overlapping files are ignored.
    shallow : bool
        filecmp.cmp(f1, f2, shallow=True) shallow argument.
        "If shallow is true and the os.stat() signatures (file type, size, and modification time) of both
        files are identical, the files are taken to be equal."
        https://docs.python.org/3/library/filecmp.html
    verbose : bool
        if True, all folders and files that were backed up or overwritten are printed.

    Returns
    ------------
    None
    '''

    def format_path(path):
        path = str(path).replace('\\','/')
        if path[-1] != '/': path = path + '/'
        return path

    origin = format_path(origin)
    destination = format_path(destination)
    if not os.path.exists(destination): os.mkdir(destination)

    for path, folders, files in os.walk(origin):
        from_path = format_path(path)
        to_path = from_path.replace(origin, destination)

        for file in files:
            copy_file = False
            from_file = from_path + file
            to_file = to_path + file
            text = to_file.replace(destination, '/')

            if os.path.exists(to_file):
                if not filecmp.cmp(from_file, to_file, shallow=shallow) and overwrite:
                    action = 'Overwrite'
                    copy_file = True
            else:
                action = 'BackingUp'
                copy_file = True

            if copy_file:
                try:
                    shutil.copyfile(from_file, to_file)
                    if verbose: print(f'{action}: {text}')
                except Exception as e:
                    if verbose: print(e)

        for folder in folders:
            to_folder = to_path + folder
            text = to_folder.replace(destination, '/') + '/'
            if not os.path.exists(to_folder):
                os.mkdir(to_folder)
                if verbose: print(f'BackingUp: {text}')



def purge_whitespace(func):
    ''' wrapper function that purges unwanted whitespace from a DataFrame '''

    def wrapper(*args, **kwargs):

        def strip_or_skip(x):
            try:
                return x.strip()
            except:
                return x

        df = func(*args, **kwargs)

        # clean column names by trimming leading/trailing whitespace and removing new lines and consecutive spaces
        df.rename(columns={k: ' '.join(k.split()) for k in df.columns if isinstance(k, str)}, inplace=True)

        # replace None with np.nan
        for k in df.columns:
            try:
                df[k] = df[k].fillna(np.nan)
            except:
                pass

        # trim leading/trailing whitespace and replace whitespace-only values with NaN
        for k in df.select_dtypes(include=['object']).columns:
            df[k] = df[k].replace(to_replace=r'^\s*$', value=np.nan, regex=True)

            # using the vectorized string method str.strip() is faster but object-type columns can have mixed data types
            df[k] = df[k].apply(strip_or_skip) #.str.strip()

        # df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

        return df

    return wrapper



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

    sorter = File

    def __init__(self, f):
        '''
        Parameters
        ----------
        f : str | File obj
            file folder
        '''
        if not self.is_file(f):
            raise ValueError(f'Passed argument {f} is not a file.')

        self.directory, self.name, self.extension = self.trifurcate(f)


    #+---------------------------------------------------------------------------+
    # Static Methods
    #+---------------------------------------------------------------------------+

    @staticmethod
    def split(x):
        ''' separates file extention from rest of string '''
        rest, ext = os.path.splitext(str(x))
        return rest, ext.replace('.','').lower()

    @staticmethod
    def is_file(f):
        ''' returns True if the argument is a file '''
        return all(FileBase.trifurcate(f)[1:])

    @staticmethod
    def is_folder(f):
        ''' returns True if the argument is a folder '''
        f = FileBase.trifurcate(f)
        return f[0] and not any(f[1:])

    @staticmethod
    def trifurcate_and_join(f):
        ''' split argument into its components (folder inferred if absent) and combine them into one string '''
        folder, name, ext = FileBase.trifurcate(f)
        return folder + name + '.' + ext if name and ext else folder

    @staticmethod
    def trifurcate(f):
        ''' split argument into folder, file name, and file extension components '''
        f = str(f).replace('\\','/')
        explicitly_folder = True if f[-1] == '/' else False
        f = '/'.join([x for x in f.split('/') if x])
        f = f + '/' if explicitly_folder or '.' not in f.split('/')[-1] else f
        if '.com' in f: f = '//' + f
        f = f.rsplit('/', 1)
        folder = f[0] + '/' if len(f) == 2 else _get_cwd()
        name, ext = FileBase.split(f[-1])
        return folder, name, ext.lower()



    #+---------------------------------------------------------------------------+
    # Classes
    #+---------------------------------------------------------------------------+

    class Decorators(object):

        @staticmethod
        def move_file(func):

        # classmethod works too but not using cls in the function below so not needed for now...
        # @classmethod
        # def move_file(cls, func):

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
                #print(f'copying {self.path} -> {f}...')

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
        return (cls.sorter or cls)(*args, **kwargs)



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

    def save(self, *args, **kwargs):
        raise NotImplementedError(f"save function is not supported for files with extension '{self.ext}'")

    def delete(self):
        ''' delete file if it exists '''
        if self.exists: os.remove(self.path)

    def trifurcate_and_fill(self, f):
        ''' trifurcates file and fills gaps with instance attributes '''
        folder, name, ext = self.trifurcate(f)
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

    def zip(self, name=None, delete_original=False):
        ''' zips a single file '''
        f = self.trifurcate_and_fill(name or self.path)
        if f.ext != 'zip': f = File(f'{f.folder}{f.name}.zip')
        if not self.exists: raise Exception(f"'{self}' cannot be zipped because it does not exist")
        zipfile.ZipFile(f.path, 'w', zipfile.ZIP_DEFLATED).write(self.path, arcname=self.fullname)
        if delete_original: self.delete()
        return f

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



class CSVFile(FileBase):

    def __init__(self, f):
        super().__init__(f)

    @purge_whitespace
    def read(self, **kwargs):
        df = pd.read_csv(
            self.path,
            encoding='ISO-8859-1',
            keep_default_na=False,
            **kwargs
            )
        return df

    def save(self, obj, **kwargs):
        if not hasattr(obj, 'to_csv'):
            raise NotImplementedError(f"saving objects of type '{type(obj)}' is not supported")
        obj.to_csv(self.path, **kwargs)



class TextFile(FileBase):

    def __init__(self, f):
        super().__init__(f)

    def read(self, mode='r'):
        encoding = 'utf-8' if mode != 'rb' else None
        with open(self.path, mode=mode, encoding=encoding) as file:
            text = file.read()
        file.close()
        return text

    def save(self, text, mode='w'):
        encoding = 'utf-8' if mode != 'wb' else None
        with open(self.path, mode=mode, encoding=encoding) as file:
            file.write(text)
        file.close()



class PickleFile(FileBase):

    def __init__(self, f):
        super().__init__(f)

    def read(self):
        out = pd.read_pickle(self.path)
        return out[0] if isinstance(out, tuple) and len(out) == 1 else out

    def save(self, args):
        if len(args) == 1 and hasattr(args[0], 'to_pickle'):
            args[0].to_pickle(self.path)
        else:
            with open(self.path, 'wb') as file:
                pickle.dump(args, file)
            file.close()



class ZipFile(FileBase):
    '''
    Description
    --------------------
    while FileBase allows you to zip individual files, this polymorphism's self.zip
    supports zipping multiple files and folders.
    '''

    def __init__(self, f):
        super().__init__(f)


    def save(self, *args, **kwargs):
        self.zip(*args, **kwargs)


    def zip(self, payload, delete_original=False, filter_func=None, include_folders=False, files_only=False, verbose=False):
        '''
        Description
        --------------------
        while FileBase allows you to zip individual files, this polymorphism's self.zip
        supports zipping multiple files and folders.

        Parameters
        ----------
        payload : str | list | tuple | FileBase | Folder
            str representing a single file or folder or \
            list or tuple comprised of str, FileBase, or Folder
            representing files and folders to be zipped
        delete_original : bool
            If True, zipped files/folders are deleted after being zipped
        filter_func : callable
            function applied to every file in the folder (argument will be of type FileBase).
            If filter_func returns True for a given file it will be included in the zip process
            otherwise excluded (e.g. filters out files that are not of file extention .py
            lambda x: x.ext == 'py').
        include_folders : bool
            if True, each object being zipped will be placed in a folder mirroring the name of the
            folder in which the unzipped version resides. This argument cannot be true when files_only
            is True.
        files_only : bool
            if True, zip folder will only include individual files, folders are disregarded. For example, if you
            pass a folder that contains subfolders, all the individual files will be zipped and subfolders are
            discarded. This is the default behavior when 'payload' contains files. This argument cannot be True when
            include_folder is True.
        verbose : bool
            If True, file names are printed after being zipped

        Returns
        ----------
        None

        Behavior Example:
        ----------
            payload = ['C:/Folder A/File 1.txt', 'C:/Folder B/File 2.txt', 'C:/Folder C/']
            'Folder C' contains 'File 3.txt' and a subfolder 'Folder D' with a file called 'File 4.txt'.

            • Default (include_folders=False, files_only=False)
                Under default behavior the only time a folder will be included in the zip folder is when a passed
                folder contains subfolders

                Output: ['File 1.txt', 'File 2.txt', 'File 3.txt', 'Folder D/File 4.txt']

            • include_folders=True
                Every file will retain the folder in which it resides

                Output: ['Folder A/File 1.txt', 'Folder B/File 2.txt', 'Folder C/File 3.txt', 'Folder C/Folder D/File 4.txt']

            • files_only=True
                No folders are retained in the zip folder

                Output: ['File 1.txt', 'File 2.txt', 'File 3.txt', 'File 4.txt']
        '''

        def verify_exists(func):
            def wrapper(x):
                x = func(x)
                if not x.exists:
                    raise ValueError(f"argument '{x}' cannot be zipped because it does not exist")
                return x
            return wrapper

        @verify_exists
        def infer_type(x):
            if isinstance(x, str):
                if self.is_file(x):
                    return File(x)
                elif self.is_folder(x):
                    return Folder(x, read_only=True)
                else:
                    raise TypeError(f"string argument '{x}' is not a file or folder")
            if isinstance(x, (FileBase, Folder)):
                return x
            else:
                raise TypeError(f"argument '{x}' of type '{type(x)}' is not recognized.")


        if self.ext != 'zip':
            raise Exception(f"file '{self.nameext}' is not a zip file")

        if include_folders and files_only:
            raise ValueError("'include_folders' and 'files_only' arguments cannot both be True")

        if not isinstance(payload, (list, tuple)): payload = [payload]
        payload = [infer_type(obj) for obj in payload]
        zip_obj = zipfile.ZipFile(self.path, 'w', zipfile.ZIP_DEFLATED)


        def zip_file(f, p):
            if filter_func and not filter_func(f):
                return False

            if files_only:
                arcname = f.nameext
            else:
                if not include_folders: p = p[1:]
                arcname = '/'.join(p + [f.nameext])

            zip_obj.write(f.path, arcname=arcname)
            if delete_original: f.delete()
            if verbose: print(f'\t• {arcname}')
            return True


        if verbose: print('Zipping:')
        for obj in payload:
            if isinstance(obj, FileBase):
                zip_file(obj, [obj.folder.name])
            else:
                folder_can_be_deleted = True
                for f in obj.walk():
                    zipped = zip_file(f, f.folder[:-1].replace(obj.parent.folder, '').split('/'))
                    if not zipped: folder_can_be_deleted = False

                if delete_original and folder_can_be_deleted:
                    shutil.rmtree(obj.folder)



class SQLiteFile(FileBase):

    def __init__(self, f):
        super().__init__(f)


    #+---------------------------------------------------------------------------+
    # Static Methods
    #+---------------------------------------------------------------------------+

    @staticmethod
    def insert_query(tbl_name, col_names):
        sql = 'INSERT OR IGNORE INTO {0} ({1}) VALUES ({2})'.format(
            tbl_name,
            ','.join('[%s]' % x for x in col_names),
            ','.join('?' for x in range(len(col_names)))
            )
        return sql

    @staticmethod
    def update_query(tbl_name, update_cols, where_cols, where_logic=''):
        sql = 'UPDATE {0} SET {1} WHERE {2} {3}'.format(
            tbl_name,
            ', '.join('[%s] = ?' % x for x in to_iter(update_cols)),
            ' AND '.join('[%s] = ?' % x for x in to_iter(where_cols)),
            where_logic
            )
        return sql

    @staticmethod
    def select_query(tbl_name, select_cols, where_cols, where_logic=''):
        sql = 'SELECT {1} FROM {0} WHERE {2} {3}'.format(
            tbl_name,
            ', '.join('[%s]' % x for x in to_iter(select_cols)),
            ' AND '.join('[%s] = ?' % x for x in to_iter(where_cols)),
            where_logic
            )
        return sql

    @staticmethod
    def delete_query(tbl_name, where_cols, where_logic=''):
        sql = 'DELETE FROM {0} WHERE {1} {2}'.format(
            tbl_name,
            ' AND '.join('[%s] = ?' % x for x in to_iter(where_cols)),
            where_logic
            )
        return sql


    #+---------------------------------------------------------------------------+
    # Instance Methods
    #+---------------------------------------------------------------------------+

    def delete(self, reconnect=False):
        if self.exists: os.remove(self.path)
        if reconnect: self.connect()

    def connect(self):
        ''' connect to the database '''
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.c = self.conn.cursor()

    def disconnect(self):
        ''' disconnect from the database '''
        self.conn.close()
        del self.conn
        del self.c

    def enable_foreign_keys(self):
        self.c.execute('PRAGMA foreign_keys = ON;')
        self.conn.commit()

    def tables(self):
        self.c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return list(x[0] for x in self.c.fetchall())

    def columns(self, tbl_name):
        return self.table_info(tbl_name).name.tolist()

    def table_info(self, tbl_name):
        return self.read_sql('PRAGMA table_info(%s)' % tbl_name.split('.')[-1])

    def data_types(self, tbl_name):
        mapping = {'NULL': None, 'INTEGER': int, 'REAL': float, 'TEXT': str, 'BLOB': object, 'INT': int}
        return {k.lower(): mapping[v] for k,v in self.table_info(tbl_name)[['name','type']].values}

    def drop(self, tbl_name):
        self.c.execute('DROP TABLE %s' % tbl_name)
        self.conn.commit()

    def clear(self, tbl_name):
        self.c.execute('DELETE FROM %s' % tbl_name)
        self.conn.commit()

    def vacuum(self):
        self.c.execute('VACUUM')
        self.conn.commit()

    def read_sql(self, sql, params=None, **kwargs):
        if len(sql.split()) == 1: sql = "SELECT * FROM %s" % sql
        return pd.read_sql(sql, con=self.conn, params=params)

    def column_to_set(self, tbl_name, col_name):
        return set(x[0] for x in self.c.execute("SELECT DISTINCT {1} FROM {0}".format(tbl_name,col_name)).fetchall())

    def execute(self, sql, params=()):
        self.c.execute(sql, params)
        self.conn.commit()

    def format_payload(self, tbl_name, col_names, payload):
        if isinstance(payload, tuple): payload = [payload]
        mapping = self.data_types(tbl_name)
        payload = [tuple([mapping[col](cell) if pd.notnull(cell) else None
                          for col,cell in zip(col_names, row)]) for row in payload]
        return payload

    def insert(self, tbl_name, payload, col_names=None, clear=False):
        if col_names is None: col_names = self.columns(tbl_name)
        sql = self.insert_query(tbl_name, col_names)
        payload = self.format_payload(tbl_name, col_names, payload)
        if clear: self.clear(tbl_name)
        self.c.executemany(sql,payload)
        self.conn.commit()

    def df_to_table(self, tbl_name, df, chunksize=0, clear_tbl=False, where_cols=None, where_logic=''):
        ''' performs bulk insert of dataframe; bulk update occurs if where_cols argument is passed '''

        def to_table(df):
            payload = self.format_payload(tbl_name, col_names, df.values)
            self.c.executemany(sql, payload)
            self.conn.commit()

        if clear_tbl: self.clear(tbl_name)

        if list(filter(None, list(df.index.names))): df.reset_index(inplace=True)
        df.rename(columns={x: x.lower() for x in df.columns}, inplace=True)
        col_names = [x for x in lower_iter(self.columns(tbl_name)) if x in set(list(df.columns))]
        df = df[col_names]
        if df.empty: raise RuntimeError('Dataframe and %s table have no columns in common' % tbl_name)

        if where_cols:
            where_cols = lower_iter(to_iter(where_cols))
            col_names = [x for x in col_names if x not in where_cols]
            sql = self.update_query(tbl_name, col_names, where_cols, where_logic)
            col_names.extend(where_cols)
            df = df[col_names]
        else:
            sql = self.insert_query(tbl_name,col_names)

        if chunksize:
            df = df.groupby(np.arange(len(df)) // chunksize)
            for key,chunk in df: to_table(chunk)
        else:
            to_table(df)

    def copy_as_temp(self, target_name, temp_name=None, index=None):
        ''' creates temporary table based on permanent table '''
        temp_name = temp_name or target_name
        sql = 'CREATE TEMP TABLE {0} AS SELECT * FROM {1} LIMIT 0;'.format(temp_name,target_name)
        self.c.execute(sql)
        self.conn.commit()
        if index:
            sql = 'CREATE INDEX temp.idx ON {0}({1})'.format(temp_name,','.join(to_iter(index)))
            self.c.execute(sql)
            self.conn.commit()

    def clear_tables(self, warn=True):
        response = input(f'Clear all {self.name}.sqlite tables (y/n)? this action cannot be undone: ') if warn else 'y'
        if response.lower() == 'y':
            for x in self.tables():
                self.clear(x)



class Folder(object):
    '''
    Description
    --------------------
    user-friendly object-oriented representation of a folder

    Class Attributes
    --------------------
    file_cls : obj
        This attribute determines what polymorphism is returned when
        a function returns a file object. Attribute must be one of the
        following types:
            1.) FileBase class
            2.) a class that inherits from FileBase
            3.) a function that returns either 1 or 2 above

    Instance Attributes
    --------------------
    path : str
        string representation of the current folder
    read_only : bool
        if True, creating and deleting folders is disallowed
        if False, the passed folder and subfolders will be automatically created
        if they do not already exist and folder deletion is permitted.
    verbose : bool
        if True, prints a notification when new folders are created
    '''

    file_cls = File

    def __init__(self, f=None, read_only=True, verbose=False):
        '''
        Parameters
        ----------
        f : str
            if None, folder will be the current working directory which is
            defined as the parent folder of the folder in which pathfinder.py
            resides. Note that you can get the folder of the project you are
            working in by passing f=__file__

        read_only : bool
            see above
        verbose : bool
            see above
        '''
        self.path = FileBase.trifurcate(_get_cwd() if f is None else f)[0]
        self.read_only = read_only
        self.verbose = verbose
        if not read_only: self.create()


    #+---------------------------------------------------------------------------+
    # Static Methods
    #+---------------------------------------------------------------------------+

    @staticmethod
    def create_folder(f):
        ''' create folder if it does not already exist '''
        f = str(f)
        if not os.path.exists(f):
            os.mkdir(f)

    @staticmethod
    def delete_folder(f):
        ''' delete folder if it exists '''
        f = str(f)
        if os.path.exists(f):
            shutil.rmtree(f)

    @staticmethod
    def get_object_folder(obj):
        ''' return file folder of Python object '''
        return os.path.absfolder(inspect.getfile(obj))



    #+---------------------------------------------------------------------------+
    # Classes
    #+---------------------------------------------------------------------------+

    class ReadOnlyError(Exception):
        def __init__(self, message):
            super().__init__(message)

    class PickFileError(Exception):
        def __init__(self, message):
            super().__init__(message)



    #+---------------------------------------------------------------------------+
    # Class Methods
    #+---------------------------------------------------------------------------+

    @classmethod
    def spawn(cls, *args, **kwargs):
        '''
        For functions that return new instances, this preserves the class.
        For example, return Folder(f) would return an instance of the parent
        class which you would not want if using a class that inherits from
        Folder '''
        return cls(*args, **kwargs)


    @classmethod
    def spawn_file(cls, f):
        ''' initialize a new file object. You cannot just do self.spawn_file(f) because
        if passing to a function that expects one argument it also passes self '''
        return cls.file_cls(f)


    #+---------------------------------------------------------------------------+
    # Properties
    #+---------------------------------------------------------------------------+

    @property
    def empty(self):
        ''' True if folder contains no files or folders otherwise False '''
        return True if len(os.listdir(self.path)) == 0 else False

    @property
    def exists(self):
        ''' True if folder exists '''
        return os.path.exists(self.path)

    @property
    def hierarchy(self):
        ''' returns list representing tree hierarchy (e.g. 'C:/Python/Projects/' -> ['H:','Python','Projects'] '''
        return list(filter(lambda x: x != '', self.path.split('/')))

    @property
    def name(self):
        ''' name of folder '''
        return self.hierarchy[-1]

    @property
    def parent(self):
        ''' parent directory '''
        hierarchy = self.hierarchy
        if len(hierarchy) == 1: raise Exception(f"'{self}' does not have a parent directory")
        parent = self.spawn('/'.join(hierarchy[:-1]) + '/', read_only=self.read_only)
        return parent

    @property
    def parent_name(self):
        ''' name of parent folder '''
        return self.hierarchy[-2]

    @property
    def files(self):
        ''' list of files in folder '''
        return natural_sort(list(filter(lambda x: x.name[:2] != '~$', list(filter(FileBase.is_file, self)))))

    @property
    def folders(self):
        ''' list of subfolders in folder '''
        return natural_sort(list(filter(FileBase.is_folder, self)))

    @property
    def subfolders(self):
        ''' same as self.folders except only returns previously accessed instances '''
        for k in self.keys():
            yield self[k]


    #+---------------------------------------------------------------------------+
    # Magic Methods
    #+---------------------------------------------------------------------------+

    def __repr__(self):
        return self.path

    def __str__(self):
        return self.path

    def __bool__(self):
        return self.exists

    def __iter__(self):
        for x in natural_sort(os.listdir(self.path)):
            yield self.join(x)

    def __add__(self, other):
        return self.join(other)

    def __getitem__(self, key):
        return self.__dict__.get(self.format_key(key), getattr(self, key))

    def __getattr__(self, name):
        '''
        Magic method is called when an attribute is referenced that does not exist.
        I'm intending the only non-existing attributes to be references to subfolders
        so when a name is referenced it will:
            1.) iterate through the folder's subfolders to see if the subfolder exists
            2.) if the subfolder does not exist and read_only = False the folder is created
        '''
        #print('__getattr__ name arg =', name)
        key = self.format_key(name)

        # check if referenced subfolder already exists. If so, add as attribute and return
        for folder in self.folders:
            if key == self.format_key(folder.name):
                self.subfolder(key, folder)
                return self[key]

        # create attribute for non-existing subfolder and return
        self.subfolder(key, self.join(name.title().replace('_', ' ')))
        return self[key]


    #+---------------------------------------------------------------------------+
    # Instance Methods
    #+---------------------------------------------------------------------------+

    def subfolder(self, key, folder):
        ''' creates and instance attribute retaining parents' read_only argument '''
        self.__dict__[key] = self.spawn(str(folder), read_only=self.read_only)

    def format_key(self, key):
        ''' formats attribute key '''
        return key.lower().replace(' ', '_').replace('/', '')

    def walk(self):
        ''' iterates through every file in the folder and subfolders '''
        for dir_folder, dir_names, file_names in os.walk(self.path):
            for file_name in natural_sort(file_names):
                file = self.spawn_file(os.path.join(dir_folder, file_name))
                yield file

    def iter_all_files(self):
        ''' deprecated alias for walk '''
        for file in self.walk():
            yield file

    def keys(self, folders_only=True, verbose=False):
        keys = set(self.__dict__.keys())
        folders = set([k for k in keys if isinstance(self[k], Folder)])
        if verbose:
            print('Folder Keys:' + '\n\t• '.join([''] + natural_sort(list(folders))))
            #print()
            #params = keys - folders
            #print('Meta Keys:' + '\n\t• '.join([''] + natural_sort(list(params))))
        return folders if folders_only else keys

    def join(self, *args, **kwargs):
        ''' join one or more subfolders to the folder or join a file '''
        f = FileBase.trifurcate_and_join(self.path + '/'.join(args).replace('/.','.'))
        if FileBase.is_file(f):
            return self.spawn_file(f)
        elif FileBase.is_folder(f):
            return self.spawn(f, read_only=kwargs.get('read_only', self.read_only))
        else:
            raise TypeError(f'Could not infer object. Join result {f} is neither a file or folder.')


    def create(self, *folders):
        ''' create one or more new folders '''
        if self.read_only:
            raise self.ReadOnlyError(f'Cannot write to "{self}" in read-only mode.')

        # note: this recursively creates missing folders because self.parent calls create as well
        if not self.exists and self.parent.exists:
            if self.verbose: print(f"Created folder: '{self}'")
            self.create_folder(self.path)

        if not folders: return

        if isinstance(iter_get(folders), (list, tuple)):
            folders = iter_get(folders)

        for folder in folders:
            if isinstance(folder, dict):
                folder, key = iter_get(list(folder.items()))
                old_key = self.format_key(folder)
                if old_key in self.keys():
                    self.__dict__[key] = self.__dict__.pop(old_key)
            else:
                key = self.format_key(folder)

            if key not in self.keys():
                self.subfolder(key, self.join(folder))

        if len(folders) == 1:
            return self[key]

    def rekey(self, remapping):
        ''' swap attribute key '''
        for old_key, new_key in remapping.items():
            old_key, new_key = old_key.lower(), new_key.lower()
            if old_key in self.keys():
                self.__dict__[new_key] = self.__dict__.pop(old_key)

    def delete(self, *folders):
        ''' delete one or more folders. if no folders are specified then the instance folder is deleted '''
        if self.read_only:
            raise self.ReadOnlyError(f'Cannot delete from "{self}" in read-only mode.')

        if not folders: self.delete_folder(self.path)

        keys = self.keys()
        del_keys = []

        for folder in folders:
            k = self.format_key(folder)
            if k in keys: del_keys.append(k)
            self.delete_folder(self.path + folder) # because folder may exist but not have an instance

        for k in del_keys:
            del self.__dict__[k]

    def copy(self, folder, overwrite=False):
        ''' copy instance folder to another folder '''
        if folder.exists and not overwrite:
            raise Exception(f"Copy failed. Folder already exists in destination:\n'{folder}'")
        folder.delete()
        shutil.copytree(self.path, folder.folder)


    def pick_file(self, by='created', func=max, regex=None, date_format=None, ext=None,
                  raise_on_empty=False, raise_on_none=False, case=True):
        '''
        Description
        --------------------
        Selects one or more files from a list of candidates based on user-defined criteria.

        Parameters
        ----------
        by : str
            Attribute by which file candidates will be sorted. Valid options include 'created' and 'modified'
            which refer to the date the file was created or the date the file was last modified, respectively.
        func : func
            Function used to select one file from the list of candidates. If None, all candidates are returned.
        regex : str
            Regular expression pattern. If provided, only file names meeting the pattern will be considered candidates.
        date_format : str
            If file candidate names contain some form of timestamp this argument will specify how to convert the timestamp string into
            a date used for sorting purposes. This argument must be provided in conjunction with a regex argument designed to isolate
            the timestamp string. For example, if file candidates follow a format similar to 'report (2019-09-13).xlsx' you might pass
            regex='report \((\d{4}-\d{2}-\d{2})\)', date_format='%Y-%m-%d', func=max to retrieve the latest report.
        ext : str
            File extention. If provided, only files of this type will be considered candidates. Argument may include period or not
            (e.g. '.xlsx','xlsx')
        raise_on_empty : bool
            If True, an exception will be raised when the target folder is empty
        raise_on_none : bool
            If True, an exception will be raised when no candidates are identified or no single file meets the specified criteria
        case : bool
            If True, regex will be case sensitive

        Returns
        ----------
        file(s) : str | list | File obj | None
            If func is None, the entire list of candidates is returned otherwise the output of func (usually one file).
        '''

        def strdate_to_timestamp(x):
            return time.mktime(datetime.datetime.strptime(x, date_format).timetuple())

        candidates = {}
        if regex and date_format:
            attr = strdate_to_timestamp
        else:
            if by == 'created':
                attr = os.path.getctime
            elif by == 'modified':
                attr = os.path.getmtime
            else:
                raise ValueError("'by' argument must be in ('created','modified')")

        if not self.exists:
            if raise_on_empty or raise_on_none:
                raise self.PickFileError(f'{self.path} does not exist.')
            else:
                return

        files = self.files
        if not files:
            if raise_on_empty or raise_on_none:
                raise self.PickFileError(f'{self.path} is empty.')
            else:
                return

        if ext:
            ext = ext.replace('.', '').lower()
            files = list(filter(lambda x: x.ext == ext, files))
            if not files:
                if raise_on_none:
                    raise self.PickFileError(f"No file candidates of extention '{ext}' detected.")
                else:
                    return

        if not case: regex = regex.lower()
        for file in files:
            if regex:
                file_name = ' '.join(file.nameext.split())
                if not case: file_name = file_name.lower()
                match = iter_get(re.findall(regex, file_name))
                if match is None: continue
                if date_format:
                    candidates[file.nameext] = attr(match)
                    continue

            candidates[file.nameext] = attr(str(file))

        if not candidates:
            if raise_on_none:
                raise self.PickFileError('No file candidates detected.')
            else:
                return

        if func is not None:
            if isinstance(func, int):
                idx = func
                if len(candidates) <= idx: return None
                func = lambda x: sorted(x, key=x.get, reverse=True)[idx]
            if func in (max, min):
                f = func(candidates, key=candidates.get)
            else:
                f = func(candidates)
            return self.spawn_file(self.join(f))
        else:
            # return all candidates
            return [self.spawn_file(self.join(x)) for x in candidates]


    # helper functions adding timestamps to folders
    def quarter(self, delta=0, **kwargs):
        return self.join(quarter_end(delta=delta).label, **kwargs)

    def qtr(self, delta=0, **kwargs):
        return self.quarter(delta=delta, **kwargs)

    def month(self, delta=0, **kwargs):
        return self.join(month_end(delta=delta).sqlsvr, **kwargs)

    def day(self, weekday, delta=0, **kwargs):
        return self.join(day_of_week(weekday=weekday, delta=delta).sqlsvr, **kwargs)

    def year(self, delta=0, **kwargs):
        return self.join(str(year_end(delta=delta).year), **kwargs)



class ExcelFile(FileBase):
    '''
    Description
    --------------------
    File object that also interfaces with ExcelWriter to facilitate working with Excel files.
    This documentation is for the polymorphism only. See the base class for more information.

    Class Attributes
    --------------------
    ...

    Instance Attributes
    --------------------
    writer : ExcelWriter
        pd.ExcelWriter instance
    formats : dict
        keys are strings representing names of formats and values are dictionaries
        containing format parameters. For Example, {{'bold': {'bold': True}}. These
        are intended to be frequently used formats that can be both used individually
        and as building blocks for more complex formats (see self.add_format for more
        information.)
    active_worksheet : xlsxwriter.worksheet.Worksheet
        The active worksheet
    format_cache : dict
        keys are hashes of sorted dictionaries containing format parameters and
        values are xlsxwriter.format.Format. This is done so that only one
        format instance will be created per unique format used.
    sheet_cache : dict
        keys are worksheet names and values are the actual worksheet names as
        they appear in the spreadsheet. The reasoning behind this is that the
        desired name may be truncated by the 31 character limit or have a page
        prefix added so this allows the user to access the sheet using the
        original name.
    number_tabs : bool
        if True, tabs in the workbook will have a number prefix. For example,
        'MyTab' would be appear as '1 - MyTab' in the workbook.
    page : int
        the current tab number
    verbose : bool
        if True, information is printed when a worksheet is created or written to.
    troubleshoot : bool
        if True, additional information is printed for troubleshooting purposes.

    Notes:
    --------------------
    This URL contains instructions on how to get LibreOffice to calculate formulas (0 by default)
    https://stackoverflow.com/questions/32205927/xlsxwriter-and-libreoffice-not-showing-formulas-result

    '''


    def __init__(self, f, number_tabs=False, verbose=True, troubleshoot=False):
        super().__init__(f)
        self.format_cache = dict()
        self.sheet_cache = dict()
        self.number_tabs = number_tabs
        self.page = 0
        self.verbose = verbose
        self.troubleshoot = troubleshoot



    #+---------------------------------------------------------------------------+
    # Static Methods
    #+---------------------------------------------------------------------------+

    @staticmethod
    def get_column_letter(index):
        ''' return the Excel column letter sequence for a given index (e.g. index=25 returns 'Z') '''
        letters = []
        while index >= 0:
            letters.append(chr(index % 26 + ord('A')))
            index = index // 26 - 1
        return ''.join(reversed(letters))


    @staticmethod
    def get_column_index(column):
        ''' return the Excel column index for a given letter sequence (e.g. column='Z' returns 25) '''
        index = 0
        for i, char in enumerate(reversed(column.upper())):
            index += (ord(char) - ord('A') + 1) * (26 ** i)
        return index - 1


    @staticmethod
    def parse_start_cell(start_cell):
        '''
        Description
        ------------
        Converts conventional representations of start cells into the zero-based numbering format
        required by xlsxwriter. For example, 'A1' would be converted to (0, 0).

        Parameters
        ------------
        start_cell : str | tuple | list
            Individual cell to be used as the starting position for writing data to an Excel
            worksheet. May be a standard excel range (e.g. 'A2') or a tuple of length two
            (column, row) (e.g. (1, 2) | ('A', 2)).

        Returns
        ------------
        out : tuple
            zero-based numbering row and column index (e.g. (0, 0)).
        '''
        if isinstance(start_cell, (tuple, list)):
            col, row = start_cell
            if isinstance(col, int):
                col -= 1
            else:
                col = ExcelFile.get_column_index(col)
        elif isinstance(start_cell, str):
            col = ExcelFile.get_column_index(re.findall(r'[a-zA-Z]+', start_cell)[0])
            row = re.findall(r'\d+', start_cell)[0]
        else:
            raise TypeError(f"start_cell must be of type 'str' or 'tuple' not {type(start_cell)}")

        row = int(row) - 1

        if row < 0 or col < 0:
            raise ValueError(f"invalid start_cell argument {start_cell}. Zero-based indices are not allowed.")

        return col, row


    @staticmethod
    def rgb_to_hex(red, green, blue):
        ''' convert RGB color to its hex equivalent '''
        return '#%02x%02x%02x' % (red, green, blue)



    #+---------------------------------------------------------------------------+
    # Properties
    #+---------------------------------------------------------------------------+

    @property
    def workbook(self):
        return self.writer.book


    @property
    def worksheet_names(self):
        return [worksheet.name for worksheet in self]



    #+---------------------------------------------------------------------------+
    # Magic Methods
    #+---------------------------------------------------------------------------+

    def __getattr__(self, name):
        '''
        Description
        ------------
        Used to access an object's attributes that do not exist or cannot be accessed through normal means.
        It is called when an attribute is accessed using the dot notation (.). In this case, attributes that
        do not exist will be gotten from the currently active worksheet object (e.g. '.hide_gridlines').

        Parameters
        ------------
        name : str
            xlsxwriter.worksheet.Worksheet attribute name

        Returns
        ------------
        out : ...
            xlsxwriter.worksheet.Worksheet attribute
        '''

        if self.troubleshoot:
            print(f"self.__getattr__(name='{name}')")

        if name == 'writer':
            self.writer = pd.ExcelWriter(self.path)
            if self.verbose: print(f'\nCreating {self.nameext}')
            return self.writer
        elif name == 'formats':
            self.formats = self._get_preset_formats()
            return self.formats
        elif name == 'active_worksheet':
            self.create_worksheet('Sheet1')
            return self.active_worksheet
        else:
            return getattr(self.active_worksheet, name)


    def __getitem__(self, key):
        '''
        Description
        ------------
        Used to access an object's items using the square bracket notation. It is called when an object is indexed
        or sliced with square brackets ([]). In this case, indexing and slicing is applied to writer.book.worksheets_objs
        which will return one or more xlsxwriter.worksheet.Worksheet objects.

        Parameters
        ------------
        key : int | slice
            key

        Returns
        ------------
        out : xlsxwriter.worksheet.Worksheet | list
             one or more worksheet objects
        '''
        if isinstance(key, (int, slice)):
            return self.workbook.worksheets_objs[key]
        elif isinstance(key, str):
            for worksheet in self:
                if worksheet.name in (key, self.sheet_cache.get(key)):
                    return worksheet
            raise KeyError(key)
        else:
            raise TypeError(f"Invalid argument type: '{type(key)}'")


    def __iter__(self):
        ''' iterate through worksheet objects '''
        for worksheet in self.workbook.worksheets_objs:
            yield worksheet


    def __contains__(self, key):
        ''' returns True if worksheet exists in workbook '''
        try:
            self[key]
            return True
        except:
            return False



    #+---------------------------------------------------------------------------+
    # Instance Methods
    #+---------------------------------------------------------------------------+

    @purge_whitespace
    def read(self, **kwargs):
        df = pd.read_excel(
            io=self.path,
            convert_float=False,
            keep_default_na=False,
            **kwargs
            )
        return df


    def save(self, args=None, sheets=None, **kwargs):

        if args is not None:
            for i, arg in enumerate(to_iter(args)):
                self.write_df(
                    df=arg,
                    sheet=sheets[i] if sheets else f'Sheet{i + 1}',
                    **kwargs
                    )

        self.writer.save()


    def set_active_worksheet(self, key):
        self.active_worksheet = self[key]


    def create_worksheet(self, name):
        ''' create a new worksheet and set it as the active worksheet '''
        if self.number_tabs:
            self.page += 1
            out = f'{self.page} - {name}'
        else:
            out = name[:]
        out = out[:31] # Excel has a 31 character limit
        self.active_worksheet = self.workbook.add_worksheet(out)
        self.sheet_cache[name] = out
        return out


    def write(self, start_cell, data, formatting=None, inverse=False, repeat=None, sheet=None, outer_border=False):
        '''
        Description
        ------------
        writes data to a worksheet

        Parameters
        ------------
        start_cell : str | tuple
            Individual cell range to be used as the starting position for the fill. May be
            a standard excel range (e.g. 'A2') or a tuple of length two (column, row)
            (e.g. (1, 2) | ('A', 2)).
        data : 1D list | 2D list | single data element | pd.DataFrame | pd.Series
            Data to be written to excel worksheet starting in start_cell. Each sub-list is
            treated as row data and written in sequential order. For example, if start_cell
            = 'A1' and data = [[1, 2, 3], ['a', 'b', 'c']], then row 1 and 2 will be filled
            with 1, 2, 3 and 'a', 'b', 'c', respectively.
        formatting : str | dict | one-dimensional list | one-dimensional tuple | None
            Excel formatting to be applied to data. Supported types include:
                • str -> Strings are interpreted as self.formats dictionary keys. Passing a
                         single string (e.g. formatting='bold') causes the corresponding format
                         to be be universally applied to all data cells.
                • dict -> Dictionaries are interpreted as format parameters. Passing a single
                          dictionary (e.g. formatting={'bold': True, 'num_format': '#,##0'})
                          causes the corresponding format to be be universally applied to all
                          data cells.
                • list | tuple -> formats included in a list/tuuple are applied to columns in
                          sequential order. If the length of the list/tuple is shorter than the
                           number of columns then no format is applied to the remaining columns.
                • None -> no formatting is applied
        inverse : bool
            If True, the 2D data is inverted such that each sub-list is treated as column data as
            opposed to row data under the default behavior.
            (e.g. [[1, 2, 3], ['a', 'b', 'c']] -> [[1, 'a'], [2, 'b'], [3, 'c']]
        repeat : int
            If 'data' argument is a single data element then it will be repeated or duplicated
            this number of times.
        sheet : str | int
            If None, self.active_worksheet is utilized. If self.active_worksheet has not yet been
            assigned then it is assigned to a newly created blank worksheet named 'Sheet1'.
            If not None, self.active_worksheet will be assigned via self.set_active_worksheet.
            If the worksheet does not already exist and sheet is of type 'str', it is created as
            a new blank worksheet.
        outer_border : bool
            If True, data is encased in an outer border.

        Returns
        ------------
        None
        '''

        def process_format(x):
            ''' return dictionary representing format parameters '''
            if x is None:
                return
            elif isinstance(x, (str, tuple, list)):
                name = self.add_format(x)
                return self.formats[name]
            elif isinstance(x, dict):
                return x
            else:
                raise TypeError(f"format must be of type 'str' or 'dict', not {type(x)}")


        def format_builder(col, row):
            fmt = deepcopy(formatting[col]) or dict()
            updates = dict()

            if outer_border:
                if col == 0: updates['left'] = 1
                if col == n_cols - 1: updates['right'] = 1
                if row == 0: updates['top'] = 1
                if row == n_rows - 1: updates['bottom'] = 1
                fmt.update(updates)

            return self.get_format(fmt)



        if sheet is not None:
            if sheet in self:
                self.set_active_worksheet(sheet)
            else:
                if isinstance(sheet, str):
                    self.create_worksheet(sheet)
                else:
                    raise TypeError(f"'sheet' argument must be a string if worksheet has not been created yet, not {type(sheet)}")


        if isinstance(data, (pd.DataFrame, pd.Series)):
            self.write_df(
                df=data,
                sheet=sheet,
                start_cell=start_cell,
                data_format=formatting,
                inverse=inverse,
                repeat=repeat,
                outer_border=outer_border,
                )
            return

        if self.verbose:
             size_label = get_size_label(sys.getsizeof(data))
             print(f"\twriting {size_label} to '{self.active_worksheet.name}' tab", end='... ')

        if not data:
            if self.verbose: print('SKIPPED')
            return

        start_col, start_row = self.parse_start_cell(start_cell)

        is_iter = lambda x: isinstance(x, (list, tuple))

        if not is_iter(data):
            # data is non-iterable
            data = [[data] * (repeat or 1)]
        else:
            if repeat: raise NotImplementedError
            # data is 1D iterable
            if not is_iter(data[0]):
                data = [data]

        if inverse: data = list(zip(*data))

        n_rows, n_cols = np.shape(data)

        formatting = list(map(process_format, formatting)) + ([None] * (n_cols - len(formatting))) \
                     if is_iter(formatting) else [process_format(formatting)] * n_cols

        for row_idx, row in enumerate(data):
            for col_idx, cell in enumerate(row):
                self.active_worksheet.write(
                    start_row + row_idx,
                    start_col + col_idx,
                    cell if pd.notnull(cell) else None,
                    format_builder(col_idx, row_idx)
                    )

        if self.verbose: print('DONE')



    def write_df(
        self,
        df,
        sheet=None,
        start_cell='A1',
        header_format='pandas_header',
        data_format='auto',
        column_widths='auto',
        normalize=True,
        autofilter=False,
        raise_on_empty=True,
        total_row=False,
        total_row_format=None,
        total_column=False,
        total_column_format=None,
        **kwargs
        ):
        '''
        Description
        ------------
        Writes a DataFrame to an Excel worksheet. This function is a superior alternative
        to df.to_excel() because it does not share the same limitations such as not being
        able format cells that already have a format including the index, headers, and
        cells that contain dates or datetimes.

        Parameters
        ------------
        df : pd.DataFrame | pd.Series
            DataFrame to be written to worksheet. If a Series is passed it will be
            converted to a DataFrame. Note: a copy is created so the original object
            will be unchanged.
        sheet : ^
            See self.write documentation.
        start_cell : ^
            See self.write documentation.
        header_format : ^
            See self.write 'formatting' argument documentation.
        data_format : ^
            Special cases:
            • 'auto' -> formatting is automatically applied to numeric, percent, and date fields.
            • dict -> if the dictionary keys are only comprised of 'df' index or column names then
                      the values are treated like format parameters and formatting is only applied
                      to those columns included in the keys. If not all the key values are column
                      names then the dictionary receives the default treatment outlined in the self.write
                      documentation. (e.g. {'Price': 'commas', 'Total': {'bold': True}})
            Other cases:
                see self.write 'formatting' argument documentation.
        column_widths : 'auto' | list | tuple | dict
            • 'auto' -> xlsxwriter does not support auto-fitting column widths so
                           this attempts replicate it by setting the column width
                           according to the length of the values in each column
                           (up to a certain limit).
            • list | tuple -> widths are applied to columns in sequential order.
                           If the length of the list/tuple is shorter than the
                           number of columns then the width is not set on the
                           remaining columns.
            • dict -> dictionary where keys are DataFrame column names and values
                           are column widths. Any column names excluded from the
                           dictionary will not have their widths set.
        normalize : bool
            if True, any date columns where the hours, minutes, seconds, microseconds are
            all set to zero (midnight) will be converted from a datetime to date.
        autofilter : bool
            if True, a filter will be applied to the column headers.
        raise_on_empty : bool
            if True and the 'df' argument is empty, an exception will be raised.
            if False, the 'df' columns will be written to an otherwise empty worksheet.
        total_row : bool
            if True, a row is added at the bottom reflecting the sum of each numeric column.
        total_row_format : ^
            format applied if 'total_row' is True. If None and data_format='auto', then
            the same formatting will be applied to the total row plus bold and a top border.
        total_column : bool
            if True, a column is added at the end reflecting the sum of all numeric values in
             each row.
        total_column_format : ^
            format applied if 'total_column' is True. If None and data_format='auto', then
            the same formatting will be applied to the total column plus bold and a left border.

        Returns
        ------------
        None
        '''

        # Type housekeeping
        if isinstance(df, pd.DataFrame):
            df = df.copy(deep=True)
        elif isinstance(df, pd.Series):
            df = df.to_frame()
        else:
            raise TypeError(f"'df' argument type {type(df)} not supported.")

        # kwargs housekeeping
        if kwargs.get('inverse'): raise NotImplementedError

        # Reset index
        if list(filter(None, list(df.index.names))): df.reset_index(inplace=True)

        # Check if empty
        if df.empty:
            if raise_on_empty: raise ValueError("'df' argument cannot be empty.")
            if not df.columns.tolist(): raise ValueError("'df' argument must have an index or columns")
            total_row, total_column = False, False

        # Add a total column to dataframe
        if total_column:
            total_column_name = 'Total'
            if total_column_name in df.columns: raise ValueError(f"'df' already has a column named '{total_column_name}'")
            df[total_column_name] = df.sum(axis=1, numeric_only=True)

        # Categorize columns
        numeric_columns = set(df._get_numeric_data().columns.tolist())
        percent_columns = set([k for k in numeric_columns if any(z in k.lower() for z in ['%','percent'])])
        numeric_columns -= percent_columns

        datelike_columns = set(df.select_dtypes(include=[np.datetime64]).columns.tolist())
        for k in df.columns:
            if isinstance(k, str) and \
               (any(x in k.lower() for x in ('date','time')) or k.lower()[-2:] == 'dt') and \
               str(df[k].dtype) != 'timedelta64[ns]':
                datelike_columns.add(k)

        date_columns, datetime_columns = [], []
        for k in list(datelike_columns):
            if not np.issubdtype(df[k].dtype, np.datetime64):
                try:
                    df[k] = pd.to_datetime(df[k])
                except:
                    datelike_columns.remove(k)

        for k in datelike_columns:
            if normalize and (df[k].dropna() == df[k].dropna().dt.normalize()).all():
                df[k] = df[k].dt.date
                date_columns.append(k)
            else:
                datetime_columns.append(k)

        # Parse start cell
        start_col, start_row = self.parse_start_cell(start_cell)

        # Write header
        self.write(
            start_cell=(start_col + 1, start_row + 1),
            data=df.columns.tolist(),
            formatting=header_format,
            sheet=sheet,
            **kwargs
            )

        # Force data_format to comply with the standard {column name : format}
        if isinstance(data_format, dict):
            if not all(k in df.columns for k in data_format):
                if any(isinstance(v, (list, tuple, dict)) for v in data_format.values()):
                    raise ValueError
                data_format = {k: data_format for k in df.columns}

        # Automatically determine the best formatting options for each dataframe column
        if data_format == 'auto':
            data_format = dict()

            # cascade auto formatting
            if total_row_format is None: total_row_format = 'auto'
            if total_column_format is None: total_column_format = 'auto'

            for k in numeric_columns:
                if not df[k].isna().all():
                    s = df[k].dropna().abs()
                    if s.max() >= 1000:
                        data_format[k] = 'commas' if s.sum() - s.round().sum() == 0 else 'commas_two_decimals'

            for k in percent_columns: data_format[k] = 'percent_two_decimals'
            for k in datetime_columns: data_format[k] = 'datetime'
            for k in date_columns: data_format[k] = 'date'

        if isinstance(data_format, str):
            data_format = {k: data_format for k in df.columns}

        if isinstance(data_format, (list, tuple)):
            data_format = {k: v for k,v in zip(df.columns, data_format)}

        if data_format is not None and not isinstance(data_format, dict):
            raise TypeError(f"'data_format' argument does not support type {type(data_format)}.")

        # Change total column to formulas
        if total_column:
            cols = []
            for k in numeric_columns:
                if k != total_column_name:
                    col = self.get_column_letter(start_col + df.columns.get_loc(k))
                    cols.append(col)

            cell_blocks = []
            for i in range(len(df)):
                row = start_row + i + 2
                cell_blocks.append([f'{col}{row}' for col in cols])

            df[total_column_name] = [f"=SUM({','.join(cells)}" for cells in cell_blocks]

            if total_column_format == 'auto':
                data_format[total_column_name] = \
                    self.add_format(to_iter(data_format.pop(total_column_name, [])) + ['bold','left'])
            else:
                if total_column_format:
                    if data_format:
                        data_format[total_column_name] = total_column_format
                    else:
                        data_format = {total_column_name: total_column_format}

        # Write data
        self.write(
            start_cell=(start_col + 1, start_row + 2),
            data=df.replace([np.inf, -np.inf], np.nan).where(df.notnull(), None).values.tolist(),
            formatting=[data_format.get(k) for k in df.columns] if data_format is not None else None,
            sheet=sheet,
            **kwargs
            )

        # Add a total row
        if total_row:
            total_row = []
            first_row = start_row + 2
            last_row = first_row + len(df) - 1
            for k in df.columns:
                if k in numeric_columns:
                    col = self.get_column_letter(start_col + df.columns.get_loc(k))
                    total_row.append(f'=SUM({col}{first_row}:{col}{last_row})')
                else:
                    total_row.append(None)

            if total_row_format == 'auto':
                total_row_format = []
                for k in df.columns:
                    fmt = ['bold','top']
                    if total_column and k == total_column_name: fmt.append('left')
                    if k in data_format: fmt.append(data_format[k])
                    total_row_format.append(fmt)

            if total_row_format is None and data_format is not None:
                total_row_format = [data_format.get(k) for k in df.columns]

            self.write(
                start_cell=(start_col + 1, last_row + 1),
                data=total_row,
                formatting=total_row_format,
                sheet=sheet,
                **kwargs
                )

        # Set column widths
        set_column_width = lambda i, w: self.set_column('{0}:{0}'.format(self.get_column_letter(start_col + i)), w)

        if column_widths == 'auto':
            column_widths = []
            for k in df.columns:
                name_width = len(str(k))
                value_width = df[k].astype(str).str.len().max()
                max_width = max(name_width, value_width) + 1
                column_widths.append(min(max_width, 50)) # setting cap at 50

        if isinstance(column_widths, (list, tuple)):
            for i, w in enumerate(column_widths):
                set_column_width(i, w)
        elif isinstance(column_widths, dict):
            for k, w in column_widths.items():
                set_column_width(df.columns.get_loc(k), w)
        elif isinstance(column_widths, (int, float)):
            for i in range(len(df.columns)):
                set_column_width(i, column_widths)
        else:
            raise TypeError(f"'column_widths' argument does not support type {type(column_widths)}.")

        # Set autofilter
        if autofilter:
            self.autofilter(
                start_row,
                start_col,
                start_row + len(df) + (0 if total_row is None else 1) - 1,
                start_col + len(df.columns) - 1
                )



    def fill_formula(self, start_cell, formula, limit, headers=None, formatting=None, down=True, outer_border=False):
        '''
        Description
        ------------
        fills a formula down or to the right

        Parameters
        ------------
        start_cell : str | tuple | list
            Individual cell range to be used as the starting position for the fill. May be
            a standard excel range (e.g. 'A2') or a tuple of length two (column, row)
            (e.g. (1, 2) | ('A', 2)). Similar to the formula argument, column names may include
            placeholders (e.g. '{Price}2' | ('{Price}', 2)).
        formula : str
            Excel formula to be written to the start cell and used as a fill template.
            Column names may include placeholders with header names if 'headers'
            argument is passed (e.g. '=A1+B1-{Price}1'. Placeholders make the formula
            robust to changes in header positioning.
        limit : int
            Number of rows or columns to fill.
        headers : list
            List of column header names. Required argument when placeholders are used in formula or
            start_cell.
        formatting : str
            see ExcelFile.write() argument of the same name.
        down : bool
            If True, formula is filled down otherwise it is filled right.
        outer_border : bool
            see ExcelFile.write() argument of the same name.

        Returns
        ------------
        None
        '''

        if isinstance(start_cell, (list, tuple)):
            col, row = start_cell
            if isinstance(col, int):
                col = self.get_column_letter(col - 1)
            start_cell = f'{col}{row}'

        if headers:
            header_to_column_map = {header: self.get_column_letter(index) for index, header in enumerate(headers)}
            start_cell = start_cell.format(**header_to_column_map)
            formula = formula.format(**header_to_column_map)

        components = list(set(re.findall('([a-zA-Z]+)(\d+)', formula)))
        cols, rows = zip(*components)
        cols, rows = [self.get_column_index(x) + 1 for x in cols], [int(x) for x in rows]

        build_counter = lambda x: OrderedDict((i, k) for i,k in enumerate(sorted(list(set(x)))))
        counter = build_counter(rows) if down else build_counter(cols)
        inverse_counter = {v: k for k,v in counter.items()}

        for c, r in zip(cols, rows):
            x, y = self.get_column_letter(c - 1), str(r)
            repl = x + ('{%d}' % inverse_counter[r]) if down else ('{%d}' % inverse_counter[c]) + y
            formula = formula.replace(x + y, repl)

        data = []
        for x in range(limit):
            format_args = list(counter.values())
            if not down: format_args = [self.get_column_letter(c - 1) for c in format_args]
            data.append(formula.format(*format_args))
            for k in counter: counter[k] += 1

        self.write(
            start_cell=start_cell,
            data=data,
            formatting=formatting,
            inverse=down,
            outer_border=outer_border
            )



    def add_format(self, name, fmt=None):
        '''
        Description
        ------------
        adds a dictionary entry into self.formats where the key is the desired name
        of the format and the value is a dictionary representing format parameters

        Parameters
        ------------
        name : str | tuple | list
            • str -> name of new format.
                     If 'fmt' is None, if the name is already in self.formats then
                     no action is taken. If the name does not already exist then
                     the name's components (delimited by underscores ('_')) will be
                     combined into a new format (e.g. 'bold_commas').
                     if 'fmt' is not None, if the name conflicts with an existing format
                     name, the existing format will be overwritten.
            • tuple | list -> If 'fmt' is None, components will be combined into a new
                     format. For example, self.add_format(name=['bold','commas'], fmt=None)
                     would add the following entry to self.formats:
                     {'bold_commas': {'bold': True, 'num_format': '#,##0'}

        fmt : dict | None
            • dict -> see https://xlsxwriter.readthedocs.io/format.html
            • None -> format will be constructed based on the components

        Returns
        ------------
        out : str
            name of format (in other words, the self.formats dictionary key value)
        '''
        if fmt is None:
            if isinstance(name, str):
                if name in self.formats: return name
                name = name.split('_')
            if not isinstance(name, (tuple, list)):
                raise TypeError(f"'name' arguments of type {type(name)} not supported.")

            fmt = dict()
            for k in name: fmt.update(self.formats[k])
            name = '_'.join(natural_sort(name))
        else:
            if not isinstance(name, str):
                raise TypeError(f"'name' argument must be a string if 'fmt' is not None, not {type(fmt)}")
            if not isinstance(fmt, dict):
                raise TypeError(f"'fmt' argument must be a dictionary, not {type(fmt)}")

        self.formats[name] = fmt
        return name



    def get_format(self, arg):
        '''
        Description
        ------------
        Takes an argument and returns the corresponding format object.
        The format will be added to the workbook and cached if it does not already exist.

        Parameters
        ------------
        arg : str | tuple | list | dict | None
            • str | tuple | list -> passed to add_format as 'name' argument.
            • dict -> Format parameters. The resulting format will not be available in self.formats
                      but it will still be cached and can be accessed by passing the same dictionary
                      to this function.
            • None -> None is returned

        Returns
        ------------
        out : xlsxwriter.format.Format | None
            format object
        '''
        if not arg: return

        if not isinstance(arg, dict):
            name = self.add_format(name=arg)
            arg = self.formats[name]

        sha256 = hashlib.sha256()
        sha256.update(bytes(str(sorted(list(arg.items()))), encoding='utf-8'))
        key = sha256.hexdigest()
        if key not in self.format_cache:
            self.format_cache[key] = self.workbook.add_format(arg)

        return self.format_cache[key]



    def _get_preset_formats(self):
        ''' frequently used formats '''

        header_body_presets = {
            'text_wrap': True,
            'border': True,
            'align': 'left',
            'valign': 'top'
            }

        formats = {
            # Font
            'bold': {'bold': True},

            # Alignment
            'text_wrap': {'text_wrap': True},

            # Pattern
            'highlight': {'fg_color': 'yellow'},

            # Protection
            'unlocked': {'locked': 0},

            # Number
            'commas': {'num_format': '#,##0'},
            'commas_two_decimals': {'num_format': '#,##0.00'},

            # Percent
            'percent': {'num_format': '0%'},
            'percent_one_decimals': {'num_format': '0.0%'},
            'percent_two_decimals': {'num_format': '0.00%'},

            # Date
            'date': {'num_format': 'mm/dd/yyyy'},
            'datetime': {'num_format': 'mm/dd/yyyy hh:mm:ss'},
            # 'datetime': {'num_format': 'mm/dd/yyyy hh:mm:ss.000 AM/PM'},

            # Border
            'top': {'top': True},
            'bottom': {'bottom': True},
            'left': {'left': True},
            'right': {'right': True},

            # Header
            'pandas_header': {'border': True, 'align': 'center', 'valign': 'vcenter', 'bold': True},
            'black_header': {'text_wrap': True, 'fg_color': 'black', 'border': True, 'align': 'center',
                             'valign': 'vcenter', 'bold': True, 'font_color': 'white'},
            'white_body': header_body_presets,
            'gold_body': {**header_body_presets, 'fg_color': '#FFE265'},
            'turqoise_body': {**header_body_presets, 'fg_color': '#CCFFFF'},

            # Miscellaneous
            'default_merge': {'border': True, 'align': 'top','text_wrap': True},
            'gold_wrap': {'border': True, 'align': 'top','text_wrap': True, 'fg_color': '#FFE265'},
            'bold_total': {'align': 'right', 'bold': True},
            'header': {'bold': True,'bottom': True, 'text_wrap': True},

            # Conditional
            'conditional_red': {'bg_color': '#FFC7CE', 'font_color': '#9C0006'},
            'conditional_green': {'bg_color': '#C6EFCE', 'font_color': '#006100'},
            'conditional_yellow': {'bg_color': '#FFEB9C', 'font_color': '#9C6500'},
            }

        return formats



#+---------------------------------------------------------------------------+
# Variables
#+---------------------------------------------------------------------------+

# file extension to object mapping dictionary
extension_mapping = {
    'xlsx': ExcelFile,
    'xls': ExcelFile,
    'csv': CSVFile,
    'pickle': PickleFile,
    'py': TextFile,
    'txt': TextFile,
    'zip': ZipFile,
    'sqlite': SQLiteFile,
    }



if __name__ == '__main__':
    pass