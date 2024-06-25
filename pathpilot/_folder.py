from functools import cached_property
import shutil
import os
import pandas as pd
from iterlab import natural_sort
from cachegrab import sha256

from clockwork import (
    quarter_end,
    month_end,
    day_of_week,
    year_end,
    )

from clockwork.utils import convert_date_format_to_regex

from .utils import (
    ReadOnlyError,
    check_read_only,
    trifurcate,
    trifurcate_and_join,
    is_file,
    is_folder,
    get_cwd,
    get_created_date,
    get_modified_date,
    create_folder,
    delete_folder,
    )



class Folder(object):
    '''
    Description
    --------------------
    Folder object

    Class Attributes
    --------------------
    factory : object
        Function or class that determines which subclass is returned when
        a new file object is initialized.
    sorter : func
        function used to sort folder and file objects
    troubleshoot : bool
        if True, some information useful in troubleshooting is printed

    Instance Attributes
    --------------------
    path : str
        string representation of the current folder
    read_only : bool
        if True, creating or deleting folders is disabled.
    _subfolder_cache : dict
        Cache of subfolders accessed by referencing attributes that do not exist.
        See  __getattr__ documentation below for more information.
    '''

    #╭-------------------------------------------------------------------------╮
    #| Class Attributes                                                        |
    #╰-------------------------------------------------------------------------╯

    sorter = natural_sort
    troubleshoot = False


    #╭-------------------------------------------------------------------------╮
    #| Initialize Instance                                                     |
    #╰-------------------------------------------------------------------------╯

    def __init__(self, f=None, read_only=True):
        '''
        Parameters
        ----------
        f : str
            Folder path. if None, path will be the current working directory.
            See get_cwd documentation for more information.
        read_only : bool
            see above
        '''
        self.path = trifurcate(get_cwd() if f is None else f)[0]
        self.read_only = read_only
        self._subfolder_cache = {}
        if not read_only: self.create()


    #╭-------------------------------------------------------------------------╮
    #| Classes                                                                 |
    #╰-------------------------------------------------------------------------╯

    class Contents(object):
        '''
        Description
        --------------------
        Class that enables operations on all contents of a folder collectively,
        including obtaining metadata, filtering, deleting, and more. This class
        encompasses both files and folders but you can access them separetly by
        using the Files and Folders subclasses, respectively.

        Note: This class and its subclasses are intended to be accessed indirectly
        by using the 'contents', 'folders', and 'files' Folder cached properties.

        Class Attributes
        --------------------
        None

        Instance Attributes
        --------------------
        folder : Folder
            folder object in which the contents reside
        '''

        #╭-----------------------------------╮
        #| Initialize Instance               |
        #╰-----------------------------------╯

        def __init__(self, folder):
            self.folder = folder


        #╭-----------------------------------╮
        #| Properties                        |
        #╰-----------------------------------╯

        @property
        def meta_data(self):
            df = pd.concat([
                obj.meta_data.to_frame().transpose()
                for obj in self
                ])
            df = df.infer_objects().set_index('hash_value')\
                .sort_values(['label','full_name'])
                # .sort_values('created_date', ascending=False)
            return df


        @property
        def count(self):
            ''' the number of objects that currently exist in the folder '''
            return len(self.to_list())


        @property
        def read_only(self):
            return self.folder.read_only


        #╭-----------------------------------╮
        #| Instance Methods                  |
        #╰-----------------------------------╯

        def to_list(self):
            ''' sorted list of all the objects in the folder '''
            if not self.folder.exists: return []
            return self.folder.sort(self._to_list())


        def to_dict(self):
            ''' dictionary where keys are object hash values and values are
                the objects themselves '''
            return {obj.hash_value: obj for obj in self}


        def delete(self):
            ''' delete folder contents '''
            self.folder.clear()


        def filter(
            self,
            name_pattern=None,
            case=True,
            regex=False,
            date_pattern=None,
            date_format=None,
            ext=None,
            index=None,
            sort_by=None,
            ascending=False,
            errors='raise',
            ):
            r'''
            Description
            --------------------
            Filter folder objects based on user-defined criteria.

            Parameters
            --------------------
            name_pattern : str
                If not None, only object names meeting this pattern will be considered candidates.
                This argument is passed as the pd.Series.str.contains 'pat' argument.
            case : bool
                if True, 'name_pattern' argument is treated as case sensitive.
                Passed as pd.Series.str.contains 'case' argument.
            regex
                if True, 'name_pattern' is evaluated as a regular expression, otherwise it is
                treated like a literal string. Passed as pd.Series.str.contains 'regex' argument.
            date_pattern : str
                If not None, only those object names containing a timestamp matching this regex
                pattern will be considered candidates.

                Example:
                --------
                Consider a folder that contains multiple files that include a YYYY-MM-DD timestamp
                in the name such as '2024-06-23 Budget.xlsx'. In this case you might pass
                date_pattern=r'\d{4}\-\d{2}\-\d{2}'

            date_format : str
                Defines the object names' timestamp format (e.g. '%Y-%m-%d'). There are a few
                scenarios to consider :
                    • 'date_pattern' is None ➜ an attempt will be made to derive the regex
                      pattern using 'date_format' as the template.
                    • 'date_pattern' is not None ➜ 'date_format' is optional but it can be
                      included if you do not want to rely on pd.to_datetime inferring the format.

                Example:
                --------
                Dealing with the same example from above, all the following combinations will
                achieve the same result since the None values being inferred:
                    • date_pattern=r'\d{4}\-\d{2}\-\d{2}', date_format='%Y-%m-%d'
                    • date_pattern=None, date_format='%Y-%m-%d'
                    • date_pattern=r'\d{4}\-\d{2}\-\d{2}', date_format=None

            ext : str
                If not None, only those file objects with this file extension will be considered
                candidates. This argument may be passed with or without a period and is not case
                sensitive (e.g. '.txt', 'txt,' '.TXT' are all valid).
            index : int
                If not None, the object at this integer index in the filtered and sorted data
                is returned.
            sort_by : str
                Name of self.meta_data column to sort by. This argument must be None if
                the 'date_pattern' or 'date_format' arguments are not None. Defaults to
                the object created date if None.
            ascending : bool
                If True, data is sorted in ascending order, otherwise, descending order.
                If the 'date_pattern' or 'date_format' arguments are not None, the sort
                will occur on the timestamps extracted from the object names, otherwise,
                the column denoted by the 'sort_by' argument is used.
            errors : bool
                determines how exceptions are handled:
                    • 'raise': exception is raised
                    • 'ignore': None is returned

            Returns
            ----------
            out : pd.DataFrame | Folder | FileBase
                if 'index' argument is not None, the corresponding file or folder object is
                returned. Otherwise, the filtered and sorted meta data DataFrame is returned.
            '''

            def format_date_pattern(x):
                start, end = x.startswith('('), x.endswith(')')
                if (start and not end) or (end and not start):
                    raise ValueError("'date_pattern' is malformed: '{date_pattern}'")
                return x if start and end else f'({x})'


            supported_errors = ['raise','ignore']
            if errors not in supported_errors:
                raise ValueError("'errors' argument '{errors}' not recognized. Must be in {supported_errors}")

            if sort_by is None:
                sort_by = 'created_date'
            else:
                if date_format or date_pattern:
                    raise ValueError(
                        "cannot pass 'sort_by' argument if either 'date_format' or 'date_pattern' "
                        'arguments are not None.')


            df = self.meta_data
            kind = self.__class__.__name__.lower()

            if df.empty:
                if errors == 'ignore': return
                raise ValueError(f"no {kind} exist to filter")

            df = df.sort_values(by=sort_by, ascending=ascending)

            if ext is not None:
                if kind == 'folders':
                    raise ValueError("cannot pass 'ext' argument when accessing 'folders' property.")
                ext = ext.replace('.', '').lower()
                df = df[ df['extension'] == ext ]
                if df.empty:
                    if errors == 'ignore': return
                    raise ValueError(f"no files with extension '.{ext}' found.")

            if name_pattern is not None:
                df = df[ df['name'].str.contains(pat=name_pattern, case=case, regex=regex) ]
                if df.empty:
                    if errors == 'ignore': return
                    message = [f"no {kind} matching name pattern '{name_pattern}'"]
                    if ext is not None: message.append(f"with extension '.{ext}'")
                    raise ValueError(' '.join(message + ['found.']))

            if date_format is not None and date_pattern is None:
                date_pattern = convert_date_format_to_regex(date_format)

            if date_pattern is not None:
                date_pattern = format_date_pattern(date_pattern)
                s = df['name'].str.extract(pat=date_pattern)
                if len(s.columns) != 1: raise ValueError('multiple match groups are not supported')
                s = s[ s.columns[0] ].rename('name_timestamp')

                if s.isna().all():
                    if errors == 'ignore': return
                    raise ValueError(f"timestamp extraction failed for pattern: '{date_pattern}'")

                s = pd.to_datetime(s.dropna(), format=date_format)
                df = df.join(s, how='inner').sort_values(by=s.name, ascending=ascending)

            if index is not None:
                if not isinstance(index, int):
                    raise TypeError(f"'index' argument must be an integer, not {type(index)}")
                try:
                    return self.to_dict()[ df.iloc[index].name ]
                except Exception as error:
                    if errors == 'ignore': return
                    raise error

            return df


        #╭-----------------------------------╮
        #| Instance Methods (Internal)       |
        #╰-----------------------------------╯

        def _to_list(self):
            return self.folder.folders.to_list() + \
                   self.folder.files.to_list()


        #╭-----------------------------------╮
        #| Magic Methods                     |
        #╰-----------------------------------╯

        def __repr__(self):
            return '\n'.join(obj.path.replace('/','\\') for obj in self)


        def __bool__(self):
            return self.count > 0


        def __iter__(self):
            for x in self.to_list():
                yield x


        def __len__(self):
            return self.count


        def __getitem__(self, key):
            ''' implements slicing & indexing '''
            objs = self.to_list()
            if isinstance(key, (int, slice)):
                return objs[key]
            elif isinstance(key, str):
                # return self.to_dict()[key]
                for obj in objs:
                    if key in (obj.name, obj.full_name, obj.path):
                        return obj
                raise KeyError(key)
            else:
                raise TypeError(f"Invalid argument type: '{type(key)}'")


        def __contains__(self, key):

            for obj in self.to_list():
                if key in (obj.name, obj.full_name, obj.path):
                    return True

            return False



    class Folders(Contents):
        '''
        Description
        --------------------
        Performs operations on all subfolders as a unified group.
        '''

        def __init__(self, folder):
             super().__init__(folder)


        def _to_list(self):
            return filter(is_folder, self.folder)


        def delete(self):
            for folder in self: folder.delete()
            self.folder._clear_subfolder_cache()



    class Files(Contents):
        '''
        Description
        --------------------
        Performs operations on all files as a unified group.
        '''

        def __init__(self, folder):
            super().__init__(folder)


        def _to_list(self):
            return filter(
                lambda x: x.name[:2] != '~$', # filter out lock files
                filter(is_file, self.folder))


        def delete(self):
            for file in self:
                file.delete()



    #╭-------------------------------------------------------------------------╮
    #| Class Methods                                                           |
    #╰-------------------------------------------------------------------------╯

    @classmethod
    def spawn(cls, *args, **kwargs):
        '''
        For functions that return new instances, this preserves the class.
        For example, return Folder(f) would return an instance of the parent
        class which you would not want if using a class that inherits from
        Folder '''
        return cls(*args, **kwargs)


    @classmethod
    def spawn_file(cls, f, *args, **kwargs):
        ''' initialize a new file object. You cannot just do self.spawn_file(f) because
        if passing to a function that expects one argument it also passes self '''
        return cls.factory(f, *args, **kwargs)


    @classmethod
    def sort(cls, x):
        return cls.sorter(x)


    #╭-------------------------------------------------------------------------╮
    #| Properties                                                              |
    #╰-------------------------------------------------------------------------╯

    @property
    def empty(self):
        ''' True if folder contains no files or folders otherwise False '''
        return len(self) == 0


    @property
    def exists(self):
        ''' True if folder exists '''
        return os.path.exists(self.path)


    @property
    def hierarchy(self):
        ''' returns list representing tree hierarchy
            (e.g. 'C:/Python/Projects/' ➜ ['C:','Python','Projects']) '''
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
            raise ValueError(f"'{self}' does not have a parent directory")

        parent = self.spawn(
            '/'.join(hierarchy[:-1]) + '/',
            read_only=True
            )

        return parent


    @property
    def cache_key(self):
        return self._convert_name_to_cache_key(self.name)


    @property
    def hash_value(self):
        return sha256(self.path)


    @property
    def created_date(self):
        ''' date the file was created '''
        return get_created_date(self.path)


    @property
    def modified_date(self):
        ''' date the file was modified '''
        return get_modified_date(self.path)


    @property
    def meta_data(self):

        s = pd.Series(name='meta_data', dtype='object')

        s.loc['label'] = 'folder'
        s.loc['type'] = self.__class__.__name__

        for k in [
            'hash_value',
            'path',
            'directory',
            'full_name',
            'name',
            'read_only',
            'exists',
            'empty',
            'file_count',
            'folder_count',
            'created_date',
            'modified_date',
            ]:
            if k == 'directory':
                v = self.parent.path
            elif k == 'full_name':
                v = self.name.join(['/'] * 2)
            elif 'count' in k:
                v = len(getattr(self, k.split('_')[0] + 's'))
            else:
                v = getattr(self, k)

            if 'date' in k:
                v = v.pandas

            s.loc[k] = v

        return s


    #╭-------------------------------------------------------------------------╮
    #| Cached Properties                                                       |
    #╰-------------------------------------------------------------------------╯

    @cached_property
    def contents(self):
        ''' see Contents class documentation '''
        return self.Contents(self)


    @cached_property
    def files(self):
        ''' see Contents class documentation '''
        return self.Files(self)


    @cached_property
    def folders(self):
        ''' see Contents class documentation '''
        return self.Folders(self)


    #╭-------------------------------------------------------------------------╮
    #| Magic Methods                                                           |
    #╰-------------------------------------------------------------------------╯

    def __len__(self):
        return len(os.listdir(self.path)) if self.exists else 0


    def __repr__(self):
        return self.path.replace('/','\\')


    def __str__(self):
        return self.path


    def __bool__(self):
        return self.exists


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
        --------------------
        Called when an attribute is referenced that does not exist.
        This implementation treats every non-existing attribute as a reference to a
        subfolder that may or may not exist. Because attributes can be formatted
        differently than their extant folder counterparts (e.g. self.my_folder ➜ 'MY FOLDER/')
        we first need to iterate through the folder's current subfolder names to see if the
        attribute is a referencing a folder that already exists. After this step, if no
        matching extant folder was found and read_only is False, the referenced folder
        will be created automatically.

        Parameters
        --------------------
        name : str
            subfolder name

        Returns
        ----------
        folder : Folder
            Folder object
        '''

        if self.troubleshoot: print('__getattr__ ► name =', name)
        key = self._convert_name_to_cache_key(name)

        # check if the subfolder being referenced was already cached
        if key in self._subfolder_cache:
            if self.troubleshoot: print("found key = '{key}' in cache")
            return self._subfolder_cache[key]

        # check if the subfolder being referenced already exists
        for folder in self.folders:
            if key == folder.cache_key:
                if self.troubleshoot: print(f"found existing subfolder = {folder.name}")
                self._subfolder_cache[key] = folder
                return self._subfolder_cache[key]

        # cache subfolder that does not exist yet (it will now if read_only=False)
        if self.troubleshoot: print(f"could not find subfolder = '{name}'")
        self._subfolder_cache[key] = self.join(
            name.replace('_', ' ').title())

        return self._subfolder_cache[key]



    #╭-------------------------------------------------------------------------╮
    #| Instance Methods                                                        |
    #╰-------------------------------------------------------------------------╯

    def walk(self):
        ''' iterates through every file in the folder and subfolders '''
        for dir_folder, dir_names, file_names in os.walk(self.path):
            for file_name in self.sort(file_names):
                file = self.spawn_file(
                    f=os.path.join(dir_folder, file_name),
                    read_only=self.read_only
                    )
                yield file


    def _convert_name_to_cache_key(self, key):
        return '_'.join(key.replace('/', '').replace('\\','').lower().split())


    def _clear_subfolder_cache(self):
        self._subfolder_cache.clear()


    def join(self, *args, **kwargs):
        '''
        Description
        --------------------
        Join one or more subfolders and/or join a file.

        Parameters
        ----------
        args : tuple
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

            • Strings preceded by a period prior to the final argument are treated
              like dot folders:

                folder.join('.A','.B','C')

                >> C:/Users/Me/MyFolder/.A/.B/C/

            • Strings preceded by a period that are also the final argument present
              a special case. It will be treated as a dot file unless you add a slash
              slash to signal you intend for it to be considered a dot folder.

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
        out : Folder | File
            folder or file object
        '''

        self.validate_join_args(args)

        if 'read_only' not in kwargs:
            kwargs['read_only'] = self.read_only

        f = trifurcate_and_join(self.path + '/'.join(args))

        if is_file(f):
            if not self.read_only:
                # creates the folder(s) in the file path if they do not already exist
                self.join(*f.replace(self.path,'').split('/')[:-1])
            return self.spawn_file(f, **kwargs)

        elif is_folder(f):
            return self.spawn(f, **kwargs)

        else:
            raise TypeError(f'Could not infer object. Join result {f} is neither a file or folder.')


    @check_read_only
    def create(self):
        ''' Creates the instance folder if it does not already exist.
        Missing parents in the hierarchy are created as well because
        accessing self.parent calls calls this method as well. '''

        if not self.exists and self.parent.exists:
            create_folder(self.path)


    @check_read_only
    def delete(self):
        ''' delete the instance folder '''
        delete_folder(self.path)
        self._clear_subfolder_cache()


    @check_read_only
    def clear(self):
        '''  '''
        self.delete()
        self.create()


    def copy(self, destination, overwrite=False):
        ''' copy instance folder to another folder '''
        if destination.read_only: raise ReadOnlyError
        if destination.exists and not overwrite:
            raise Exception(f"Copy failed. Folder already exists in destination:\n'{destination}'")
        destination.delete()
        shutil.copytree(self.path, destination.path)


    # helper functions adding timestamps to folders
    def quarter(self, delta=0, **kwargs):
        return self.join(quarter_end(delta=delta).label, **kwargs)


    def month(self, delta=0, **kwargs):
        return self.join(month_end(delta=delta).ymd, **kwargs)


    def day(self, weekday, delta=0, **kwargs):
        return self.join(day_of_week(weekday=weekday, delta=delta).ymd, **kwargs)


    def year(self, delta=0, **kwargs):
        return self.join(str(year_end(delta=delta).year), **kwargs)


    #╭-------------------------------------------------------------------------╮
    #| Static Methods                                                          |
    #╰-------------------------------------------------------------------------╯

    @staticmethod
    def validate_join_args(args):
        ''' verify join args are valid '''

        error_template = 'Invalid join argument detected at index %d:'
        period_rules = 'However, arguments may begin with or contain a single period.'

        for index, arg in enumerate(args):
            err_msg = error_template % index

            if not isinstance(arg, str):
                raise TypeError(f'{err_msg} Arg is of type {type(arg)}, but all args should be strings.')

            if arg == '.':
                raise ValueError(f"{err_msg} Single period arguments ('.') are not allowed. {period_rules}")

            if arg.strip() == '':
                raise ValueError(f"{err_msg} Empty or whitespace-only arguments are not allowed.")

            if '..' in arg:
                raise ValueError(f"{err_msg} Consecutive periods are not allowed. {period_rules}")

            if arg[-1] == '.':
                raise ValueError(f"{err_msg} Arguments may not end in with a period ('.'). {period_rules}")

            if arg != arg.strip():
                raise ValueError(f"{err_msg} Argument ('{arg}') contains leading or trailing whitespace.")

            if any(x in arg for x in [':','*','?','"','<','>']):
                raise ValueError(f"{err_msg} Argument ('{arg}') contains a reserved character.")