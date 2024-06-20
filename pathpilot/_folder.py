import datetime
import shutil
import inspect
import time
import os
import re
from iterlab import natural_sort, iter_get
from clockwork import (
    quarter_end, 
    month_end, 
    day_of_week, 
    year_end,
    )

from .utils import (
    ReadOnlyError,
    trifurcate, 
    trifurcate_and_join, 
    is_file, 
    is_folder, 
    get_cwd,
    )



class Folder(object):
    '''
    Description
    --------------------
    user-friendly object-oriented representation of a folder

    Class Attributes
    --------------------
    sorter : obj
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

    #+---------------------------------------------------------------------------+
    # Initialize Instance
    #+---------------------------------------------------------------------------+

    def __init__(self, f=None, read_only=True, verbose=False):
        '''
        Parameters
        ----------
        f : str
            if None, folder will be the current working directory which is
            defined as the parent folder of the folder in which pathpilot.py
            resides. Note that you can get the folder of the project you are
            working in by passing f=__file__

        read_only : bool
            see above
        verbose : bool
            see above
        '''
        self.path = trifurcate(get_cwd() if f is None else f)[0]
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


    @staticmethod
    def validate_join_args(args):
        ''' verify all join args are valid '''

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



    #+---------------------------------------------------------------------------+
    # Classes
    #+---------------------------------------------------------------------------+

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
    def spawn_file(cls, f, *args, **kwargs):
        ''' initialize a new file object. You cannot just do self.spawn_file(f) because
        if passing to a function that expects one argument it also passes self '''
        return cls.sorter(f, *args, **kwargs)


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
        return natural_sort(list(filter(lambda x: x.name[:2] != '~$', list(filter(is_file, self)))))


    @property
    def folders(self):
        ''' list of subfolders in folder '''
        return natural_sort(list(filter(is_folder, self)))


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
        # print('__getattr__ name arg =', name)
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
                file = self.spawn_file(
                    f=os.path.join(dir_folder, file_name),
                    read_only=self.read_only
                    )
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
        return folders if folders_only else keys


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


    def create(self, *folders):
        ''' create one or more new folders '''
        if self.read_only:
            raise ReadOnlyError(f'Cannot write to "{self}" in read-only mode.')

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
            raise ReadOnlyError(f'Cannot delete from "{self}" in read-only mode.')

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
        if folder.read_only: raise ReadOnlyError
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
            return self.spawn_file(self.join(f), read_only=self.read_only)
        else:
            # return all candidates
            return [self.spawn_file(self.join(x), read_only=self.read_only) for x in candidates]


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