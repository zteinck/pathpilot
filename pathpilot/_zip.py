import zipfile
import shutil

from ._folder import Folder
from ._file import FileBase



class ZipFile(FileBase):
    '''
    Description
    --------------------
    while FileBase allows you to zip individual files, this polymorphism's self.zip
    supports zipping multiple files and folders.
    '''

    #╭-------------------------------------------------------------------------╮
    #| Initialize Instance                                                     |
    #╰-------------------------------------------------------------------------╯

    def __init__(self, f, **kwargs):
        super().__init__(f, **kwargs)


    #╭-------------------------------------------------------------------------╮
    #| Instance Methods                                                        |
    #╰-------------------------------------------------------------------------╯

    def _save(self, *args, **kwargs):
        self.zip(*args, **kwargs)


    def zip(
        self,
        payload,
        delete_original=False,
        filter_func=None,
        include_folders=False,
        files_only=False,
        verbose=False
        ):
        '''
        Description
        -----------
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
        -----------------
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
                    return self.spawn(x)
                elif self.is_folder(x):
                    return Folder(x, read_only=True)
                else:
                    raise TypeError(f"string argument '{x}' is not a file or folder")
            if isinstance(x, (FileBase, Folder)):
                return x
            else:
                raise TypeError(f"argument '{x}' of type '{type(x)}' is not recognized.")


        if self.ext != 'zip':
            raise Exception(f"file '{self.name_ext}' is not a zip file")

        if include_folders and files_only:
            raise ValueError("'include_folders' and 'files_only' arguments cannot both be True")

        if not isinstance(payload, (list, tuple)): payload = [payload]
        payload = [infer_type(obj) for obj in payload]
        zip_obj = zipfile.ZipFile(self.path, 'w', zipfile.ZIP_DEFLATED)


        def zip_file(f, p):
            if filter_func and not filter_func(f):
                return False

            if files_only:
                arcname = f.name_ext
            else:
                if not include_folders: p = p[1:]
                arcname = '/'.join(p + [f.name_ext])

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
                    zipped = zip_file(f, f.directory[:-1].replace(obj.parent.path, '').split('/'))
                    if not zipped: folder_can_be_deleted = False

                if delete_original and folder_can_be_deleted:
                    shutil.rmtree(obj.path)