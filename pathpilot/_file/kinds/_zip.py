import shutil

import zipfile as zf

from ..base import File
from ..._folder import Folder
from ...decorators import check_read_only
from ...utils import is_file, is_folder


class ZipFile(File):

    #╭-------------------------------------------------------------------------╮
    #| Initialize Instance                                                     |
    #╰-------------------------------------------------------------------------╯

    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)


    #╭-------------------------------------------------------------------------╮
    #| Instance Methods                                                        |
    #╰-------------------------------------------------------------------------╯

    def _save(self, *args, **kwargs):
        self.zip(*args, **kwargs)


    @check_read_only
    def zip(
        self,
        paths,
        delete_source=False,
        filter_func=None,
        include_folders=False,
        files_only=False,
        ):
        '''
        Description
        ------------
        Zips one or more files and folders.

        Parameters
        ------------
        paths : str | list | tuple | File | Folder
            String representing a single file or folder path or a list or
            tuple comprised of string, File, or Folder objects representing
            files and folders to be zipped.
        delete_source : bool
            If True, source files and folders are deleted after being zipped.
        filter_func : callable
            Function applied to every file in the folder (argument will be of
            type File). If filter_func returns True for a given file it
            will be included in the zip process otherwise excluded (e.g.
            filters out files that are not of file extention .py lambda x:
            x.ext == 'py').
        include_folders : bool
            If True, each object being zipped will be placed in a folder
            mirroring the name of the folder in which the unzipped version
            resides.
            Note: This argument cannot be true when files_only is True.
        files_only : bool
            if True, zip folder will only include individual files, folders
            are disregarded. For example, if you pass a folder that contains
            subfolders, all the individual files will be zipped and subfolders
            are discarded. This is the default behavior when 'paths' contains
            files. This argument cannot be True when include_folder is True.

        Returns
        ------------
        None

        Example:
        ------------
        paths = [
            'C:/Folder A/File 1.txt',
            'C:/Folder B/File 2.txt',
            'C:/Folder C/'
            ]

        'Folder C' contains 'File 3.txt' and a subfolder 'Folder D' with a
        file called 'File 4.txt'.

        • Default (include_folders=False, files_only=False)
            Under default behavior the only time a folder will be included in
            the zip folder is when a passed folder contains subfolders

            Output: [
                'File 1.txt',
                'File 2.txt',
                'File 3.txt',
                'Folder D/File 4.txt'
                ]

        • include_folders=True
            Every file will retain the folder in which it resides

            Output: [
                'Folder A/File 1.txt',
                'Folder B/File 2.txt',
                'Folder C/File 3.txt',
                'Folder C/Folder D/File 4.txt'
                ]

        • files_only=True
            No folders are retained in the zip folder

            Output: [
            'File 1.txt',
            'File 2.txt',
            'File 3.txt',
            'File 4.txt'
            ]
        '''

        def to_object(x):

            if isinstance(x, str):
                if is_file(x):
                    x = self.spawn(x)
                elif is_folder(x):
                    x = Folder(x, read_only=True)
                else:
                    raise TypeError(
                        f'paths string is not a file or folder path: {x!r}'
                        )

            if isinstance(x, (File, Folder)):
                if x.exists:
                    return x
                else:
                    raise ValueError(
                        f'paths argument does not exist:\n{x.path}'
                        )

            raise TypeError(
                f'Invalid paths argument type: <{type(x).__name__}>'
                )


        def write_file(file, hierarchy):
            if filter_func and not filter_func(file):
                return False

            if files_only:
                arcname = file.name_ext
            else:
                if not include_folders:
                    hierarchy = hierarchy[1:]

                arcname = '/'.join([
                    *hierarchy,
                    file.name_ext
                    ])

            zip_file.write(
                filename=file.path,
                arcname=arcname
                )

            if delete_source:
                file.delete()

            if self.verbose:
                print(' ' * 4 + f'• {arcname}')

            return True


        if self.ext != 'zip':
            raise Exception(
                f'File is not a zip file: {self.name_ext!r}'
                )

        if include_folders and files_only:
            raise ValueError(
                "'include_folders' and 'files_only' "
                "arguments cannot both be True."
                )

        if not isinstance(paths, (list, tuple)):
            paths = [paths]

        paths = list(map(to_object, paths))

        zip_file = zf.ZipFile(
            file=self.path,
            mode='w',
            compression=zf.ZIP_DEFLATED
            )

        if self.verbose:
            print('Zipping:')

        for obj in paths:
            if isinstance(obj, File):
                write_file(obj, [obj.folder.name])
            else:
                folder_deletable = True

                for file in obj.walk():
                    hierarchy = (
                        file
                        .directory[:-1]
                        .replace(obj.parent.path, '')
                        .split('/')
                        )

                    zipped = write_file(file, hierarchy)

                    if not zipped:
                        folder_deletable = False

                if delete_source and folder_deletable:
                    shutil.rmtree(obj.path)


    @check_read_only
    def unzip(self, folder=None, delete_source=False):
        ''' unzips file '''

        folder = (
            self.directory
            if folder is None
            else str(folder)
            )

        zip_file = zf.ZipFile(
            file=self.path,
            mode='r'
            )

        with zip_file as z:
            z.extractall(path=folder)

        if delete_source:
            self.delete()