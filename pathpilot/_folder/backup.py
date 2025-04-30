import shutil
import filecmp
import os


def backup_folder(
    origin,
    destination,
    overwrite=True,
    shallow=True,
    verbose=True
    ):
    '''
    Description
    ------------
    Backs up the the folders and files in the 'origin' folder to the
    'destination' folder. Files in 'destination' are overwritten if they are
    different than files of the same name in 'origin' according to filecmp.cmp
    (e.g. doc.xlsx exists in both directories but the version in 'origin' was
    updated since the last time a backup was performed.)

    Parameters
    ------------
    origin : str | Folder
        folder to backup
    destination : str | Folder
        backup folder
    overwrite : bool
        if True, if the destination file already exists and it is different than
                 the origin file, it will be overwritten.
        If False, overlapping files are ignored.
    shallow : bool
        filecmp.cmp(f1, f2, shallow=True) shallow argument.
        "If shallow is true and the os.stat() signatures (file type, size, and
        modification time) of both files are identical, the files are taken to be
        equal."
        https://docs.python.org/3/library/filecmp.html
    verbose : bool
        if True, all folders and files that were backed up or overwritten are
        printed.

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
                if not filecmp.cmp(
                    from_file,
                    to_file,
                    shallow=shallow
                    ) and overwrite:
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