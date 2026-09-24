import oddments as odd


class TempFile(odd.ReprMixin):
    '''
    Description
    --------------------
    Temp File object.

    Class Attributes
    --------------------
    None

    Instance Attributes
    --------------------
    _dest_file : File or subclass
        Destination file. Represents the temporary file's counterpart.
    _temp_file : File or subclass
        Temporary file.
    '''

    #╭-------------------------------------------------------------------------╮
    #| Initialize Instance                                                     |
    #╰-------------------------------------------------------------------------╯

    def __init__(self, file):
        self._dest_file = file
        self._temp_file = self._spawn_temp_file()


    #╭-------------------------------------------------------------------------╮
    #| Instance Methods                                                        |
    #╰-------------------------------------------------------------------------╯

    def commit(self):
        if not self._temp_file.exists:
            raise FileNotFoundError(
                f'Temporary file does not exist: {self._temp_file}'
                )

        self.replace(self._dest_file)


    def _spawn_temp_file(self):
        temp_file = self._dest_file.with_suffix('.temp')

        if temp_file.exists:
            raise FileExistsError(
                f'Temporary file unexpectedly exists: {temp_file}'
                )

        return temp_file


    #╭-------------------------------------------------------------------------╮
    #| Magic Methods                                                           |
    #╰-------------------------------------------------------------------------╯

    def __getattr__(self, name):
        return getattr(self._temp_file, name)