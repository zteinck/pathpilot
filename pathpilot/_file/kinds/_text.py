import oddments as odd

from ..base import File


class TextFile(File):

    #╭-------------------------------------------------------------------------╮
    #| Initialize Instance                                                     |
    #╰-------------------------------------------------------------------------╯

    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)


    #╭-------------------------------------------------------------------------╮
    #| Static Methods                                                          |
    #╰-------------------------------------------------------------------------╯

    @staticmethod
    def _validate_mode(mode, letter):
        name = 'mode'

        odd.validate_value(
            value=mode,
            name=name,
            types=str,
            )

        if mode[0] != letter:
            raise ValueError(
                "'mode' should start with {letter!r}, got: {mode!r}"
                )


    @staticmethod
    def _get_encoding(mode):
        return None if mode[-1] == 'b' else 'utf-8'


    #╭-------------------------------------------------------------------------╮
    #| Instance Methods                                                        |
    #╰-------------------------------------------------------------------------╯

    def read(self, mode='r'):
        self._validate_mode(mode, 'r')
        encoding = self._get_encoding(mode)

        with open(
            file=self.path,
            mode=mode,
            encoding=encoding
            ) as file:
            text = file.read()

        return text


    def _save(self, text, mode='w'):
        self._validate_mode(mode, 'w')
        encoding = self._get_encoding(mode)

        with open(
            file=self.path,
            mode=mode,
            encoding=encoding
            ) as file:
            file.write(text)