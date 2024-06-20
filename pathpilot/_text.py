from ._file import FileBase



class TextFile(FileBase):

    def __init__(self, f, **kwargs):
        super().__init__(f, **kwargs)


    def read(self, mode='r'):
        encoding = 'utf-8' if mode != 'rb' else None
        with open(self.path, mode=mode, encoding=encoding) as file:
            text = file.read()
        file.close()
        return text


    def _save(self, text, mode='w'):
        encoding = 'utf-8' if mode != 'wb' else None
        with open(self.path, mode=mode, encoding=encoding) as file:
            file.write(text)
        file.close()