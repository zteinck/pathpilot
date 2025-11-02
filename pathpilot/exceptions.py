

class ReadOnlyError(Exception):

    def __init__(self, message=None):
        default = 'cannot perform this action in read-only mode'
        super().__init__(message or default)