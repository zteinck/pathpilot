


class ReadOnlyError(Exception):

    def __init__(self, message=None):
        super().__init__(message or 'cannot perform this action in read-only mode')