from functools import wraps

from .exceptions import ReadOnlyError


def check_read_only(func):

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.read_only:
            raise ReadOnlyError
        return func(self, *args, **kwargs)

    return wrapper


def inject_read_only(func):

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        params = {'read_only': self.read_only} | kwargs
        return func(self, *args, **params)

    return wrapper