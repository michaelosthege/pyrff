"""
This module contains custom exception types.
"""


class ShapeError(Exception):
    """Error that the shape of a variable is incorrect."""
    def __init__(self, message, actual=None, expected=None):
        if actual is not None and expected is not None:
            super().__init__('{} (actual {} != expected {})'.format(message, actual, expected))
        elif actual is not None and expected is None:
            super().__init__('{} (actual {})'.format(message, actual))
        elif actual is None and expected is not None:
            super().__init__('{} (expected {})'.format(message, expected))
        else:
            super().__init__(message)


class DtypeError(TypeError):
    """Error that the dtype of a variable is incorrect."""
    def __init__(self, message, actual=None, expected=None):
        if actual is not None and expected is not None:
            super().__init__('{} (actual {} != expected {})'.format(message, actual, expected))
        elif actual is not None and expected is None:
            super().__init__('{} (actual {})'.format(message, actual))
        elif actual is None and expected is not None:
            super().__init__('{} (expected {})'.format(message, expected))
        else:
            super().__init__(message)
