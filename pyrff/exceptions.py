"""
This module contains custom exception types.
"""

class ShapeError(Exception):
    """Error that the shape of a variable is incorrect."""
    def __init__(self, message, actual=None, expected=None):
        if expected and actual:
            super().__init__('{} (actual {} != expected {})'.format(message, actual, expected))
        else:
            super().__init__(message)


class DtypeError(TypeError):
    """Error that the dtype of a variable is incorrect."""
    def __init__(self, message, actual=None, expected=None):
        if expected and actual:
            super().__init__('{} (actual {} != expected {})'.format(message, actual, expected))
        else:
            super().__init__(message)
