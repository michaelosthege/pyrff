import pytest

from . import exceptions


class TestExceptions:
    def test_dtype_error(self):
        with pytest.raises(exceptions.DtypeError):
            raise exceptions.DtypeError('Just the message.')
        with pytest.raises(exceptions.DtypeError):
            raise exceptions.DtypeError('With types.', actual=str)
        with pytest.raises(exceptions.DtypeError):
            raise exceptions.DtypeError('With types.', expected=str)
        with pytest.raises(exceptions.DtypeError):
            raise exceptions.DtypeError('With types.', actual=int, expected=str)
        pass

    def test_shape_error(self):
        with pytest.raises(exceptions.ShapeError):
            raise exceptions.ShapeError('Just the message.')
        with pytest.raises(exceptions.ShapeError):
            raise exceptions.ShapeError('With shapes.', actual=(2,3))
        with pytest.raises(exceptions.ShapeError):
            raise exceptions.ShapeError('With shapes.', expected='(2,3) or (5,6')
        with pytest.raises(exceptions.ShapeError):
            raise exceptions.ShapeError('With shapes.', actual=(), expected='(5,4) or (?,?,6)')
        pass
