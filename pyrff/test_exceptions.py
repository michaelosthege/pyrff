import pytest

from . import exceptions


class TestExceptions:
    def test_dtype_error(self):
        with pytest.raises(exceptions.DtypeError) as exinfo:
            raise exceptions.DtypeError('Just the message.')
        assert 'Just' in exinfo.value.args[0]

        with pytest.raises(exceptions.DtypeError) as exinfo:
            raise exceptions.DtypeError('With types.', actual=str)
        assert 'str' in exinfo.value.args[0]

        with pytest.raises(exceptions.DtypeError) as exinfo:
            raise exceptions.DtypeError('With types.', expected=float)
        assert 'float' in exinfo.value.args[0]

        with pytest.raises(exceptions.DtypeError) as exinfo:
            raise exceptions.DtypeError('With types.', actual=int, expected=str)
        assert 'int' in exinfo.value.args[0] and 'str' in exinfo.value.args[0]
        pass

    def test_shape_error(self):
        with pytest.raises(exceptions.ShapeError) as exinfo:
            raise exceptions.ShapeError('Just the message.')
        assert 'Just' in exinfo.value.args[0]

        with pytest.raises(exceptions.ShapeError) as exinfo:
            raise exceptions.ShapeError('With shapes.', actual=(2,3))
        assert '(2, 3)' in exinfo.value.args[0]

        with pytest.raises(exceptions.ShapeError) as exinfo:
            raise exceptions.ShapeError('With shapes.', expected='(2,3) or (5,6)')
        assert '(5,6)' in exinfo.value.args[0]

        with pytest.raises(exceptions.ShapeError) as exinfo:
            raise exceptions.ShapeError('With shapes.', actual=(), expected='(5,4) or (?,?,6)')
        assert '(?,?,6)' in exinfo.value.args[0]
        pass
