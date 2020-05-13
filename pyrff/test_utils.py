import numpy
import pytest

from . import utils


class TestOptimization:
    def test_multi_start_fmin(self):
        def fun(x):
            """ 2D function with the global minimum at [0, 0]. """
            x, y = x
            return numpy.cos(x + numpy.pi) + numpy.cos(y + numpy.pi) + (x / 5)**2 + (y / 8)**2
        assert fun([0, 0]) == -2.0
        x0s = numpy.random.uniform(-10, 10, size=(100, 2))
        x_best, y_best = utils.multi_start_fmin(fun, x0s)
        assert numpy.shape(x_best) == (2,)
        assert numpy.isscalar(y_best)
        numpy.testing.assert_array_almost_equal(x_best, [0, 0], decimal=4)
        assert numpy.allclose(y_best, -2)
        pass
