import numpy
import pytest

from . import rff


class TestRffHelpers:
    def test_vectorize(self):
        @rff._vectorize
        def compute_function(x):
            return numpy.atleast_2d(x) + 5

        x_1d = numpy.array([1, 2, 3, 4])
        result_1d = compute_function(x_1d)
        assert result_1d.shape == (4,)
        numpy.testing.assert_array_equal(result_1d, x_1d + 5)

        x_2d = numpy.array([
            [2, 3],
            [4, 5]
        ])
        result_2d = compute_function(x_2d)
        assert result_2d.shape == (2, 2)
        numpy.testing.assert_array_equal(result_2d, x_2d + 5)
        pass


class TestSampleRFF:
    def test_rff_1d(self):
        X = numpy.array([[0, 0.75, -0.75, -0.375, 1.125, 0.375, -1.125]]).T
        Y = numpy.array([
            0.17817326, -0.14260799, 0.13385016,
            0.44349725, -0.11945854, -0.28416883, -0.06767212
        ])
        mp = {
            'ls': numpy.array([0.33805151]),
            'scaling': numpy.array(0.87595652),
            'sigma': numpy.array(0.00041511)
        }

        f_rff, f_rff_compiled, g_rff_compiled = rff.sample_rff(
            lengthscales=mp['ls'],
            scaling=mp['scaling'],
            noise=mp['sigma'],
            kernel_nu=numpy.inf,
            X=X, Y=Y, M=200
        )

        # run checks with randomly sampled coordinates
        N = 500
        D = X.shape[1]
        X = numpy.random.uniform(low=-2, high=2, size=(N, D))

        # check the output shapes
        assert f_rff(X).shape == (N,)
        assert f_rff_compiled(X).shape == (N,)
        assert g_rff_compiled(X).shape == (N, D)

        # check correctness of gradient by manual differentiation
        epsilon = 0.000001
        grad_diff = numpy.zeros((N, D))
        for d in range(D):
            x_plus, x_minus = X.copy(), X.copy()
            x_plus[:, d] += epsilon
            x_minus[:, d] -= epsilon
            grad_diff[:, d] = (f_rff(x_plus) - f_rff(x_minus)) / (2*epsilon)
        numpy.testing.assert_allclose(g_rff_compiled(X), grad_diff, atol=1e-6)
        pass
