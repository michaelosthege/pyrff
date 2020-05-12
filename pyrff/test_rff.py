import h5py
import numpy
import pathlib
import pytest
import tempfile

from . import exceptions
from . import rff


class TestRffHelpers:
    def test_allow_1d_inputs(self):
        class TestClass:
            @rff._allow_1d_inputs
            def compute_function(self, x):
                return numpy.atleast_2d(x) + 5

        obj = TestClass()
        x_1d = numpy.array([1, 2, 3, 4])
        result_1d = obj.compute_function(x_1d)
        assert result_1d.shape == (4,)
        numpy.testing.assert_array_equal(result_1d, x_1d + 5)

        x_2d = numpy.array([
            [2, 3],
            [4, 5]
        ])
        result_2d = obj.compute_function(x_2d)
        assert result_2d.shape == (2, 2)
        numpy.testing.assert_array_equal(result_2d, x_2d + 5)
        pass


class TestSampleRFF:
    @pytest.mark.parametrize('kwarg,override,error_cls,message_fragment', [
        ('X', numpy.random.uniform(size=(47, 8)), exceptions.ShapeError, 'X and Y'),
        ('Y', numpy.random.normal(size=(44,)), exceptions.ShapeError, 'X and Y'),
        ('lengthscales', [1,2,3], exceptions.ShapeError, 'Lengthscales'),
        ('scaling', [12, 7], exceptions.ShapeError, 'scaling'),
        ('noise', [1,2,3], exceptions.ShapeError, 'noise'),
        ('kernel_nu', 'squared_exponential', ValueError, 'kernel_nu'),
        ('kernel_nu', -1, ValueError, 'kernel_nu'),
        ('kernel_nu', 0, ValueError, 'kernel_nu'),
    ])
    def test_sample_rff_inputchecks(self, kwarg, override, error_cls, message_fragment):
        D = 8
        kwargs = dict(
            lengthscales=numpy.random.uniform(size=(D,)),
            scaling=1.2,
            noise=0.001,
            kernel_nu=3/2,
            X=numpy.random.uniform(0, 1, size=(34, D)),
            Y=numpy.random.normal(size=(34,)),
            M=200
        )
        rff.sample_rff(**kwargs)

        kwargs[kwarg] = override
        with pytest.raises(error_cls) as exinfo:
            rff.sample_rff(**kwargs)
        assert message_fragment in exinfo.value.args[0]
        pass

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

        fun = rff.sample_rff(
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
        assert fun(X).shape == (N,)
        assert fun.grad(X).shape == (N, D)

        # check correctness of gradient by manual differentiation
        epsilon = 0.000001
        grad_diff = numpy.zeros((N, D))
        for d in range(D):
            x_plus, x_minus = X.copy(), X.copy()
            x_plus[:, d] += epsilon
            x_minus[:, d] -= epsilon
            grad_diff[:, d] = (fun(x_plus) - fun(x_minus)) / (2*epsilon)
        numpy.testing.assert_allclose(fun.grad(X), grad_diff, atol=1e-6)
        pass


class TestRffSaveLoad:
    def test_save_load_rffs(self):
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

        with pytest.raises(ValueError) as execinfo:
            rff.save_rffs([], 'testfile.h5')
        assert 'empty' in execinfo.value.args[0]

        # sample a list of RFFs
        rffs = [
            rff.sample_rff(
                lengthscales=mp['ls'],
                scaling=mp['scaling'],
                noise=mp['sigma'],
                kernel_nu=numpy.inf,
                X=X, Y=Y, M=200
            )
            for i in range(20)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            fp = pathlib.Path(tmpdir, 'testrffs.h5')
            rff.save_rffs(rffs, fp)

            with h5py.File(fp, 'r') as hfile:
                assert 'sqrt_2_alpha_over_m' in hfile
                assert 'W' in hfile
                assert 'B' in hfile
                assert 'sample_of_theta' in hfile

            rffs_loaded = rff.load_rffs(fp)
            assert len(rffs_loaded) == len(rffs)
        
        pass