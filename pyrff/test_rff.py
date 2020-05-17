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
        numpy.random.seed(1234)
        rff.sample_rff(**kwargs)

        kwargs[kwarg] = override
        with pytest.raises(error_cls) as exinfo:
            rff.sample_rff(**kwargs)
        assert message_fragment in exinfo.value.args[0]
        pass

    @pytest.mark.parametrize('M', [100, 200, 300])
    def test_rff_1d(self, M):
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

        numpy.random.seed(1234)
        fun = rff.sample_rff(
            lengthscales=mp['ls'],
            scaling=mp['scaling'],
            noise=mp['sigma'],
            kernel_nu=numpy.inf,
            X=X, Y=Y, M=M
        )
        assert fun.D == 1
        assert fun.M == M

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

    @pytest.mark.parametrize('M', [100, 200, 300])
    def test_rff_2d(self, M):
        X = numpy.array([
            [2.91766097, 2.24035167],
            [0.05847315, -1.36898571],
            [-0.97848763, -1.69827441],
            [-1.34113714, -0.94010644],
            [2.17295361, -2.05980198],
            [-2.15467653, 1.54248168],
            [1.41794951, -0.86602145],
            [-0.9534419, 1.0008183],
            [-1.69739617, 0.36856189],
            [-2.25492733, -1.08158109],
            [2.71928324, -2.17585926],
            [0.41647858, 2.8539929],
            [0.02020237, 1.00598524],
            [-2.79485101, -0.2632838],
            [-2.06489182, -0.1437062],
            [-1.98178537, 2.37755002],
            [-0.75963745, -0.72184242],
            [2.14989953, 0.87636633],
            [0.50077019, 1.01010016],
            [-1.93324428, 2.0954881],
            [-0.34576451, 1.98880626],
            [1.5835243, 2.51814586],
            [-2.57655944, -2.0630075],
            [0.82136539, 0.33417412],
            [-1.84842772, -0.4460641],
            [0.08052125, -1.38373449],
            [0.59412011, -1.67895633],
            [-1.19482503, -2.7096331],
            [0.38592035, 2.61619323],
            [1.81816585, 1.1838309]
        ])
        Y = numpy.array([
            0.2684595, -3.12345661, -1.20876944, -2.0739321, -0.02077514,
            0.50824162, -2.01574938, 1.43596611, -0.81839906, -0.17958556,
            -0.02243076, -0.30501982, 3.87966439, -0.25128758, -0.99743626,
            -0.18646517, -4.7381341, 7.1723458, 5.83390888, -0.01031394,
            -0.10963341, -0.07243528, -0.10766074, -0.9695199, -1.10566236,
            -3.04485123, -1.64010435, 0.01366085, -0.23411911, 7.61113715
        ])
        mp = {
            'ls': numpy.array([3.86353836, 1.24398732]),
            'scaling': numpy.array(8.82195995),
            'sigma': numpy.array(0.9789097)
        }

        numpy.random.seed(1234)
        fun = rff.sample_rff(
            lengthscales=mp['ls'],
            scaling=mp['scaling'],
            noise=mp['sigma'],
            kernel_nu=numpy.inf,
            X=X, Y=Y, M=M
        )
        assert fun.D == 2
        assert fun.M == M

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
        numpy.random.seed(1234)
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
                assert 'uuid' in hfile

            rffs_loaded = rff.load_rffs(fp)
            assert len(rffs_loaded) == len(rffs)
            for r in range(len(rffs)):
                assert rffs_loaded[r].uuid == rffs[r].uuid

        pass
