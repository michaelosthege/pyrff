"""
Implementation of random fourier features (RFF).

Adapted from other implementations:
+ [Hernández-Lobato, 2014](https://bitbucket.org/jmh233/codepesnips2014/src/master/sourceFiles/sampleMinimum.m)
+ [Cornell-MOE](https://github.com/wujian16/Cornell-MOE/blob/master/pes/PES/sample_minimum.py)
+ [Bradford, 2018](https://github.com/Eric-Bradford/TS-EMO/blob/master/TSEMO_V3.m#L495)
"""
import functools
import hashlib
import h5py
import numba
import numpy
import os
import scipy.linalg
import scipy.stats
import typing


from . exceptions import ShapeError, DtypeError


def _compute_inverse(A:numpy.ndarray) -> numpy.ndarray:
    """Compute the inverse of a matrix.

    Uses `scipy.linalg.cho_factor`, but falls back to `scipy.linalg.inv` on errors.
    """
    try:
        A_cholesky = scipy.linalg.cho_factor(A)
        A_inverse = scipy.linalg.cho_solve(A_cholesky, numpy.eye(A.shape[0]))
        return A_inverse
    except scipy.linalg.LinAlgError:
        A_inverse = scipy.linalg.inv(A)
        return A_inverse


def _allow_1d_inputs(method):
    """
    Decorates a method requiring 2D inputs such that 1D inputs are automatically
    expanded but the dimensionality of the return value is decremented again.
    """
    @functools.wraps(method)
    def wrapper(self, x):
        x2d = numpy.atleast_2d(x)
        if numpy.ndim(x) == 1:
            return method(self, x2d)[0]
        else:
            return method(self, x2d)
    return wrapper


class RffApproximation:
    """
    A function that approximates a sample from a GP by random fourier features.

    Actual computation functions are njitted static methods to speed up evaluation and optimization.
    """
    def __init__(
        self,
        sqrt_2_alpha_over_m:float,
        W:numpy.ndarray, B:numpy.ndarray, sample_of_theta:numpy.ndarray
    ):
        """Creates an RFF function object from function features.

        Parameters
        ----------
        sqrt_2_alpha_over_m : float
        W : numpy.ndarray
            (M, D) array
        B : numpy.ndarray
            (M, 1) array
        sample_of_theta : numpy.ndarray
            (M,) vector
        """
        self.sqrt_2_alpha_over_m = sqrt_2_alpha_over_m
        self.W = W
        self.B = B
        self.sample_of_theta = sample_of_theta
        super().__init__()

    @property
    def M(self) -> int:
        """Number of fourier features."""
        return self.W.shape[0]

    @property
    def D(self) -> int:
        """Number of input dimensions."""
        return self.W.shape[1]

    @property
    def uuid(self) -> str:
        """ A unique hash of the RFF function, created from its parameter arrays. """
        components = [
            hashlib.md5(self.sqrt_2_alpha_over_m).hexdigest(),
            hashlib.md5(self.W).hexdigest(),
            hashlib.md5(self.B).hexdigest(),
            hashlib.md5(self.sample_of_theta).hexdigest(),
        ]
        return hashlib.md5(','.join(components).encode('ascii')).hexdigest().encode('ascii')

    @_allow_1d_inputs
    def __call__(self, x:numpy.ndarray) -> typing.Union[numpy.ndarray, float]:
        """Evalues an RFF approximation function, specified by the function features.

        Parameters
        ----------
        x : numpy.ndarray
            a 2D array of coordinates (?, D)

        Returns
        -------
        approx_y : numpy.ndarray
            function evaluations (?,)
        """
        return RffApproximation._evaluate(x, self.sqrt_2_alpha_over_m, self.W, self.B, self.sample_of_theta)

    @_allow_1d_inputs
    def grad(self, x:numpy.ndarray) -> numpy.ndarray:
        """Evaluates the gradient of an RFF approximation function, specified by the function features.

        Parameters
        ----------
        x : numpy.ndarray
            a 2D array of coordinates (?, D)

        Returns
        -------
        dydx : numpy.ndarray
            evaluations of the gradient w.r.t. x (?, D)
        """
        return RffApproximation._grad(x, self.sqrt_2_alpha_over_m, self.W, self.B, self.sample_of_theta)

    @staticmethod
    @numba.njit
    def _evaluate(
        x:numpy.ndarray,
        sqrt_2_alpha_over_m:float, W:numpy.ndarray, B:numpy.ndarray, sample_of_theta:numpy.ndarray
    ) -> numpy.ndarray:
        M, D = W.shape
        N = x.shape[0]
        assert x.shape == (N, D)
        phi_x = sqrt_2_alpha_over_m * numpy.cos(numpy.dot(W, x.T) + B)
        assert phi_x.shape == (M, N)
        approx_y = numpy.dot(phi_x.T, sample_of_theta)
        assert approx_y.shape == (N,)
        return approx_y

    @staticmethod
    @numba.njit
    def _grad(x:numpy.ndarray, sqrt_2_alpha_over_m:float, W:numpy.ndarray, B:numpy.ndarray, sample_of_theta:numpy.ndarray) -> numpy.ndarray:
        M, D = W.shape
        N = x.shape[0]
        assert x.shape == (N, D)
        temp = numpy.sin(numpy.dot(W, x.T) + B)
        assert temp.shape == (M, N)
        # the following is a refactor of the Cornell-MOE implementation that supports numba.njit (about 6x faster)
        gradient = numpy.empty((N, D))
        for n in range(N):
            gradient_of_phi_x = -sqrt_2_alpha_over_m * temp[:,n] * W.T
            gradient[n] = numpy.dot(gradient_of_phi_x, sample_of_theta)
        return gradient


def sample_rff(
    lengthscales:numpy.ndarray,
    scaling:float,
    noise:float,
    kernel_nu:float,
    X:numpy.ndarray,
    Y:numpy.ndarray,
    M:int,
) -> typing.Tuple[
    typing.Callable[[numpy.ndarray], typing.Union[numpy.ndarray, float]],
    typing.Callable[[numpy.ndarray], typing.Union[numpy.ndarray, float]],
    typing.Callable[[numpy.ndarray], numpy.ndarray],
]:
    """
    Samples an approximate function from a GP with given kernel function and zero mean.

    It returns not only the approximate function, but also a njit-decorated version and its njit-decorated gradient.
    The njit-decorated callables have a compilation overhead on the first call!
    After that they are about 3.5x and 6x faster than non-njitted alternatives.

    Parameters
    ----------
    lengthscales : numpy.ndarray
        lengthscale values for each dimension (D,). It corresponds to...
            '1/l²' from [Hernández-Lobato, 2014] implementation
            'l' from Cornell-MOE implementation
            'ell' from [Bradford, 2018] implementation, line 501
    scaling : float
        the kernel standard deviation sigma. (As in k(x,x') = scaling**2 * ExpQuad()) It corresponds to...
            'sqrt(sigma)' from [Hernández-Lobato, 2014] implementation
            'sqrt(alpha)' from Cornell-MOE implementation
            'sqrt(sf2)' from [Bradford, 2018] implementation, line 502
    noise : float
        the observation noise from the likelihood function. It corresponds to...
            'sigma0' from [Hernández-Lobato, 2014] implementation
            'noise' from Cornell-MOE implementation
            'sn2' from [Bradford, 2018] implementation, line 503
    kernel_nu : float
        degree of freedom of the GP kernel, for example
            `numpy.inf` for squared exponential / RBF / exponential quadratic
            1/2 for Matern12
            3/2 for Matern32
    X : numpy.ndarray
        coordinates of observations (?, D)
    Y : numpy.ndarray
        observed values at coordinates (?,)
    M : int
        number of fourier features to use in the approximation
        according to [Mutny & Krause, 2018], the approximation error scales O(m^(1/2))

    Returns
    -------
    f_rff : callable
        may be called with (?, D) or (D,) coordinates, returning (?,) or float respectively
    f_rff_njit : callable
        jit-compiled version of the above
    g_rff_njit : callable
        jit-compiled gradient, takes (?, D) or (D,) coordinates, returning (?, D) or (?, D) respectively
    """
    # check inputs
    X = numpy.atleast_2d(X)
    Y = numpy.atleast_1d(Y)
    D = X.shape[1]
    N_obs = len(Y)
    lengthscales = numpy.atleast_1d(lengthscales)
    if not X.shape == (N_obs, D):
        raise ShapeError('Shapes of X and Y do not match.', expected='(?, D), (?,)', actual=f'{X.shape}, {Y.shape}')
    if not lengthscales.shape == (D,):
        raise ShapeError('Lengthscales and data dimensions do not match.', lengthscales.shape, (D,))
    if not numpy.ndim(scaling) == 0:
        raise ShapeError('Argument "scaling" must be a scalar.')
    if not numpy.ndim(noise) == 0:
        raise ShapeError('Argument "noise" must be a scalar.')
    if not isinstance(kernel_nu, (int, float)) or kernel_nu <= 0:
        raise ValueError('Argument "kernel_nu" must be a positive-definite scalar.')

    # see Bradford 2018, equation 26
    alpha = scaling**2
    # just to avoid re-computing it all the time:
    sqrt_2_alpha_over_m = numpy.sqrt(2*alpha/M)

    # construct function to sample p(w) (see [Bradford, 2018], equations 27 and 28)
    if numpy.isinf(kernel_nu):
        def p_w(size:tuple) -> numpy.ndarray:
            return numpy.random.normal(loc=0, scale=1 / lengthscales, size=size)
    else:
        def p_w(size:tuple) -> numpy.ndarray:
            return scipy.stats.t.rvs(loc=0, scale=1 / lengthscales, df=kernel_nu, size=size)

    # sample function features
    W = p_w(size=(M, D))
    assert W.shape == (M, D)
    B = numpy.random.uniform(0, 2*numpy.pi, size=(M, 1))
    assert B.shape == (M, 1)

    zeta = sqrt_2_alpha_over_m * numpy.cos(numpy.dot(W, X.T) + B)
    assert zeta.shape == (M, N_obs)

    A = numpy.divide(numpy.dot(zeta, zeta.T), noise) + numpy.eye(M)
    A_inverse = _compute_inverse(A)

    assert A_inverse.shape == (M, M)
    mean_of_post_theta = numpy.divide(numpy.dot(numpy.dot(A_inverse, zeta), Y), noise)
    assert mean_of_post_theta.shape == (M,)

    variance_of_post_theta = A_inverse
    assert variance_of_post_theta.shape == (M, M)

    sample_of_theta = numpy.random.multivariate_normal(mean_of_post_theta, variance_of_post_theta)
    assert sample_of_theta.shape == (M,)

    # create a function object
    rff = RffApproximation(sqrt_2_alpha_over_m, W, B, sample_of_theta)
    # call the function with test data to make sure that it works
    X_test = numpy.array([X[0], X[0]])
    assert numpy.shape(rff(X_test)) == (2,)
    assert numpy.shape(rff.grad(X_test)) == (2, D)
    return rff


def save_rffs(rffs:typing.Sequence[RffApproximation], fp:os.PathLike):
    """
    Saves a collection of RffApproximations to an HDF5 file.

    Parameters
    ----------
    rffs : list
        a collection of RffApproximation objects, each with the same D and M
    fp : os.PathLike
        filepath for the HDF5 file to create

    Returns
    -------
    fp : os.PathLike
        the filepath passed as the input, now pointing to an HDF5 file with datasets:
        (N, 1) sqrt_2_alpha_over_m
        (N, M, D) W
        (N, M, 1) B
        (N, M) sample_of_theta
        (N,) uuid
    """
    # check the inputs
    N = len(rffs)
    if N == 0:
        raise ValueError('The [rffs] collection is empty.')
    M = rffs[0].M
    D = rffs[0].D
    for rff in rffs:
        if not rff.M == M and rff.D == D:
            raise ValueError('All elements in the [rff] collection must have the same D and M.')
    with h5py.File(fp, 'w') as hfile:
        for name, shape, dtype in [
            ('sqrt_2_alpha_over_m', (N, 1), float),
            ('W', (N, M, D), float),
            ('B', (N, M, 1), float),
            ('sample_of_theta', (N, M), float),
            ('uuid', (N,), 'S32'),
        ]:
            hfile.create_dataset(name, shape, dtype=dtype, data=[
                getattr(rff, name)
                for rff in rffs
            ])
    return fp


def load_rffs(fp:os.PathLike) -> typing.Sequence[RffApproximation]:
    """
    Loads a collection of RffApproximations from an HDF5 file.

    Parameters
    ----------
    fp : os.PathLike
        filepath to an HDF5 file containing datasets for:
        (N, 1) sqrt_2_alpha_over_m
        (N, M, D) W
        (N, M, 1) B
        (N, M) sample_of_theta
        (N,) uuid

    Returns
    -------
    rffs : list
        loaded RffApproximation objects
    """
    rffs = None
    with h5py.File(fp, 'r') as hfile:
        N, M, D = hfile['W'].shape
        rffs = [None] * N
        for n in range(N):
            rffs[n] = RffApproximation(
                sqrt_2_alpha_over_m=hfile['sqrt_2_alpha_over_m'][n],
                W=hfile['W'][n],
                B=hfile['B'][n],
                sample_of_theta=hfile['sample_of_theta'][n],
            )
            # validate by comparing function hashes
            expected = hfile['uuid'][n]
            actual = rffs[n].uuid
            if actual != expected:
                raise Exception(f'The uuid of the function at position {n} did not match. ({actual} != {expected})')
    return rffs
