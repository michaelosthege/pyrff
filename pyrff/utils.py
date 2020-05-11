import numpy
import scipy.optimize
import typing


def multi_start_fmin(fun:typing.Callable[[numpy.ndarray], float], x0s:numpy.ndarray) -> typing.Tuple[numpy.ndarray, float]:
    """
    Helper function to run fmin-optimization from many start points.

    Parameters
    ----------
    fun : callable
        the function to minimize
    x0s : numpy.ndarray
        (N_starts, D) array of initial guesses

    Returns
    -------
    x_best : numpy.ndarray
        (D,) coordinate of the found minimum
    y_best : float
        function value at the minimum
    """
    x_peaks = [
        scipy.optimize.fmin(fun, x0=x0, disp=False)
        for x0 in x0s
    ]
    y_peaks = [
        fun(x)
        for x in x_peaks
    ]
    ibest = numpy.argmin(y_peaks)
    return x_peaks[ibest], y_peaks[ibest]
