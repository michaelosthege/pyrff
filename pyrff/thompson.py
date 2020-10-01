"""
Convenience implementation of the standard Thompson Sampling algorithm.
"""
import numpy
import typing

# custom type shortcuts
Sample = typing.Union[int, float]


def sample_batch(
    candidate_samples: typing.Sequence[typing.Sequence[Sample]], *,
    ids:typing.Sequence,
    batch_size:int, seed:typing.Optional[int]=None
) -> tuple:
    """
    Draws a batch of candidates by Thompson Sampling from posterior samples.

    Parameters
    ----------
    candidate_samples : array-like
        posterior samples for each candidate (C,)
        (sample count may be different)
    ids : numpy.ndarray, list, tuple
        (C,) candidate identifiers
    batch_size : int
        number of candidates to draw (B)

    Returns
    -------
    chosen_candidates : tuple
        (B,) chosen candidate ids for the batch
    """
    n_candidates = len(candidate_samples)
    n_samples = tuple(map(len, candidate_samples))
    if len(ids) != n_candidates:
        raise ValueError(f"Number of candidate ids ({len(ids)}) does not match number of candidate_samples ({n_candidates}).")
    ids = numpy.atleast_1d(ids)
    # work with matrix even if sample count is varies to get more efficient slicing
    samples = numpy.zeros((max(n_samples), n_candidates))
    samples[:] = numpy.nan
    for c, (samps, s) in enumerate(zip(candidate_samples, n_samples)):
        samples[:s, c] = samps
    chosen_candidates = []
    random = numpy.random.RandomState(seed)
    for i in range(batch_size):
        # for every sample in the batch, randomize the column order
        # to prevent always selecting lower-numbered candidates when >=2 samples are equal
        col_order = random.permutation(n_candidates)
        idx = random.randint(n_samples, size=n_candidates)
        selected_samples = samples[:, col_order][idx, numpy.arange(n_candidates)]
        best_candidate = ids[col_order][numpy.argmax(selected_samples)]
        chosen_candidates.append(best_candidate)
    random.seed(None)
    return tuple(chosen_candidates)


def get_probabilities(samples:numpy.ndarray, nit:int=100_000):
    """Get thompson sampling probabilities from posterior.

    Parameters
    ----------
        samples : numpy.ndarray
            (S, C) array of posterior samples (S) for each candidate (C)
        nit : int
            how many thompson draws samples to draw for the estimation

    Returns
    -------
        probabilities : numpy.ndarray
            (C,) probabilities that the candidates are sampled
    """
    n_samples, n_candidates = samples.shape

    frequencies = numpy.zeros(n_candidates)
    for _ in range(nit):
        idx = numpy.random.randint(n_samples, size=n_candidates)
        selected_samples = samples[idx, numpy.arange(n_candidates)]
        frequencies[numpy.argmax(selected_samples)] += 1
    return frequencies / frequencies.sum()
