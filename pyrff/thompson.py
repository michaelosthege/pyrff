"""
Convenience implementation of the standard Thompson Sampling algorithm.
"""
import itertools
import typing
import fastprogress
import numpy

from . import exceptions

# custom type shortcuts
Sample = typing.Union[int, float]


def sample_batch(
    candidate_samples: typing.Sequence[typing.Sequence[Sample]], *,
    ids:typing.Sequence,
    correlated:bool,
    batch_size:int,
    seed:typing.Optional[int] = None,
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
    correlated : bool
        Switches between jointly (correlated=True) or independently (correlated=False) sampling the candidates.
        When correlated=True, all candidates must have the same number of samples.
    batch_size : int
        number of candidates to draw (B)
    seed : int
        seed for the random number generator (will reset afterwards)

    Returns
    -------
    chosen_candidates : tuple
        (B,) chosen candidate ids for the batch
    """
    n_candidates = len(candidate_samples)
    n_samples = tuple(map(len, candidate_samples))
    if correlated and len(set(n_samples)) != 1:
        raise exceptions.ShapeError("For correlated sampling, all candidates must have the same number of samples.")
    if len(ids) != n_candidates:
        raise exceptions.ShapeError(f"Number of candidate ids ({len(ids)}) does not match number of candidate_samples ({n_candidates}).")
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
        if correlated:
            idx = numpy.repeat(numpy.random.randint(n_samples[0]), n_candidates)
        else:
            idx = random.randint(n_samples, size=n_candidates)
        selected_samples = samples[:, col_order][idx, numpy.arange(n_candidates)]
        best_candidate = ids[col_order][numpy.argmax(selected_samples)]
        chosen_candidates.append(best_candidate)
    random.seed(None)
    return tuple(chosen_candidates)


def _sort_samples(
    candidate_samples: typing.Sequence[typing.Sequence[Sample]],
) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
    """ Flattens samples into a sorted array and corresponding group numbers.

    Parameters
    ----------
    candidate_samples : array-like
        posterior samples for each candidate (C,)
        (sample count may be different)

    Returns
    -------
    samples : array
        sorted array of all samples
    sample_candidates : array
        the group numbers
    """
    flat_samples = numpy.vstack([
        numpy.stack([samps, numpy.repeat(g, len(samps))]).T
        for g, samps in enumerate(candidate_samples)
    ])
    flat_samples = flat_samples[numpy.argsort(flat_samples[:, 0]), :]
    return flat_samples[:, 0], flat_samples[:, 1].astype(int)


def _win_draw_prob(cprobs: numpy.ndarray) -> float:
    """ Calculate the probability of winning by draw.

    This function iterates over all possible combinations of draws.
    The runtime complexity explodes exponentially with O(2^N - N - 1).

    Parameters
    ----------
    cprobs : numpy.ndarray
        (3, N) array of probabilities that a win/loose/draw occurs
        between the candidate value and other candidates (candidate itself is not included in [cprobs])

    Returns
    -------
    p_win_draw : float
        probability of winning in a fair draw against the other candidates
    """
    C = cprobs.shape[1]
    columns = numpy.arange(C)

    drawable = tuple(columns[cprobs[2, :] > 0])
    p_win_draw = 0
    for n in range(1, len(drawable) + 1):
        p_win = 1 / (n + 1)
        for combination in itertools.combinations(drawable, r=n):
            draw_probs = cprobs[2, list(combination)]
            win_probs = cprobs[0, list(sorted(set(columns).difference(combination)))]
            p_event = numpy.prod(draw_probs) * numpy.prod(win_probs)
            p_win_draw += p_win * p_event
            combo = ["W"]*C
            for c in combination:
                combo[c] = "D"
    return p_win_draw


def _rolling_probs_calculation(
    samples: numpy.ndarray, sample_candidates: numpy.ndarray,
    s_totals: numpy.ndarray,
) -> typing.Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """ Calculates win, loose and win-by-draw probabilities for all samples.

    Parameters
    ----------
    samples : numpy.ndarray
        (S,) values of candidate samples
    sample_candidates : numpy.ndarray
        (S,) corresponding candidate indices
    s_totals : numpy.ndarray
        (C,) numbers of samples per candidate

    Returns
    -------
    df_probs : pandas.DataFrame
        table with win/loose/win-by-draw probabilities for each sample
    """
    C = s_totals.shape[0]

    p_win = numpy.zeros_like(samples, dtype=float)
    p_loose = numpy.zeros_like(samples, dtype=float)
    p_win_draw = numpy.zeros_like(samples, dtype=float)

    # s_smaller: number of samples IN EACH COLUMN that are smaller than [value]
    s_smaller = numpy.zeros((C,))

    # now iterate over groups with identical sample values in the DataFrame
    # pandas.DataFrame.groupby is too slow for this -> DIY iterator using the unique idx & counts
    unique_values, idx_from, counts = numpy.unique(samples, return_counts=True, return_index=True)
    for value, ifrom, nsame in fastprogress.progress_bar(tuple(zip(unique_values, idx_from, counts))):
        ito = ifrom + nsame
        candidates_with_value = sample_candidates[ifrom:ito]

        # s_same: number of samples IN EACH COLUMN that have the same [value]
        s_same = numpy.zeros(C)
        same_cols, same_counts = numpy.unique(candidates_with_value, return_counts=True)
        s_same[same_cols] = same_counts
        # s_larger: number of samples IN EACH COLUMN that are larger than [value]
        s_larger = s_totals - s_smaller - s_same

        # from the counts of smaller/same/larger values in other columns,
        # calculate the probabilities of direct win, direct loss and draw
        cprobs_all = numpy.array([
            s_smaller,
            s_larger,
            s_same
        ]) / s_totals
        for s, fc in zip(range(ifrom, ito), candidates_with_value):
            # do not look at probabilities w.r.t. the same column:
            cprobs = numpy.hstack([cprobs_all[:, :fc], cprobs_all[:, fc+1:]])

            p_win[s] = numpy.prod(cprobs[0, :])
            p_loose[s] = 1 - numpy.prod(1 - cprobs[1, :])

            if s_same[fc] != nsame:
                # draws with other columns are possible -> calculate combinatorial event & win probabilities
                p_win_draw[s] = _win_draw_prob(cprobs)
            # else:
            #    # no other candidate has a sample of this value
            #    p_win_draw[s] was initialized to 0

        # increment the column-wise count of samples that are smaller than [value]
        # by this iterative counting, we avoid doing S*C </=/> comparisons, dramatically reducing complexity
        s_smaller += s_same

    return p_win, p_loose, p_win_draw


def sampling_probabilities(
    candidate_samples: typing.Sequence[typing.Sequence[Sample]],
    correlated:bool,
) -> numpy.ndarray:
    """ Calculates the thompson sampling probability of each candidate.

    ATTENTION: When correlated=False is specified, the occurence of non-unique sample values can
    increase the runtime complexity to worst-case O(2^total_samples).

    Parameters
    ----------
    candidate_samples : array-like
        posterior samples for each candidate (C,)
        (sample count may be different)
    correlated : bool
        Switches between jointly (correlated=True) or independently (correlated=False) sampling the candidates.
        When correlated=True, all candidates must have the same number of samples.

    Returns
    -------
    probabilities : numpy.ndarray
        (C,) array of sampling probabilities
    """
    C = len(candidate_samples)
    s_totals = numpy.array(tuple(map(len, candidate_samples)))
    if correlated and len(set(s_totals)) != 1:
        raise exceptions.ShapeError("For correlated sampling, all candidates must have the same number of samples.")

    probabilities = numpy.zeros(C, dtype=float)
    if correlated:
        # this case is O(S^2 * C) because it does not need to account for combinations
        S = s_totals[0]
        for s, samples in enumerate(numpy.array(candidate_samples).T):
            vwin = numpy.max(samples)
            # which candidates have the highest value?
            i_winners = numpy.argwhere(samples == vwin)
            n_winners = len(i_winners)
            # attribute winning probability to the winners
            probabilities[i_winners] += 1 / n_winners
        probabilities /= S
    else:
        # For uncorrelated TS, all possible combinations must be considered.
        # Naively doing all combinations would be O(S^C), but this implementation simplifies it:
        # 1. it's sufficient to categorize win/loose/draw
        # 2. sorting the samples allows for an iteration that needs much fewer </=/> comparisons
        # 3. combinatorics for draw win probabilities is only required for non-unique sample values

        # first sort all samples into a vector
        samples, sample_candidates = _sort_samples(candidate_samples)
        # then calculate the win/loose/win-by-draw probabilities for each sample
        p_win, p_loose, p_win_draw = _rolling_probs_calculation(samples, sample_candidates, s_totals)
        # finally summarize the sample-wise probabilities by the corresponding candidate
        for c in range(C):
            mask = sample_candidates == c
            probabilities[c] = numpy.sum(p_win[mask] + p_win_draw[mask]) / numpy.sum(mask)
    return probabilities
