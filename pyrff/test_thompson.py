import numpy
import pytest

from . import thompson


class TestThompsonSampling:
    @pytest.mark.parametrize('batch_size', [1_000, 5_000])
    @pytest.mark.parametrize('seed', [None, 123, 349857])
    def test_sample_batch(self, batch_size, seed):
        S = 500
        C = 3
        ids = [f'C{c:02d}' for c in range(C)]
        samples = numpy.random.uniform(
            low=[0, 0, -1],
            high=[0.2, 1, 0],
            size=(S, C)
        ).T
        numpy.testing.assert_array_equal(samples.shape, (C, S))

        batch = thompson.sample_batch(
            candidate_samples=samples, ids=ids,
            batch_size=batch_size, seed=seed
        )
        assert len(batch) == batch_size
        # all samples for C02 are less then all others
        assert 'C02' not in batch
        assert batch.count('C00') < batch.count('C01')
        pass

    def test_no_bias_on_sample_collisions(self):
        samples = [
            [2, 2, 2],
            [2, 2],
            [2, 2, 2],
        ]
        batch = thompson.sample_batch(samples, ids=('A', 'B', 'C'), batch_size=100, seed=1234)
        assert batch.count('A') != 100
        assert batch.count('C') != 0
        pass


class TestThompsonProbabilities:
    def test_sort_samples(self):
        samples, sample_cols = thompson._sort_samples([
            [3,1,2],
            [4,-1],
            [7],
        ])
        numpy.testing.assert_array_equal(samples, [-1, 1, 2, 3, 4, 7])
        numpy.testing.assert_array_equal(sample_cols, [1, 0, 0, 0, 1, 2])
        pass

    def test_win_draw_prob(self):
        assert thompson._win_draw_prob(numpy.array([
            [1, 0, 0],
            [0, 1, 1],
            [0, 0, 0],
        ])) == 0.0

        assert thompson._win_draw_prob(numpy.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
        ])) == 0.25

        numpy.testing.assert_allclose(thompson._win_draw_prob(numpy.array([
            [0, 0],
            [0.5, 0.75],
            [0.5, 0.25],
        ])), 0.041666666)
        pass

    def test_sampling_probability(self):
        numpy.testing.assert_array_equal(thompson.sampling_probabilities([
            [0, 1, 2],
            [0, 1, 2],
        ]), [0.5, 0.5])
        
        numpy.testing.assert_array_equal(thompson.sampling_probabilities([
            [0, 1, 2],
            [10],
        ]), [0, 1])

        numpy.testing.assert_array_equal(thompson.sampling_probabilities([
            [0, 1, 2],
            [3, 4, 5],
            [5, 4, 3],
        ]), [0, 0.5, 0.5])

        numpy.testing.assert_array_equal(thompson.sampling_probabilities([
            [5, 6],
            [0, 0, 10, 20],
            [5, 6],
        ]), [0.25, 0.5, 0.25])
        pass