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
        )
        numpy.testing.assert_array_equal(samples.shape, (S, C))

        batch = thompson.sample_batch(
            samples=samples, ids=ids,
            batch_size=batch_size, seed=seed
        )
        assert len(batch) == batch_size
        # all samples for C02 are less then all others
        assert 'C02' not in batch
        assert batch.count('C00') < batch.count('C01')
        pass

    def test_no_bias_on_sample_collisions(self):
        samples = numpy.array([
            [2, 2, 2],
            [2, 2, 2],
        ])
        batch = thompson.sample_batch(samples, ids=('A', 'B', 'C'), batch_size=100, seed=1234)
        assert batch.count('A') != 100
        assert batch.count('C') != 0
        pass

    @pytest.mark.xfail(reason='Probabilities are currently computed by brute force and non-exact.')
    def test_get_probabilities_exact_on_identical(self):
        samples = numpy.array([
            [1, 2, 3, 4, 5],
            [5, 3, 4, 2, 1],
            [1, 3, 4, 2, 5]
        ]).T
        S, C = samples.shape
        assert S == 5
        assert C == 3

        probabilities = thompson.get_probabilities(samples)
        numpy.testing.assert_array_equal(probabilities, [1/C]*C)
        pass
