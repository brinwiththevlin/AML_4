import unittest
import numpy as np
from main import updateNetwork, Energy, weightMatrix


class TestMain(unittest.TestCase):
    pattern_1 = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    pattern_2 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    pattern_3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    pattern_4 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    data = np.load("data.npz")

    expected_weights = [data["pattern_1"], data["pattern_2"], data["pattern_3"], data["pattern_4"]]
    hopfield = sum(expected_weights)

    def test_updateNetwork(self):
        patterns = [self.pattern_1, self.pattern_2, self.pattern_3, self.pattern_4]
        for i in range(4):
            recalled_pattern, energies = updateNetwork(patterns[i], self.expected_weights[i])
            # self.assertTrue(sum(recalled_pattern == patterns[i]) >= 24)
            self.assertTrue(np.array_equal(recalled_pattern, patterns[i]))
            self.assertEqual(energies[-1], -325)
            recalled_pattern, energies = updateNetwork(patterns[i], self.hopfield)
            # self.assertTrue(sum(recalled_pattern == patterns[i]) >= 24)
            self.assertTrue(np.array_equal(recalled_pattern, patterns[i]))
            self.assertEqual(energies[-1], -624)

            test_pattern = noisyPattern(patterns[i], 0.1)
            recalled_pattern, energies = updateNetwork(test_pattern, self.expected_weights[i])
            self.assertFalse(np.array_equal(recalled_pattern, test_pattern))
            # self.assertTrue(sum(recalled_pattern == patterns[i]) >= 24)
            self.assertTrue(np.array_equal(recalled_pattern, patterns[i]))
            recalled_pattern, energies = updateNetwork(test_pattern, self.hopfield)
            self.assertFalse(np.array_equal(recalled_pattern, test_pattern))
            # self.assertTrue(sum(recalled_pattern == patterns[i]) >= 24)
            self.assertTrue(np.array_equal(recalled_pattern, patterns[i]))



    def test_Energy(self):
        patterns = [self.pattern_1, self.pattern_2, self.pattern_3, self.pattern_4]

        for i in range(4):
            pattern = patterns[i] * 2 - 1
            energy = Energy(pattern, self.expected_weights[i])
            self.assertEqual(energy, -325)
            energy = Energy(pattern, self.hopfield)
            self.assertEqual(energy, -624)


    def test_weight_matrix(self):
        # Test with pattern_1, pattern_2, pattern_3, pattern_4
        patterns = [self.pattern_1, self.pattern_2,
                    self.pattern_3, self.pattern_4]
        weights = [weightMatrix(pattern) for pattern in patterns]

        for i in range(len(weights)):
            assert np.array_equal(weights[i], self.expected_weights[i])
            assert self.expected_weights[i].shape == weights[i].shape
            assert self.expected_weights[i].diagonal().all() == 0


def noisyPattern(pattern, noise_level):
    noisy_pattern = np.copy(pattern)
    num_flips = int(noise_level * len(pattern))
    flip_indices = np.random.choice(len(pattern), num_flips, replace=False)
    noisy_pattern[flip_indices] = 1 - noisy_pattern[flip_indices]
    return noisy_pattern

if __name__ == "__main__":
    unittest.main()
