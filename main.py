from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np


def main():
    train_data = loadmat('characters.mat')
    train_data = train_data['char3'][0]
    test_data = loadmat('test.mat')
    test_data = test_data['test'][0]

    # Flatten the data
    patterns = [pattern.flatten() for pattern in train_data]
    test_patterns = [pattern.flatten() for pattern in test_data]
    weights = sum([weight_matrix(pattern) for pattern in patterns])

    # test "memory" recallk
    recalled_patterns_and_energies = [updateNetwork(pattern, weights)
                                      for pattern in test_patterns]

    for i in range(len(patterns)):
        transformationAnalysis(
            i, patterns[i], recalled_patterns_and_energies[i])


def transformationAnalysis(i, pattern, recalled_pattern_and_energies):
    pattern = pattern.copy().reshape(12, 12)
    recalled_pattern, energies = recalled_pattern_and_energies
    recalled_pattern = recalled_pattern.reshape(12, 12)

    fig, ax = plt.subplots(1, 2)
    fig.suptitle(f'Pattern {i+1}')
    ax[0].imshow(pattern, cmap='gray')
    ax[0].set_title('Original Pattern')
    ax[1].imshow(recalled_pattern, cmap='gray')
    ax[1].set_title('Recalled Pattern')
    plt.show()

    plt.plot(energies)
    plt.title(f'Energy of Pattern {i+1}')
    plt.xlabel('Iterations')
    plt.ylabel('Energy')
    plt.show()


def updateNetwork(pattern, weights, max_iterations=100):
    energies = [Energy(pattern, weights)]
    new_pattern = pattern.copy()

    for _ in range(max_iterations):
        for i in np.random.permutation(len(new_pattern)):
            weighted_sum = np.dot(weights[i], new_pattern)
            new_pattern[i] = 1 if weighted_sum > 0 else 0

        current_energy = Energy(new_pattern, weights)
        energies.append(current_energy)

        if np.array_equal(new_pattern, pattern):
            break

        pattern = new_pattern.copy()

    return new_pattern, energies


def Energy(pattern, weights):
    return -0.5 * np.matmul(pattern.T, np.matmul(weights, pattern))


def weight_matrix(query):
    # Create a 2D array of the same size as the query
    w_matrix = np.zeros((len(query), len(query)))
    for i in range(len(query)):
        for j in range(len(query)):
            if i != j:
                w_matrix[i][j] = (2*query[i] - 1) * (2*query[j] - 1)

    return w_matrix


if __name__ == '__main__':
    main()
