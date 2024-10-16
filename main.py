from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import random


def main():
    # indexes = random.sample(range(18), 7)
    train_data = loadmat("characters.mat")
    train_data = train_data["char3"][0]
    test_data = loadmat("test.mat")
    test_data = test_data["test"][0]

    # Flatten the data
    patterns = [pattern.reshape(-1, 1).astype(int) for pattern in train_data]
    test_patterns = [pattern.reshape(-1, 1).astype(int)
                     for pattern in test_data]

    for i in range(len(patterns)):
        plt.imshow(patterns[i].reshape(12, 12), cmap="gray")
        plt.savefig(f"input/pattern_{i+1}.png")
        plt.close()

    weights = sum([weightMatrix(pattern) for pattern in patterns])

    # test "memory" recall
    recalled_patterns_and_energies = [updateNetwork(
        pattern, weights) for pattern in patterns]

    for i in range(len(patterns)):
        transformationAnalysis(
            i, patterns[i], recalled_patterns_and_energies[i], test=False, early_stop=False)

    recalled_patterns_and_energies = [
        updateNetwork(pattern, weights) for pattern in test_patterns
    ]

    for i in range(len(patterns)):
        transformationAnalysis(
            i, test_patterns[i], recalled_patterns_and_energies[i], test=True, early_stop=False)


def transformationAnalysis(i, pattern, recalled_pattern_and_energies, test=False, early_stop=False):
    pattern = pattern.copy().reshape(12, 12)
    recalled_pattern, energies = recalled_pattern_and_energies
    recalled_pattern = recalled_pattern.reshape(12, 12)

    fig, ax = plt.subplots(1, 3)
    fig.suptitle(f"Pattern {i+1}")
    ax[0].imshow(pattern, cmap="gray")
    ax[1].imshow(recalled_pattern, cmap="gray")
    ax[1].set_title("Recalled Pattern")
    ax[2].plot(energies)
    ax[2].set_title("Energy")
    ax[2].set_xlabel("Iterations")
    ax[2].set_ylabel("Energy")
    ax[2].yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{int(x/1000)}k" if abs(x) > 1000 else x)
    )
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=None
    )

    filename = f"test_{i+1}" if test else f"pattern_{i+1}"
    filename += "_early_stop" if early_stop else ""
    plt.savefig(f"results/{filename}.png")
    plt.close()

    # TODO calculate error between pattern and recalled pattern, display calculated error
    print(
        f"Error between pattern {i+1} and recalled pattern: {np.sum(np.abs(pattern - recalled_pattern))}")


def updateNetwork(pattern, weights, max_iterations=100):

    pattern = (pattern * 2) - 1
    energies = [Energy(pattern, weights)]

    for _ in range(max_iterations):
        # for i in random.sample(range(len(pattern)), len(pattern)):
        i = random.randint(0, len(pattern) - 1)
        weighted_sum = np.dot(weights[i, :], pattern)
        pattern[i] = 1 if weighted_sum > 0 else -1

        current_energy = Energy(pattern, weights)
        energies.append(current_energy)

        # if energies[-1] == energies[-2]:
        #     break


    return (pattern + 1) // 2, energies


def Energy(pattern, weights):
    energy = 0
    for i in range(len(pattern)):
        for j in range(len(pattern)):
            energy += weights[i][j] * pattern[i] * pattern[j]

    return energy / (-2)


def weightMatrix(query):
    # Create a 2D array of the same size as the query
    query = query * 2 - 1
    w_matrix = np.outer(query, query)
    np.fill_diagonal(w_matrix, 0)
    return w_matrix


if __name__ == "__main__":
    main()
