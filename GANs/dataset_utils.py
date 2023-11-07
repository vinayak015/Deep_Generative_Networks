import numpy as np
import matplotlib.pyplot as plt


def gaussian_data_2mode(n=30000):
    assert n % 2 == 0
    gaussian1 = np.random.normal(loc=-1, scale=0.25, size=(n // 2,))
    gaussian2 = np.random.normal(loc=0.5, scale=0.5, size=(n // 2,))
    data = (np.concatenate([gaussian1, gaussian2]) + 1).reshape([-1, 1])
    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    return 2 * scaled_data - 1


def visualize_dataset():
    data = gaussian_data_2mode()
    plt.hist(data, bins=50, alpha=0.7, label='train data')
    plt.legend()
    plt.savefig("2-mode_gaussian.png")
    plt.show()

# visualize_dataset()
