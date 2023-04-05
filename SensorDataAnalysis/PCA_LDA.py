"""Author: Nicola Vidovic

PCA and LDA, I build in ML1.

"""

import numpy as np


def standardize(data):
    centered_data = data - np.mean(data, axis=0)
    std_deviation = np.sqrt(np.mean(np.square(centered_data), axis=0))
    std_data = np.divide(centered_data, std_deviation)
    return std_data


def covariance_matrix(data):
    return np.matmul(np.transpose(data), data) / data.shape[0]


def scatter_matrix_w(data):
    data = data - np.mean(data, axis=0)
    return np.sum([np.outer(data_point, data_point)
                   for data_point in data],
                  axis=0)


def scatter_matrix_b(data, global_mu):
    diff_mu = np.mean(data, axis=0) - global_mu
    return np.outer(diff_mu, diff_mu) * data.shape[0]


class PCA_():
    def __init__(self, data: list) -> None:
        # Standardize data. (center and normalize)
        # std_data = standardize(data)
        std_data = data
        # Calculate the covariance matrix.
        cov_matrix = covariance_matrix(std_data)

        # Calculate eigenvalues and eigenvectors.
        eig_values, eig_vectors = np.linalg.eig(cov_matrix)

        # Sort eigenvectors in decreasing order according to their eigenvalues.
        sorted_indexes = np.argsort(eig_values)[::-1]
        eig_values = eig_values[sorted_indexes]
        eig_vectors = eig_vectors[:, sorted_indexes]

        self.eig_vectors = eig_vectors

    def fit(self, data: list, k: int) -> list:
        return np.matmul(data, self.eig_vectors[:, :k])


def PCA(data, k):

    assert 0 < k <= data.shape[1]

    # Standardize data. (center and normalize)
    # std_data = standardize(data)
    std_data = data
    # Calculate the covariance matrix.
    cov_matrix = covariance_matrix(std_data)

    # Calculate eigenvalues and eigenvectors.
    eig_values, eig_vectors = np.linalg.eig(cov_matrix)

    # Sort eigenvectors in decreasing order according to their eigenvalues.
    sorted_indexes = np.argsort(eig_values)[::-1]
    eig_values = eig_values[sorted_indexes]
    eig_vectors = eig_vectors[:, sorted_indexes]

    # Projection of all data on the first k principle components.
    reduced_data = np.matmul(std_data, eig_vectors[:, :k])

    return reduced_data


def LDA(data, label, k):

    assert 0 < k < data.shape[1]

    # Standardize data. Note: Was not mentioned in the exercise but was
    # mentioned in many sources.
    # data = standardize(data)

    # Compute mean of all data.
    mu_data = np.average(data)

    # Compute the scatter matrix for each class.
    # TODO: Not clear if it should be squared or not... Don't think so.
    Sw = np.sum([scatter_matrix_w(data[label == i])
                 for i in range(max(label))],
                axis=0)
    Sb = np.sum([scatter_matrix_b(data[label == i], mu_data)
                 for i in range(max(label))],
                axis=0)

    # Calculate S.
    S = np.matmul(np.linalg.inv(Sw), Sb)

    # Calculate eigenvalues and eigenvectors.
    eig_values, eig_vectors = np.linalg.eig(S)

    # Sort eigenvectors in decreasing order according to their eigenvalues.
    sorted_indexes = np.argsort(eig_values)[::-1]
    eig_values = eig_values[sorted_indexes]
    eig_vectors = eig_vectors[:, sorted_indexes]

    # Projection of all data on the first two principle components.
    reduced_data = np.matmul(data, eig_vectors[:, :k])

    return reduced_data


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from sklearn import datasets

    sns.set()

    iris = datasets.load_iris()
    data, targets = iris['data'], iris['target']

    # Scatterplot projection on the first two principle components using PCA.
    reduced_data = PCA(data, 2)
    reduced_dataframe = pd.DataFrame(data=np.c_[reduced_data, targets],
                                     columns=["x", "y"] + ['target'])
    sns.scatterplot(data=reduced_dataframe, x="x", y="y", hue="target")
    plt.show()

    # Scatterplot projection on the first two principle component using LDA.
    reduced_data = LDA(data, targets, 2)
    reduced_dataframe = pd.DataFrame(data=np.c_[reduced_data, targets],
                                     columns=["x", "y"] + ['target'])
    sns.scatterplot(data=reduced_dataframe, x="x", y="y", hue="target")
    plt.show()
