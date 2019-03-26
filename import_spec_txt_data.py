import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import utils_cretin_m1_sym as utils_cretin
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d


class SimulatedDataSet:

    def __init__(self, pca_components=30):
        self.hu, xtrain, ytrain, xtest, ytest, X, Y = utils_cretin.cretin_data()
        self.pca = PCA(n_components=pca_components) 
        self.pca.fit(xtrain)


def import_as_np(fname, new_hu):
    data = np.loadtxt(fname, skiprows=1)
    hu, I, *_ = data[(data[:, 0] > 10) & (data[:, 0] < 13)]
    assert not np.isinf(d).any()  # make sure no infinity values
    assert not np.isnan(d).any()  # make sure no nan values
    return interp1d(hu, I)(new_hu)


if __name__ == '__main__':
    sim_dataset = SimulatedDataSet()
    X = import_as_np('nif_ge_shell_data.txt', sim_dataset.hu)
    transformed_X = sim_dataset.pca.transform(X)

