

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

import os


import numpy as np
import matplotlib.pyplot as plt

import time
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

os.environ["CUDA_VISIBLE_DEVICES"] = "0"








def preprocess(dataset, m, npoints_output, is_test=False):
    # Given a dataset, preprocess it to get the input for branch and trunk networks
    # m is the number of points in the branch network
    # npoints_output is the number of points in the trunk network
    p = dataset['del_p']
    t = dataset['t']
    r = dataset['R']

    P = interp1d(t, p, kind='cubic')
    R = interp1d(t, r, kind='cubic')

    t_min = 0
    t_max = 5 * 10**-4

    X_func = P(np.linspace(t_min, t_max, m)) #[1500, m]
    X_loc  = np.linspace(t_min, t_max, npoints_output)[:, None] #[npoints_output,1]
    y      = R(np.ravel(X_loc)) #[1500, npoints_output]

    if is_test:
        X_func = X_func[:50]
        y = y[:50]

    return X_func, X_loc, y

def normalize(X_func, X_loc, y, Par):
    X_func = (X_func - Par['p_mean'])/Par['p_std'] #Tarun and Sayan
    X_loc  = (X_loc - np.min(X_loc))/(np.max(X_loc) - np.min(X_loc))
    y = (y - Par['r_mean'])/Par['r_std'] #Tarun and Sayan

    return X_func.astype(np.float32), X_loc.astype(np.float32), y.astype(np.float32)

def train(don_model, X_func, X_loc, y):
    #TODO: Implement loss computation and backward propagation
    return loss

def main():
    Par = {}
    train_dataset = np.load('res_1000_1.npz')
    test_dataset = np.load('res_1000_08.npz')

    m = 200
    npoints_output = 500

    Par['address'] = 'don_'+str(m)

    print(Par['address'])
    print('------\n')

    X_func_train, X_loc_train, y_train = preprocess(train_dataset, m, npoints_output)
    X_func_test, X_loc_test, y_test = preprocess(test_dataset, m, npoints_output, is_test=True)

    Par['p_mean'] = np.mean(X_func_train)
    Par['p_std']  = np.std(X_func_train)

    Par['r_mean'] = np.mean(y_train)
    Par['r_std']  = np.std(y_train)

    X_func_train, X_loc_train, y_train = normalize(X_func_train, X_loc_train, y_train, Par)
    X_func_test, X_loc_test, y_test    = normalize(X_func_test, X_loc_test, y_test, Par)

    print('X_func_train: ', X_func_train.shape, '\nX_loc_train: ', X_loc_train.shape, '\ny_train: ', y_train.shape)
    print('X_func_test: ', X_func_test.shape, '\nX_loc_test: ', X_loc_test.shape, '\ny_test: ', y_test.shape)

    #TODO: Implement training loop


if __name__ == "__main__":
    main()
