import numpy as np


# automatically calculate the input dimensions and number of zero columns need to be added
def calculate_input_dims_and_zero_cols(num_feature):
    dim = 1
    num_zero = dim ** 2 + dim - num_feature
    while num_zero < 0:
        num_zero = dim ** 2 + dim - num_feature
        dim += 1
    print('...num_dim, num_zero:', dim - 1, num_zero)

    return dim - 1, num_zero


def reshape_X(cnn_X, num_dim, num_zero):
    zeros_X = np.zeros((len(cnn_X), num_zero))
    X = np.append(cnn_X, zeros_X, axis=1)
    X = np.reshape(X, (-1, num_dim + 1, num_dim)).astype(float)
    X = np.reshape(X, (-1, 1, num_dim + 1, num_dim))
    return X
