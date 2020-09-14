import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import ensemble
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras.layers import Convolution2D, MaxPooling2D, Flatten, AveragePooling2D
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.regularizers import l1, l2
from PredictionService.cnn_model.show_accuracy import Show_Accuracy
from PredictionService.config import PredictionServiceConfig
from PredictionService.config import constants
from PredictionService.cnn_model.cnn_mapper import init_cnn_mapper, process_data_for_cnn


# build CNN model
def cnnModel(hyp1, hyp2, num_dim):
    print('Building the CNN model...')
    model = Sequential()
    # conv block 1
    model.add(Convolution2D(batch_input_shape=(None, 1, num_dim + 1, num_dim),
                            filters=8,
                            kernel_size=3,
                            strides=1,
                            padding='same',
                            data_format='channels_first',
                            kernel_initializer='glorot_uniform',
                            activity_regularizer=l1(hyp1),
                            kernel_regularizer=l2(hyp2)))  # (8, 28, 27)
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Convolution2D(8, 3, strides=1, padding='same', data_format='channels_first',
                            kernel_initializer='glorot_uniform',
                            activity_regularizer=l1(hyp1),
                            kernel_regularizer=l2(hyp2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))  # (8, 14, 14)
    # model.add(Dropout(0.5))

    # conv block 2
    model.add(Convolution2D(16, 3, strides=1, padding='same', data_format='channels_first',
                            kernel_initializer='glorot_uniform',
                            activity_regularizer=l1(hyp1),
                            kernel_regularizer=l2(hyp2)))  # (16, 14, 14)
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Convolution2D(16, 3, strides=1, padding='same', data_format='channels_first',
                            kernel_initializer='glorot_uniform',
                            activity_regularizer=l1(hyp1),
                            kernel_regularizer=l2(hyp2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))  # (16, 7, 7)
    # model.add(Dropout(0.5))

    # fully connected layers
    model.add(Flatten())
    model.add(Dense(512, kernel_initializer='glorot_uniform',
                    activity_regularizer=l1(hyp1),
                    kernel_regularizer=l2(hyp2)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))

    model.add(Dense(512, kernel_initializer='glorot_uniform',
                    activity_regularizer=l1(hyp1),
                    kernel_regularizer=l2(hyp2)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))

    model.add(Dense(1, kernel_initializer='glorot_uniform'))

    # model.add(Convolution2D(32, 7, strides=1, padding='valid', data_format='channels_first'))  # (32, 1, 1)
    # model.add(Dropout(0.05))
    # model.add(Convolution2D(1, 1, strides=1, padding='same', data_format='channels_first'))  # (1, 1, 1)
    # output
    # model.add(Flatten())
    return model


# build DNN model
def dnnModel():
    print('Building the DNN model...')
    model = Sequential()
    # TODO: how to get them?
    # params of each layer
    layers = [31, 64, 256, 1]
    # define each layer
    for layer, numInputs in enumerate(layers):
        if layer == 0:
            # define the input layer
            model.add(Dense(numInputs, input_dim=numInputs, kernel_initializer='glorot_uniform'))
            # model.add(BatchNormalization())
            model.add(Activation('relu'))
            # model.add(LeakyReLU(alpha=0.1))
            # model.add(BatchNormalization())
        elif layer != len(layers) - 1:
            # define hiddenlayers
            model.add(Dense(numInputs, kernel_initializer='glorot_uniform'))
            # model.add(Dense(numInputs, kernel_initializer='glorot_uniform', activity_regularizer=l1(0.0003)))
            # model.add(BatchNormalization())
            model.add(Activation('relu'))
            # model.add(LeakyReLU(alpha=0.1))
            # model.add(BatchNormalization())
            # model.add(Dropout(0.1))
        else:
            # define output layer
            model.add(Dense(layers[-1], kernel_initializer='glorot_uniform'))
            # model.add(Activation('linear'))
    return model
