import sklearn
from sklearn_pandas import DataFrameMapper
import pickle
import numpy as np
import pandas as pd
import gc
import os
from PredictionService.config import PredictionServiceConfig
from PredictionService.config import constants
from PredictionService.utils.utils import check_local_file_exist

'''
identify feature names
'''


def identify_feature_names(num):
    feature_name = []
    for i in range(num):
        feature_name.append('cnn' + str(num) + '_' + str(i))
    return feature_name


def initial_mappers(data_file):
    df_all = pd.read_csv(data_file, sep=',')
    pickle_train = PredictionServiceConfig.PICKLE_FM_TRAIN
    pickle_target = PredictionServiceConfig.PICKLE_FM_TARGET

    if check_local_file_exist(pickle_train):
        os.remove(pickle_train)
    if check_local_file_exist(pickle_target):
        os.remove(pickle_target)

    mapper_init_train(df_all, pickle_train)
    mapper_init_target(df_all, pickle_target)


def mapper_init_train(df, pickle_train):
    cnn_features256 = identify_feature_names(256)
    cnn_features1000 = identify_feature_names(1000)
    cnn_features1256 = cnn_features256 + cnn_features1000
    mapper = DataFrameMapper([
        ('A_c', sklearn.preprocessing.LabelBinarizer()),
        ('Bsmt1_out', sklearn.preprocessing.LabelBinarizer()),
        ('Community', sklearn.preprocessing.LabelBinarizer()),
        ('Gar_type', sklearn.preprocessing.LabelBinarizer()),
        ('Heating', sklearn.preprocessing.LabelBinarizer()),
        ('Pool', sklearn.preprocessing.LabelBinarizer()),
        ('Style', sklearn.preprocessing.LabelBinarizer()),
        ('Type_own1_out', sklearn.preprocessing.LabelBinarizer()),
        ('Den_fr', sklearn.preprocessing.LabelBinarizer()),

        (['Dom'], sklearn.preprocessing.StandardScaler()),
        (['Taxes'], sklearn.preprocessing.StandardScaler()),
        (['Area_code'], sklearn.preprocessing.StandardScaler()),
        (['Depth'], sklearn.preprocessing.StandardScaler()),
        (['Front_ft'], sklearn.preprocessing.StandardScaler()),
        (['Bath_tot'], sklearn.preprocessing.StandardScaler()),
        (['Br'], sklearn.preprocessing.StandardScaler()),
        (['Br_plus'], sklearn.preprocessing.StandardScaler()),
        (['Park_spcs'], sklearn.preprocessing.StandardScaler()),
        (['Kit_plus'], sklearn.preprocessing.StandardScaler()),
        (['Rms'], sklearn.preprocessing.StandardScaler()),
        (['Rooms_plus'], sklearn.preprocessing.StandardScaler()),
        (['Garage'], sklearn.preprocessing.StandardScaler()),
        (['lat'], sklearn.preprocessing.StandardScaler()),
        (['lng'], sklearn.preprocessing.StandardScaler()),
        (['Lp_dol'], sklearn.preprocessing.StandardScaler()),
        (['Sp_dol'], sklearn.preprocessing.StandardScaler()),

        (['num_pic'], sklearn.preprocessing.StandardScaler()),
        # (cnn_features1000, None),
        # (cnn_features256, None),
        # (cnn_features1256, None),

        (constants.DISCRETE_ROOM_AREA, None),
        (constants.MONTH, None),

    ], input_df=True)

    data_temp = np.round(mapper.fit_transform(df.copy()).astype(np.double), 3)

    with open(pickle_train, "wb") as f:
        pickle.dump(mapper, f)
    print("Fitting: ", type(mapper))
    del data_temp
    gc.collect()
    return


def mapper_init_target(df, pickle_target):
    cnn_features256 = identify_feature_names(256)
    cnn_features1000 = identify_feature_names(1000)
    cnn_features1256 = cnn_features256 + cnn_features1000
    mapper = DataFrameMapper([
        ('A_c', sklearn.preprocessing.LabelBinarizer()),
        ('Bsmt1_out', sklearn.preprocessing.LabelBinarizer()),
        ('Community', sklearn.preprocessing.LabelBinarizer()),
        ('Gar_type', sklearn.preprocessing.LabelBinarizer()),
        ('Heating', sklearn.preprocessing.LabelBinarizer()),
        ('Pool', sklearn.preprocessing.LabelBinarizer()),
        ('Style', sklearn.preprocessing.LabelBinarizer()),
        ('Type_own1_out', sklearn.preprocessing.LabelBinarizer()),
        ('Den_fr', sklearn.preprocessing.LabelBinarizer()),

        (['Dom'], sklearn.preprocessing.StandardScaler()),
        (['Taxes'], sklearn.preprocessing.StandardScaler()),
        (['Area_code'], sklearn.preprocessing.StandardScaler()),
        (['Depth'], sklearn.preprocessing.StandardScaler()),
        (['Front_ft'], sklearn.preprocessing.StandardScaler()),
        (['Bath_tot'], sklearn.preprocessing.StandardScaler()),
        (['Br'], sklearn.preprocessing.StandardScaler()),
        (['Br_plus'], sklearn.preprocessing.StandardScaler()),
        (['Park_spcs'], sklearn.preprocessing.StandardScaler()),
        (['Kit_plus'], sklearn.preprocessing.StandardScaler()),
        (['Rms'], sklearn.preprocessing.StandardScaler()),
        (['Rooms_plus'], sklearn.preprocessing.StandardScaler()),
        (['Garage'], sklearn.preprocessing.StandardScaler()),
        (['lat'], sklearn.preprocessing.StandardScaler()),
        (['lng'], sklearn.preprocessing.StandardScaler()),
        (['Lp_dol'], sklearn.preprocessing.StandardScaler()),

        (['num_pic'], sklearn.preprocessing.StandardScaler()),
        # (cnn_features1000, None),
        # (cnn_features256, None),
        # (cnn_features1256, None),

        (constants.DISCRETE_ROOM_AREA, None),
        (constants.MONTH, None),
    ], input_df=True)

    data_temp = np.round(mapper.fit_transform(df.copy()).astype(np.double), 3)

    with open(pickle_target, "wb") as f:
        pickle.dump(mapper, f)
    print("Fitting: ", type(mapper))
    del data_temp
    gc.collect()
    return


def mapper_proc_train(df):
    pickle_train = PredictionServiceConfig.PICKLE_FM_TRAIN
    if os.path.isfile(pickle_train):
        with open(pickle_train, 'rb') as f:
            mapper = pickle.load(f)
            data_tmp = np.round(mapper.transform(df.copy()).astype(np.double), 3)
            return pd.DataFrame(data_tmp, columns=mapper.transformed_names_)
    else:
        print('Cannot find mapper_train.pkl')


def mapper_proc_target(df):
    pickle_target = PredictionServiceConfig.PICKLE_FM_TARGET
    if os.path.isfile(pickle_target):
        with open(pickle_target, 'rb') as f:
            mapper = pickle.load(f)
            data_tmp = np.round(mapper.transform(df.copy()).astype(np.double), 3)
            return pd.DataFrame(data_tmp, columns=mapper.transformed_names_)
    else:
        print('Cannot find mapper_target.pkl')
