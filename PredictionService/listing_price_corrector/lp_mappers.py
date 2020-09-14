import sklearn
from sklearn_pandas import DataFrameMapper
import pickle
import numpy as np
import pandas as pd
import gc
import os
from PredictionService.config import PredictionServiceConfig
from PredictionService.config import constants


def initial_mappers(df_all):
    regression_train_mapper_init(df_all)
    regression_predict_mapper_init(df_all)


def regression_train_mapper_init(df):
    pickle_train = PredictionServiceConfig.PICKLE_REG_TRAIN

    mapper = DataFrameMapper([
        ('A_c', sklearn.preprocessing.LabelBinarizer()),
        ('Bsmt1_out', sklearn.preprocessing.LabelBinarizer()),
        ('Gar_type', sklearn.preprocessing.LabelBinarizer()),
        ('Heating', sklearn.preprocessing.LabelBinarizer()),
        ('Pool', sklearn.preprocessing.LabelBinarizer()),
        ('Style', sklearn.preprocessing.LabelBinarizer()),
        ('Type_own1_out', sklearn.preprocessing.LabelBinarizer()),
        ('Den_fr', sklearn.preprocessing.LabelBinarizer()),

        (['Dom'], sklearn.preprocessing.StandardScaler()),
        (['Taxes'], sklearn.preprocessing.StandardScaler()),
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
        (constants.MONTH, None),
        (['Sp_dol'], None)

    ], input_df=True)

    data_temp = np.round(mapper.fit_transform(df.copy()).astype(np.double), 3)

    with open(pickle_train, "wb") as f:
        pickle.dump(mapper, f)
    del data_temp
    gc.collect()
    return


def regression_train_mapper_proc(df):
    pickle_train = PredictionServiceConfig.PICKLE_REG_TRAIN
    if os.path.isfile(pickle_train):
        with open(pickle_train, 'rb') as f:
            mapper = pickle.load(f)
            data_tmp = np.round(mapper.transform(df.copy()).astype(np.double), 3)
            return pd.DataFrame(data_tmp, columns=mapper.transformed_names_)
    else:
        print('Cannot find train_regression_mapper.pkl')


def regression_predict_mapper_init(df):
    pickle_predict = PredictionServiceConfig.PICKLE_REG_PREDICT

    mapper = DataFrameMapper([
        ('A_c', sklearn.preprocessing.LabelBinarizer()),
        ('Bsmt1_out', sklearn.preprocessing.LabelBinarizer()),
        ('Gar_type', sklearn.preprocessing.LabelBinarizer()),
        ('Heating', sklearn.preprocessing.LabelBinarizer()),
        ('Pool', sklearn.preprocessing.LabelBinarizer()),
        ('Style', sklearn.preprocessing.LabelBinarizer()),
        ('Type_own1_out', sklearn.preprocessing.LabelBinarizer()),
        ('Den_fr', sklearn.preprocessing.LabelBinarizer()),

        (['Dom'], sklearn.preprocessing.StandardScaler()),
        (['Taxes'], sklearn.preprocessing.StandardScaler()),
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
        (constants.MONTH, None),

    ], input_df=True)

    data_temp = np.round(mapper.fit_transform(df.copy()).astype(np.double), 3)

    with open(pickle_predict, "wb") as f:
        pickle.dump(mapper, f)
    del data_temp
    gc.collect()
    return


def regression_predict_mapper_proc(df):
    pickle_predict = PredictionServiceConfig.PICKLE_REG_PREDICT
    if os.path.isfile(pickle_predict):
        with open(pickle_predict, 'rb') as f:
            mapper = pickle.load(f)
            data_tmp = np.round(mapper.transform(df.copy()).astype(np.double), 3)
            return pd.DataFrame(data_tmp, columns=mapper.transformed_names_)
    else:
        print('Cannot find predict_regression_mapper.pkl')
