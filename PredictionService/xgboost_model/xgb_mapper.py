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


def init_xgb_mapper(df, xgb_pickle=PredictionServiceConfig.XGB_MAPPER):
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
        (constants.DISCRETE_ROOM_AREA, None),
        (constants.MONTH, None),

    ], input_df=True)

    data_temp = np.round(mapper.fit_transform(df.copy()).astype(np.double), 3)

    if check_local_file_exist(xgb_pickle):
        os.remove(xgb_pickle)

    with open(xgb_pickle, "wb") as f:
        pickle.dump(mapper, f)
    print("Fitting: ", type(mapper))
    del data_temp
    gc.collect()
    return


def proc_xgb_mapper(df):
    pickle_train = PredictionServiceConfig.XGB_MAPPER
    if os.path.isfile(pickle_train):
        with open(pickle_train, 'rb') as f:
            mapper = pickle.load(f)
            data_tmp = np.round(mapper.transform(df.copy()).astype(np.double), 3)
            return pd.DataFrame(data_tmp, columns=mapper.transformed_names_)
    else:
        print('Cannot find xgb_mapper.pkl')

