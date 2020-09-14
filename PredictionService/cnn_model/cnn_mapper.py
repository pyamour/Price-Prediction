import sklearn
from sklearn_pandas import DataFrameMapper
import pickle
import os
import sys
import numpy as np
import pandas as pd
from PredictionService.config import PredictionServiceConfig
from PredictionService.utils.utils import check_local_file_exist

all_columns = ['_id', 'Dom', 'A_c', 'Area_code', 'Bath_tot', 'Br', 'Br_plus', 'Bsmt1_out',
               'Community', 'Park_spcs', 'Gar_type', 'Heating', 'Kit_plus', 'Lp_dol',
               'Pool', 'Rms', 'Rooms_plus', 'Style', 'Type_own1_out', 'Cd', 'lat', 'lng', 'Taxes',
               'Den_fr', 'Depth', 'Front_ft', 'Garage', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'less-than50', '50-to-100', '100-to-150',
               '150-to-200', '200-to-250', '250-to-350', 'larger-than350']

exps = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
        'less-than50', '50-to-100', '100-to-150',
        '150-to-200', '200-to-250', '250-to-350', 'larger-than350',
        'Taxes', 'Lp_dol']


def process_data_for_cnn(df, usefor):  # usefor == 'train'; usefor == 'predict'
    # .csv file check
    if usefor == 'train':
        feature_columns = all_columns + ['Sp_dol']
        # sort based on Cd
        df.sort_values(by=['Cd'], ascending=True, inplace=True)
    elif usefor == 'predict':
        feature_columns = all_columns
    else:
        print("Wrong parameter value: usefor should be either 'train' or 'predict'.")
        sys.exit()
    if not list(feature_columns).sort() == list(df.columns.tolist()).sort():
        print('Error: make sure u r using the correct .csv file')
        sys.exit()
    # remove redundant columns
    df = df.drop(['Cd'], axis=1)
    df = df.drop(['Bath_tot', 'Br', 'Br_plus'], axis=1)
    df['Total_rooms'] = df['Rms'] + df['Rooms_plus']
    df = df.drop(['Rms', 'Rooms_plus'], axis=1)

    # apply log(1+x) to all elements of the column
    df["Taxes"] = np.log1p(df["Taxes"])
    df["Lp_dol"] = np.log1p(df["Lp_dol"])
    # convert objects to nums and downsampling
    # ‘A_c’
    maps = {'Central Air': 1, 'None': 0, 'Wall Unit': 1,
            'Window Unit': 1, 'Other': 1}
    df['A_c'] = df['A_c'].map(maps)
    # 'Pool', it matters whether has pool or not, if has, give 1
    maps = {'None': 0, 'Inground': 1, 'Abv Grnd': 1,
            'Indoor': 1, 'Outdoor': 1}
    df['Pool'] = df['Pool'].map(maps)
    # 'Den_fr'
    maps = {'Y': 1, 'N': 0}
    df['Den_fr'] = df['Den_fr'].map(maps)
    return df


def process_train_data(df, timenow):
    df = process_data_for_cnn(df, usefor='train')
    res = init_cnn_mapper(df)
    col_list = list(res.columns.values)
    res['_id'] = df['_id']
    res["Sp_dol"] = np.log1p(df["Sp_dol"])
    print(res["Sp_dol"])
    col_list.insert(0, '_id')
    # col_list.remove('Sp_dol')
    if check_local_file_exist(PredictionServiceConfig.CNN_COL_ORDER):
        os.remove(PredictionServiceConfig.CNN_COL_ORDER)
    # generate columns order
    print('colums order: ', col_list)
    with open(PredictionServiceConfig.CNN_COL_ORDER, 'wb') as fp:
        pickle.dump(col_list, fp)
    col_list.append('Sp_dol')
    res = res[col_list]
    res.index = range(len(res))
    res.to_csv(PredictionServiceConfig.CNN_PATH + 'cnn_data' + str(timenow) + '.csv', index=False)
    return res


def process_predict_data(df):
    df = process_data_for_cnn(df, usefor='predict')
    # get the mapper which was used for the whole dataset(train + validate + test)
    with open(PredictionServiceConfig.CNN_MAPPER, 'rb') as f:
        mapper = pickle.load(f)
        # apply the mapper
        res = mapper.transform(df.copy()).astype(np.double)
    res['_id'] = df['_id']
    with open(PredictionServiceConfig.CNN_COL_ORDER, 'rb') as fp:
        columns_order = pickle.load(fp)
    res = res[columns_order]
    res.index = range(len(res))
    return res


def init_cnn_mapper(df, cnn_pickle=PredictionServiceConfig.CNN_MAPPER):
    mapper = DataFrameMapper([
        ('Lp_dol', None),
        ('Taxes', None),
        ('Jan', None),
        ('Feb', None),
        ('Mar', None),
        ('Apr', None),
        ('May', None),
        ('Jun', None),
        ('Jul', None),
        ('Aug', None),
        ('Sep', None),
        ('Oct', None),
        ('Nov', None),
        ('Dec', None),
        ('less-than50', None),
        ('50-to-100', None),
        ('100-to-150', None),
        ('150-to-200', None),
        ('200-to-250', None),
        ('250-to-350', None),
        ('larger-than350', None),
        (['Bsmt1_out'], sklearn.preprocessing.LabelBinarizer()),
        (['Gar_type'], sklearn.preprocessing.LabelBinarizer()),
        (['Heating'], sklearn.preprocessing.LabelBinarizer()),
        (['Style'], sklearn.preprocessing.LabelBinarizer()),
        (['Type_own1_out'], sklearn.preprocessing.LabelBinarizer()),
        (['Community'], sklearn.preprocessing.LabelBinarizer()),

        (['Dom'], sklearn.preprocessing.MinMaxScaler()),
        (['A_c'], sklearn.preprocessing.MinMaxScaler()),
        (['Area_code'], sklearn.preprocessing.MinMaxScaler()),
        (['Park_spcs'], sklearn.preprocessing.MinMaxScaler()),
        (['Kit_plus'], sklearn.preprocessing.MinMaxScaler()),
        (['Pool'], sklearn.preprocessing.MinMaxScaler()),
        (['lat'], sklearn.preprocessing.MinMaxScaler()),
        (['lng'], sklearn.preprocessing.MinMaxScaler()),
        (['Den_fr'], sklearn.preprocessing.MinMaxScaler()),
        (['Depth'], sklearn.preprocessing.MinMaxScaler()),
        (['Front_ft'], sklearn.preprocessing.MinMaxScaler()),
        (['Garage'], sklearn.preprocessing.MinMaxScaler()),
        (['Total_rooms'], sklearn.preprocessing.MinMaxScaler()),
    ], df_out=True)

    data_mapper = np.round(mapper.fit_transform(df.copy()).astype(np.double), 3)

    # save as a cnn mapper
    if check_local_file_exist(cnn_pickle):
        os.remove(cnn_pickle)
    with open(cnn_pickle, "wb") as f:
        pickle.dump(mapper, f)
    print("Fitting: ", type(mapper))
    return data_mapper


if __name__ == '__main__':
    df = pd.read_csv(PredictionServiceConfig.DATA_PATH + 'test_clean_house_data_2019523.csv')
    df = process_train_data(df, '1212')
    print(df)
