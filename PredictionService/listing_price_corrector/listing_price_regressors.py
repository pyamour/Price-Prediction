import pandas as pd
import numpy as np
from PredictionService.listing_price_corrector import lp_mappers as mapper
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
import math
import dill as pickle
from PredictionService.config import PredictionServiceConfig
from PredictionService.config import constants
from multiprocessing import Pool
from datetime import timedelta
from PredictionService.utils.utils import check_local_file_exist, remove_files_in_dir
from itertools import repeat


def multiprocess_train_regressors(df):
    # remove regressors first
    remove_files_in_dir(PredictionServiceConfig.LP_MODEL_PATH, '*.pkl')
    df = rename_community(df, 'Community')
    period_1year = '1year'
    period_5months = '5months'
    period_2months = '2months'

    comm_arr_1year, df_house_1year = get_comm_list(period_1year, df)
    comm_arr_5months, df_house_5months = get_comm_list(period_5months, df)
    comm_arr_2months, df_house_2months = get_comm_list(period_2months, df)

    pool = Pool(processes=int(constants.CORE * 0.8))
    # starmap_async
    pool.starmap_async(train_regressor_for_each_community,
                       zip(comm_arr_1year, repeat(period_1year), repeat(df_house_1year)))
    pool.starmap_async(train_regressor_for_each_community,
                       zip(comm_arr_5months, repeat(period_5months), repeat(df_house_5months)))
    pool.starmap_async(train_regressor_for_each_community,
                       zip(comm_arr_2months, repeat(period_2months), repeat(df_house_2months)))
    pool.close()
    pool.join()


def train_regressor_for_each_community(comm, period, df_house):
    df_train = df_house[df_house['Community'] == comm]
    if len(df_train) == 0:
        print(comm)
        return
    df_normalize = mapper.regression_train_mapper_proc(df_train)
    y_train = df_normalize['Sp_dol']
    X_train = df_normalize.drop(columns=['Sp_dol'])
    regressor_name = comm + '_regressor'

    fit_gradient_boost_random_search_regressor(X_train, y_train, regressor_name, period)
    fit_random_forest_random_search_regressor(X_train, y_train, regressor_name, period)
    fit_adaboost_random_search_regressor(X_train, y_train, regressor_name, period)


def rename_community(df, col_name):
    df_copy = df.copy(deep=True)
    df_copy[col_name] = df_copy[col_name].str.replace('/', '_')
    df_copy[col_name] = df_copy[col_name].str.replace(' ', '_')
    return df_copy


def get_the_latest_date(df_house):
    df_house['Cd'] = pd.to_datetime(df_house['Cd'])
    df_house.sort_values(by=['Cd'], ascending=False, inplace=True)  # from now to long long ago
    df_house.index = range(len(df_house))
    date_now = df_house.at[0, 'Cd']
    return date_now


def get_comm_list(period, df_house=None):
    comm_count_file_path = PredictionServiceConfig.LP_PATH + PredictionServiceConfig.COMM_COUNT_PREFIX + period + '.csv'
    if df_house is not None:

        date_now = get_the_latest_date(df_house)
        if period == '2months':
            start_date = date_now + timedelta(days=-60)

        elif period == '5months':
            start_date = date_now + timedelta(days=-150)

        elif period == '1year':
            start_date = date_now + timedelta(days=-360)

        df_house = df_house[df_house['Cd'] >= start_date]
        df_comm_count = df_house.groupby(['Community']).count()
        df_comm_count = df_comm_count['Area_code']
        df_comm_count = df_comm_count.reset_index()
        df_comm_count.columns = ['Community', 'count']
        df_comm_count.to_csv(comm_count_file_path, index=None)
        df_comm_count = rename_community(df_comm_count, 'Community')
        comm_arr = list(set(
            df_comm_count[(df_comm_count['count'] >= 12) & (df_comm_count['Community'] != 'Other')][
                'Community'].values))
        return comm_arr, df_house

    elif check_local_file_exist(comm_count_file_path):
        df_comm_count = pd.read_csv(comm_count_file_path)
        df_comm_count = rename_community(df_comm_count, 'Community')
        comm_arr = list(set(
            df_comm_count[(df_comm_count['count'] >= 12) & (df_comm_count['Community'] != 'Other')][
                'Community'].values))
        return comm_arr
    else:
        print("Cannot find comm_count file!")
        return None


def get_depth_range(X):
    depth = int(math.log(len(X.columns), 2))
    return sp_randint(max((depth - 2), 1), depth + 2)


def fit_gradient_boost_random_search_regressor(X, y, regressor_name, period):
    param_dist = {
        'learning_rate': np.logspace(-3, 2, 10),
        'n_estimators': sp_randint(20, 150),
        'min_samples_split': sp_randint(2, 15),
        'min_samples_leaf': sp_randint(1, 10),
        'max_depth': get_depth_range(X)
    }
    gb_model = GradientBoostingRegressor()
    random_search = RandomizedSearchCV(gb_model, param_distributions=param_dist, n_iter=20, cv=min(int(len(X) / 4), 10))
    random_search.fit(X, y)
    pickle.dump(random_search,
                open(
                    PredictionServiceConfig.LP_MODEL_PATH + PredictionServiceConfig.GB_PREFIX + regressor_name + '-' + period + '.pkl',
                    'wb'))


def fit_random_forest_random_search_regressor(X, y, regressor_name, period):
    param_dist = {
        'max_depth': get_depth_range(X),
        'max_features': sp_randint(1, len(X.columns)),
        'min_samples_split': sp_randint(2, 11),
    }
    rf_model = RandomForestRegressor()
    random_search = RandomizedSearchCV(rf_model, param_distributions=param_dist, n_iter=20, cv=min(int(len(X) / 4), 10))
    random_search.fit(X, y)
    pickle.dump(random_search,
                open(
                    PredictionServiceConfig.LP_MODEL_PATH + PredictionServiceConfig.RF_PREFIX + regressor_name + '-' + period + '.pkl',
                    'wb'))


def fit_adaboost_random_search_regressor(X, y, regressor_name, period):
    param_dist = {
        'n_estimators': sp_randint(20, 150),
        'learning_rate': np.logspace(-3, 2, 10)
    }
    ada_model = AdaBoostRegressor()
    random_search = RandomizedSearchCV(ada_model, param_distributions=param_dist, n_iter=20,
                                       cv=min(int(len(X) / 4), 10))
    random_search.fit(X, y)
    pickle.dump(random_search,
                open(
                    PredictionServiceConfig.LP_MODEL_PATH + PredictionServiceConfig.ADA_PREFIX + regressor_name + '-' + period + '.pkl',
                    'wb'))
    return


def test_regressor():
    house_file = '/var/csv_file/clean_house_data.csv'
    df_house = pd.read_csv(house_file)
    multiprocess_train_regressors(df_house)
