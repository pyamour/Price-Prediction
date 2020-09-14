import dill as pickle
import pandas as pd
import os
from PredictionService.listing_price_corrector import lp_mappers as mapper
from PredictionService.listing_price_corrector.listing_price_regressors import rename_community, get_comm_list, \
    get_the_latest_date
from PredictionService.config import PredictionServiceConfig
from datetime import timedelta


def get_nearby_sold_house(lat, lng, df_index):
    lat1 = lat - 0.027
    lat2 = lat + 0.027
    lng1 = lng - 0.037
    lng2 = lng + 0.037
    df_nearby = df_index[
        (df_index['lat'] >= lat1) & (df_index['lat'] <= lat2) & (df_index['lng'] >= lng1) & (df_index['lng'] <= lng2)]
    nearby_num = len(df_nearby)

    return nearby_num


def adjust_lp(df_compare, df_house):
    date_now = get_the_latest_date(df_house)
    start_date = date_now + timedelta(days=-200)
    df_index = df_house[df_house['Cd'] >= start_date][['_id', 'Cd', 'lat', 'lng']]
    id = []
    for i, row in df_compare.iterrows():

        lp = row['Lp_dol']
        gb = row['gb_pred']
        ada = row['ada_pred']
        rf = row['rf_pred']

        diff_gb = cal_diff(lp, gb)
        diff_ada = cal_diff(lp, ada)
        diff_rf = cal_diff(lp, rf)

        diff_dic = {diff_gb: gb, diff_ada: ada, diff_rf: rf}

        diff_arr = sorted([diff_gb, diff_ada, diff_rf])  # small -> large

        nearby_sold_num = get_nearby_sold_house(row['lat'], row['lng'], df_index)

        if sum(diff_arr) == 3:
            continue
        elif ((lp > 2000000) or (lp < 300000)) and (diff_arr[0] <= 1.24857):
            continue
        elif diff_arr[0] <= 0.142857:
            continue
        elif nearby_sold_num < 3:
            continue
        else:
            df_compare.at[i, 'Lp_dol'] = diff_dic[diff_arr[0]]
            id.append(i)
    df_compare.drop(columns=['gb_pred', 'ada_pred', 'rf_pred'], inplace=True)
    return df_compare, id


def predict_lp_for_each_period(df, df_test, X_test, regressor_name, period):
    gb_path = PredictionServiceConfig.LP_MODEL_PATH + PredictionServiceConfig.GB_PREFIX + regressor_name + '-' + period + '.pkl'
    rf_path = PredictionServiceConfig.LP_MODEL_PATH + PredictionServiceConfig.RF_PREFIX + regressor_name + '-' + period + '.pkl'
    ada_path = PredictionServiceConfig.LP_MODEL_PATH + PredictionServiceConfig.ADA_PREFIX + regressor_name + '-' + period + '.pkl'

    gb_results = [0] * len(df_test)
    rf_results = [0] * len(df_test)
    ada_results = [0] * len(df_test)

    if os.path.isfile(gb_path):
        gb_regressor = pickle.load(open(gb_path, 'rb'))
        gb_results = gb_regressor.predict(X_test)

    if os.path.isfile(rf_path):
        rf_regressor = pickle.load(open(rf_path, 'rb'))
        rf_results = rf_regressor.predict(X_test)

    if os.path.isfile(ada_path):
        ada_regressor = pickle.load(open(ada_path, 'rb'))
        ada_results = ada_regressor.predict(X_test)

    df_pred_lp = pd.DataFrame(
        {'gb_pred': gb_results, 'rf_pred': rf_results, 'ada_pred': ada_results},
        index=df_test.index)

    for i, row in df_pred_lp.iterrows():
        df.at[i, 'gb_pred'] = row['gb_pred']
        df.at[i, 'rf_pred'] = row['rf_pred']
        df.at[i, 'ada_pred'] = row['ada_pred']

    return df


def predict_lp_use_regressor(df_list):
    df_list = df_list.drop_duplicates(subset=['_id'], keep='last')
    df_list.set_index('_id', inplace=True)
    df_list_reg = rename_community(df_list, 'Community')

    comm_list = list(set(df_list_reg[df_list_reg['Community'] != 'Other']['Community'].values))
    comm_arr_1y = get_comm_list('1year')
    comm_arr_5m = get_comm_list('5months')
    comm_arr_2m = get_comm_list('2months')

    for comm in comm_list:
        df_test = df_list_reg[df_list_reg['Community'] == comm]
        if len(df_test) == 0:
            continue
        X_test = mapper.regression_predict_mapper_proc(df_test)
        regressor_name = comm + '_regressor'
        if comm in comm_arr_2m:
            df_list = predict_lp_for_each_period(df_list, df_test, X_test, regressor_name, '2months')
        elif comm in comm_arr_5m:
            df_list = predict_lp_for_each_period(df_list, df_test, X_test, regressor_name, '5months')
        elif comm in comm_arr_1y:
            df_list = predict_lp_for_each_period(df_list, df_test, X_test, regressor_name, '1year')
    df_list.fillna(0, inplace=True)
    return df_list


def cal_diff(lp, pred):
    diff = abs(lp - pred) / float(lp)
    return diff


def generate_new_listing_file_after_lp_adjustment(df_list, df_house):
    df = predict_lp_use_regressor(df_list)
    df, convert_id = adjust_lp(df, df_house)
    df = df.reset_index()
    return df, convert_id


def test_adjustor():
    house_file = '/var/qindom/realmaster/csv_file/clean_house_data.csv'
    df_house = pd.read_csv(house_file)
    df_house = df_house[df_house['Cd'] < '2019-02-14']
    list_file = '/var/qindom/realmaster/csv_file/slice_23.csv'
    df_list = pd.read_csv(list_file)

    df, convert_id = generate_new_listing_file_after_lp_adjustment(df_list, df_house)
    print(len(convert_id))
    df_test = df_list[df_list['_id'].isin(convert_id)][['_id', 'Sp_dol', 'Lp_dol']]
    df_test.set_index('_id', inplace=True)
    df_pred = df[df['_id'].isin(convert_id)][['_id', 'Lp_dol']]
    df_pred.set_index('_id', inplace=True)
    df_result = pd.concat([df_pred, df_test], axis=1)
    print(df_result)
    df.to_csv(PredictionServiceConfig.LP_PATH + 'Listing#20190308_test.csv', index=None)
