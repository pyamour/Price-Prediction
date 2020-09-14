import pandas as pd
from PredictionService.config import PredictionServiceConfig
from functools import reduce


def generate_merged_results(cnn_file):
    origin_file = cnn_file + 'Predict#original.csv'
    feature256_file = cnn_file + 'Predict#feature256.csv'
    feature1000_file = cnn_file + 'Predict#feature1000.csv'
    feature1256_file = cnn_file + 'Predict#feature1256.csv'
    pic_file = cnn_file + 'Predict#num_pic.csv'
    sp_file = cnn_file + 'house_data_feature1256+pic-test.csv'

    df_origin = pd.read_csv(origin_file)[['_id', 'mean_price']]
    df_f256 = pd.read_csv(feature256_file)[['_id', 'mean_price']]
    df_f1000 = pd.read_csv(feature1000_file)[['_id', 'mean_price']]
    df_f1256 = pd.read_csv(feature1256_file)[['_id', "mean_price"]]
    df_pic = pd.read_csv(pic_file)[['_id', 'mean_price']]
    df_sp = pd.read_csv(sp_file)[['_id', 'Sp_dol']]

    df_list = [df_sp, df_origin, df_pic, df_f256, df_f1000, df_f1256]

    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['_id'], how='inner'), df_list)
    df_merged.columns = ['_id', 'Sp_dol', 'origin_price', 'pic_price', 'feature256', 'feature1000', 'feature1256']
    print(df_merged.columns)
    print(df_merged)
    df_merged.to_csv(cnn_file + 'merged_results.csv', index=False)


def cal_diff(true, pred):
    diff = abs(true - pred) / true
    return diff


def calculate_diff(cnn_file):
    merged_file = cnn_file + 'merged_results.csv'
    df_merge = pd.read_csv(merged_file)
    df_merge['origin_diff'] = cal_diff(df_merge['Sp_dol'], df_merge['origin_price'])
    df_merge['pic_diff'] = cal_diff(df_merge['Sp_dol'], df_merge['pic_price'])
    df_merge['feature256_diff'] = cal_diff(df_merge['Sp_dol'], df_merge['feature256'])
    df_merge['feature1000_diff'] = cal_diff(df_merge['Sp_dol'], df_merge['feature1000'])
    df_merge['feature1256_diff'] = cal_diff(df_merge['Sp_dol'], df_merge['feature1256'])

    prct50_origin = len(df_merge[df_merge['origin_diff'] <= 0.05]) / len(df_merge)
    prct30_origin = len(df_merge[df_merge['origin_diff'] <= 0.03]) / len(df_merge)
    prct50_pic = len(df_merge[df_merge['pic_diff'] <= 0.05]) / len(df_merge)
    prct30_pic = len(df_merge[df_merge['pic_diff'] <= 0.03]) / len(df_merge)
    prct50_f256 = len(df_merge[df_merge['feature256_diff'] <= 0.05]) / len(df_merge)
    prct30_f256 = len(df_merge[df_merge['feature256_diff'] <= 0.03]) / len(df_merge)
    prct50_f1000 = len(df_merge[df_merge['feature1000_diff'] <= 0.05]) / len(df_merge)
    prct30_f1000 = len(df_merge[df_merge['feature1000_diff'] <= 0.03]) / len(df_merge)
    prct50_f1256 = len(df_merge[df_merge['feature1256_diff'] <= 0.05]) / len(df_merge)
    prct30_f1256 = len(df_merge[df_merge['feature1256_diff'] <= 0.03]) / len(df_merge)

    print("**********")
    print("origin < 0.03:", prct30_origin)
    print("origin < 0.05:", prct50_origin)
    print("**********")
    print("pic < 0.03:", prct30_pic)
    print("pic < 0.05:", prct50_pic)
    print("**********")
    print("feature256 < 0.03:", prct30_f256)
    print("feature256 < 0.05:", prct50_f256)
    print("**********")
    print("feature1000 < 0.03:", prct30_f1000)
    print("feature1000 < 0.05:", prct50_f1000)
    print("**********")
    print("feature1256 < 0.03:", prct30_f1256)
    print("feature1256 < 0.05:", prct50_f1256)


if __name__ == "__main__":
    cnn_file = PredictionServiceConfig.DATA_PATH + 'cnn_features/'
    # generate_merged_results(cnn_file)
    calculate_diff(cnn_file)
