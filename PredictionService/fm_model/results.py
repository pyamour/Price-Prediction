from PredictionService.config import PredictionServiceConfig
import pandas as pd
import math


def calculate_price(row):
    return row['pair_price'] / math.exp(float(row['pred_diff']))


def convert_to_price(output_pre, pairs_tmp):
    f = open(output_pre, "r")
    result = list(f.read().split('\n'))
    df_result = pd.read_csv(pairs_tmp, header=None)
    df_result.columns = ['pair_id', 'pair_price', 'target_id', 'date']
    df_result.index = range(len(df_result))
    try:
        df_result['pred_diff'] = pd.Series(result)
        df_result['pred_price'] = df_result.apply(calculate_price, axis=1)
    except Exception as e:
        raise e
    return df_result


def output_normal_distribution(df_result, timestamp, result_path):
    final_result_path = result_path + PredictionServiceConfig.RESULT_PREFIX + str(timestamp) + '_fm.csv'
    df_target = df_result.groupby(['target_id']).mean()
    df = pd.merge(df_result, df_target,
                  on=['target_id', 'target_id'])[['target_id', 'pred_price_x', 'pred_price_y']]
    df.rename(columns={'target_id': '_id', 'pred_price_x': 'pred_price', 'pred_price_y': 'median'}, inplace=True)
    df['diff'] = abs(df['pred_price'] - df['median']) / df['median']
    df = df[df['diff'] < 0.08]
    df_mean = df.groupby(['_id']).mean()['pred_price'].astype(int)
    df_std = df.groupby(['_id']).std()['pred_price']
    df_nd = pd.concat((df_mean.rename('mean_price'), df_std.rename('std')), axis=1)

    # if std == 0, give std a value
    df_na = df_nd[(df_nd['std'].isnull()) | (df_nd['std'] == 0.0)]
    df_na['std'] = df_na['mean_price'] * 0.04
    na_index = df_na.index.values
    for id in na_index:
        df_nd.loc[id, 'std'] = df_na.loc[id, 'std']
    df_nd['std'] = df_std['std'].astype(int)
    df_nd.to_csv(final_result_path)

    # save Predict# result file to internal_result folder
    internal_use_result_path = PredictionServiceConfig.INTERNAL_RESULT_PATH + PredictionServiceConfig.RESULT_PREFIX + str(
        timestamp) + '_fm.csv'
    df_nd.to_csv(internal_use_result_path)

    return final_result_path, df_nd


if __name__ == "__main__":
    output = '/var/csv_file/result/Predict#20190207004502_fm.csv'
    df_nd = pd.read_csv(output)

    print(df_nd.at[0,'std'])
    df_na = df_nd[(df_nd['std'].isnull()) | (df_nd['std'] == 0.0)]
    df_na['std'] = df_na['mean_price'] * 0.04
    na_index = df_na.index.values
    for id in na_index:
        df_nd.loc[id, 'std'] = df_na.loc[id, 'std']
    print(df_nd)

