import pandas as pd
from PredictionService.add_features_manager import new_mapper
from datetime import datetime
from PredictionService.config import PredictionServiceConfig
from PredictionService.house_prediction_manager.prediction_flow import predict_process, train_process
from PredictionService.utils.utils import remove_files_in_dir


def slice_data(rawdata, interval=20):
    rawdata['Cd'] = pd.to_datetime(rawdata['Cd'])
    # TODO
    rawdata.sort_values(by=['Cd'], ascending=True, inplace=True)  # long long ago -> now
    start_date = datetime.strptime('2000-01-01', '%Y-%m-%d')
    cols = rawdata.columns.values.tolist()
    cur_df = pd.DataFrame(columns=cols)
    i = 0
    for index, row in rawdata.iterrows():
        cur_date = row['Cd']
        delta = cur_date - start_date
        if delta.days >= interval:
            if int(cur_df.shape[0]) > 0:
                i += 1
                cur_df.to_csv(PredictionServiceConfig.DATA_PATH + 'slice/slice_' + str(i) + '.csv', index=None)
            start_date = cur_date
            cur_df = pd.DataFrame(columns=cols)
        cur_df = pd.concat([cur_df, pd.DataFrame(row).transpose()], ignore_index=True)
    i += 1
    cur_df.to_csv(PredictionServiceConfig.DATA_PATH + 'slice/slice_' + str(i) + '.csv', index=None)
    return i


def retrain_process_for_new_features(house_train_file):
    new_mapper.initial_mappers(house_train_file)
    df_train_data = pd.read_csv(house_train_file)
    slice_data(df_train_data)
    train_process(1, 3)


def clear_environ():
    remove_files_in_dir(PredictionServiceConfig.SLICE_PATH, '*')
    remove_files_in_dir(PredictionServiceConfig.PAIR_PATH, '*')


if __name__ == "__main__":
    house_train_file = PredictionServiceConfig.DATA_PATH + 'house_data_feature1256+pic-train.csv'
    house_test_file = PredictionServiceConfig.DATA_PATH + 'house_data_feature1256+pic-test.csv'
    clear_environ()
    retrain_process_for_new_features(house_train_file)
    df_slice = pd.read_csv(house_train_file)
    df_pred = pd.read_csv(house_test_file)
    predict_process(df_slice, df_pred, 'num_pic')
