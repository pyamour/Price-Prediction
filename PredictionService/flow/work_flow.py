import glob
import os
import time
import pandas
from datetime import timedelta
from PredictionService.utils.utils import check_local_file_exist, remove_files_in_dir, combine_files, \
    extract_time_stamp
from PredictionService.config import PredictionServiceConfig
from PredictionService.config.s3config import real_price_prediction_bucket, internal_price_prediction_bucket, \
    price_prediction_results_bucket
from PredictionService.data.s3_data_exchanger import S3DataExchanger
from PredictionService.house_prediction_manager.prediction_flow import train_process
from PredictionService.listing_price_corrector.listing_price_regressors import multiprocess_train_regressors
from PredictionService.listing_price_corrector.lp_mappers import initial_mappers
from PredictionService.condo_prediction_manager.condo_price_prediction import predict_condo_price
from PredictionService.condo_prediction_manager.condo_index_files_preparation import init_condo_index_files
from PredictionService.cnn_model.prediction_stage import cnn_predict, prediction_stack
import gc


class WorkFlow:
    def __init__(self):
        self.s3_exchanger = S3DataExchanger()

    def training_flow(self):
        print("try to train")
        self.s3_exchanger.fetch_pair_data()
        self.s3_exchanger.fetch_slice_data()
        print("fetched pair slice from s3")
        gc.collect()
        try:
            start, end = WorkFlow._get_regenerate_pair_range(PredictionServiceConfig.PAIR_PATH,
                                                             PredictionServiceConfig.SLICE_PATH)
        except Exception as e:
            raise e
        print("Start END ： ", start, ' ', end)
        if end >= start:
            train_process(start, end, input_path=PredictionServiceConfig.SLICE_PATH,
                          output_path=PredictionServiceConfig.MODEL_PATH)

            self.s3_exchanger.push_local_dir_to_s3_bucket(PredictionServiceConfig.PAIR_PATH, 'pair/',
                                                          bucket=internal_price_prediction_bucket)
            self.train_regression_flow()
            self.condo_training_flow()
            return None

    def predict_flow(self):

        WorkFlow._merge_results_with_same_timestamp_in_dir(PredictionServiceConfig.RESULT_PATH)

        print("Starting try to pred")
        self.s3_exchanger.fetch_shard_data()
        print("Fetched shard file")
        df_pred_cd, df_pred_hs = self.s3_exchanger.fetch_predict_data()
        cd_timestamp = []
        hs_timestamp = []
        if df_pred_cd is not None:
            for pred in df_pred_cd:
                cd_timestamp.append(pred[1])
                predict_condo_price(pred[0], pred[1])
                self.s3_exchanger.remove_pred_folder_on_success_pred('pred/Listing#' + pred[1] + '_condo.csv')
        if df_pred_hs is not None and len(df_pred_hs) is not 0:
            print("fetched pred file : ", len(df_pred_hs))

            for pred in df_pred_hs:
                hs_timestamp.append(pred[1])
                df_cnn = cnn_predict(pred[0], pred[1])
                print(df_cnn)
                prediction_stack(df_CNN=df_cnn, df_listing=pred[0], time_stamp=pred[1])
                # FM model
                # print("generate new listing files: ", pred[1])
                # pred[0], convert_id = generate_new_listing_file_after_lp_adjustment(pred[0], df_sold)
                # print("Pred : file : ", pred[1])
                # df_fm = predict_process(df_slice, pred[0], pred[1])
                # print(df_fm)

                self.s3_exchanger.remove_pred_folder_on_success_pred('pred/Listing#' + pred[1] + '.csv')

        WorkFlow._merge_results_with_same_timestamp(cd_timestamp, hs_timestamp, PredictionServiceConfig.RESULT_PATH)
        self.s3_exchanger.push_local_dir_to_s3_bucket(PredictionServiceConfig.RESULT_PATH, '',
                                                      bucket=real_price_prediction_bucket)
        self.s3_exchanger.push_local_dir_to_s3_bucket(PredictionServiceConfig.RESULT_PATH, '',
                                                      bucket=price_prediction_results_bucket)
        remove_files_in_dir(PredictionServiceConfig.RESULT_PATH, '*')

    def train_regression_flow(self):
        # self.s3_exchanger.fetch_slice_data()
        # self.s3_exchanger.fetch_shard_data()
        df_sold = WorkFlow._combine_all_shard_and_slice_data_for_regression(PredictionServiceConfig.SLICE_PATH,
                                                                            PredictionServiceConfig.SHARD_PATH)
        print("try to init lp_mappers")
        initial_mappers(df_sold)
        print("try to train regressors")
        multiprocess_train_regressors(df_sold)
        return None

    def condo_training_flow(self):
        self.s3_exchanger.fetch_condo_data()
        df_condo = WorkFlow._combine_condo_pieces_for_quantile_file(PredictionServiceConfig.CONDO_PATH)
        init_condo_index_files(df_condo)

        return

    @staticmethod
    def _get_regenerate_pair_range(pair_dir, slice_dir):
        pair_suffix_max = WorkFlow._get_max_suffix_in_dir(pair_dir, prefix='train')
        if pair_suffix_max < 0:
            pair_suffix_max = 0
        slice_suffix_max = WorkFlow._get_max_suffix_in_dir(slice_dir, prefix='slice')
        if slice_suffix_max < 0:
            raise Exception('Slice dir has no slices')

        diff = slice_suffix_max - pair_suffix_max
        if diff > 0:
            return pair_suffix_max + 1, slice_suffix_max
        elif diff is 0:
            return pair_suffix_max + 1, slice_suffix_max
        else:
            raise Exception('More pairs than slices')

    @staticmethod
    def _get_max_suffix_in_dir(path, prefix='slice'):
        files = os.listdir(path)
        max = -1
        for name in files:
            if name.startswith(prefix):
                try:
                    temp_suffix = int(name.split('_')[-1].split('.')[0])

                except:
                    continue
                if temp_suffix > max:
                    max = temp_suffix
        return max

    @staticmethod
    def _combine_shard_and_slice_data_for_pairing(slice_path, shard_path):
        os.chdir(shard_path)
        results = pandas.DataFrame([])
        for counter, file in enumerate(glob.glob('*.csv')):
            namedf = pandas.read_csv(file)
            results = pandas.concat([namedf, results])
        slice_max = WorkFlow._get_max_suffix_in_dir(slice_path, prefix='slice')
        first_csv = slice_path + '/slice_' + str(slice_max) + '.csv'
        second_csv = slice_path + '/slice_' + str(slice_max - 1) + '.csv'

        results = WorkFlow._merge_file_with_df_if_exist(results, first_csv)
        results = WorkFlow._merge_file_with_df_if_exist(results, second_csv)
        return results

    @staticmethod
    def _combine_all_shard_and_slice_data_for_regression(slice_path, shard_path):
        df_slice = combine_files(slice_path)
        df_shard = combine_files(shard_path)
        df = pandas.concat([df_slice, df_shard])
        return df

    @staticmethod
    def _combine_all_shard_and_slice_data_for_lp_adj(slice_path, shard_path):
        df_slice = pandas.DataFrame([])
        max_slice = WorkFlow._get_max_suffix_in_dir(slice_path, prefix='slice')
        if max_slice > 8:
            start_slice = max_slice - 8
        else:
            start_slice = 1
        for slice_num in range(start_slice, max_slice + 1):
            slice_file = slice_path + '/slice_' + str(slice_num) + '.csv'
            df_slice = WorkFlow._merge_file_with_df_if_exist(df_slice, slice_file)
        df_shard = combine_files(shard_path)
        df = pandas.concat([df_slice, df_shard])
        return df

    @staticmethod
    def _combine_condo_pieces_for_quantile_file(condo_path):
        df_condo = combine_files(condo_path)
        df_condo['Cd'] = pandas.to_datetime(df_condo['Cd'])
        df_condo.sort_values(by=['Cd'], ascending=False, inplace=True)
        df_condo.drop_duplicates(keep='first', inplace=True)
        df_condo.index = range(len(df_condo))
        date_now = df_condo.at[0, 'Cd']
        start_date = date_now + timedelta(days=-360)
        df_condo = df_condo[df_condo['Cd'] >= start_date]
        return df_condo

    @staticmethod
    def _merge_file_with_df_if_exist(df, file):
        exists = os.path.isfile(file)
        if exists:
            temp_df = pandas.read_csv(file)
            df = pandas.concat([temp_df, df])
        return df

    @staticmethod
    def _merge_results_with_same_timestamp(cd_stamp, hs_stamp, result_path):
        timestamps = list(set(cd_stamp + hs_stamp))
        os.chdir(result_path)
        for stamp in timestamps:
            df_result = pandas.DataFrame([])
            for counter, file in enumerate(glob.glob('*' + str(stamp) + "*")):
                df = pandas.read_csv(file)
                df_result = pandas.concat([df, df_result])
            df_result.to_csv(result_path + PredictionServiceConfig.RESULT_PREFIX + str(stamp) + '.csv', index=False)
            condo_result_file = result_path + PredictionServiceConfig.RESULT_PREFIX + str(stamp) + '_condo.csv'
            house_result_file = result_path + PredictionServiceConfig.RESULT_PREFIX + str(stamp) + '_house.csv'

            if check_local_file_exist(condo_result_file):
                os.remove(condo_result_file)
            if check_local_file_exist(house_result_file):
                os.remove(house_result_file)

    @staticmethod
    def _merge_results_with_same_timestamp_in_dir(result_path):
        file_name_list = os.listdir(result_path)
        timestamps = []
        if len(file_name_list) == 0:
            return None
        for file_name in file_name_list:
            timestamp = extract_time_stamp(file_name)
            timestamps.append(str(timestamp))
        timestamps = list(set(timestamps))
        for stamp in timestamps:
            timestamp_file_list = [item for item in file_name_list if stamp in item]
            num_file = len(timestamp_file_list)
            if (num_file < 3) and (num_file > 0):
                df_result = pandas.DataFrame([])
                for file in timestamp_file_list:
                    df = pandas.read_csv(result_path + file)
                    df_result = pandas.concat([df, df_result])
                df_result.to_csv(result_path + PredictionServiceConfig.RESULT_PREFIX + str(stamp) + '.csv', index=False)
                condo_result_file = result_path + PredictionServiceConfig.RESULT_PREFIX + str(stamp) + '_condo.csv'
                house_result_file = result_path + PredictionServiceConfig.RESULT_PREFIX + str(stamp) + '_house.csv'
                if check_local_file_exist(condo_result_file):
                    os.remove(condo_result_file)
                if check_local_file_exist(house_result_file):
                    os.remove(house_result_file)
            else:
                for file_name in timestamp_file_list:
                    os.remove(result_path + file_name)


if __name__ == "__main__":
    wf = WorkFlow()
    while True:
        # wf.training_flow()
        gc.collect()
        wf.predict_flow()
        time.sleep(60)
