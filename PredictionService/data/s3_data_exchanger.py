import glob
import os

import boto3
import pandas

from PredictionService.config import s3config, PredictionServiceConfig
from PredictionService.config.PredictionServiceConfig import PRED_DATA_FOLDER
from PredictionService.utils.utils import remove_local_file, check_local_file_exist, remove_files_in_dir, extract_time_stamp
from PredictionService.config.s3config import real_price_prediction_bucket, internal_price_prediction_bucket

class S3DataExchanger:
    realmaster_bucket = s3config.real_master_bucket
    internal_realmaster_bucket = s3config.internal_real_master_bucket
    shard_bucket_prefix = 'shard/'
    slice_bucket_prefix = 'slice/'
    pair_bucket_prefix = 'pair/'
    sold_condo_bucket_prefix = 'condo/'

    def __init__(self):
        self.s3client = boto3.client('s3', aws_access_key_id=s3config.aws_access_key_id,
                                     aws_secret_access_key=s3config.aws_secret_access_key)

    def fetch_slice_data(self):
        self._fetch_from_bucket_to_dir(self.internal_realmaster_bucket, self.slice_bucket_prefix,
                                       PredictionServiceConfig.DATA_PATH)

    def fetch_pair_data(self):
        self._fetch_from_bucket_to_dir(self.internal_realmaster_bucket, self.pair_bucket_prefix,
                                       PredictionServiceConfig.DATA_PATH)

    def fetch_condo_data(self):
        self._fetch_from_bucket_to_dir(self.internal_realmaster_bucket, self.sold_condo_bucket_prefix,
                                       PredictionServiceConfig.DATA_PATH)

    def fetch_predict_data(self):
        self._fetch_from_bucket_to_dir(self.internal_realmaster_bucket, 'pred/Listing',
                                       PredictionServiceConfig.DATA_PATH)
        df_cd = []
        df_hs = []
        os.chdir(PredictionServiceConfig.DATA_PATH + 'pred/')
        for counter, file in enumerate(glob.glob('*_condo.csv')):
            cd_namedf = pandas.read_csv(file)
            timestamp = extract_time_stamp(file)
            df_cd.append([cd_namedf, str(timestamp)])
        for counter, file in enumerate(set(glob.glob('*.csv')) - set(glob.glob('*condo*'))):
            hs_namedf = pandas.read_csv(file)
            if len(hs_namedf) == 0:
                continue
            timestamp = extract_time_stamp(file)
            df_hs.append([hs_namedf, str(timestamp)])
        return df_cd, df_hs

    def remove_pred_folder_on_success_pred(self, key):
        self.s3client.delete_object(Bucket=self.internal_realmaster_bucket, Key=key)

    def fetch_shard_data(self):
        remove_files_in_dir(PRED_DATA_FOLDER, '*')
        self._fetch_from_bucket_to_dir(self.internal_realmaster_bucket, self.shard_bucket_prefix,
                                       PredictionServiceConfig.DATA_PATH)

    def _fetch_from_bucket_to_dir(self, bucket_name, prefix, dest):
        response = self.s3client.list_objects(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' in response.keys():
            for file in response['Contents']:
                fname = file['Key']
                if len(fname) == len(prefix):
                    continue
                download_full_path = dest + file['Key']
                if not check_local_file_exist(download_full_path):
                    self.s3client.download_file(bucket_name, file['Key'], download_full_path)

    def push_local_dir_to_s3_bucket(self, local_dir_full_path, s3_folder, bucket=internal_realmaster_bucket):
        for filename in os.listdir(local_dir_full_path):
            file_full_path = local_dir_full_path + filename
            self.s3client.upload_file(file_full_path, bucket, s3_folder + filename)


if __name__ == '__main__':
    s = S3DataExchanger()

