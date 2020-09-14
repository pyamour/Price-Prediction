import glob
import os
import time

import pandas

import boto3

from DataClean.config import s3config, DataCleanServiceConfig
from DataClean.data_cleaning.data_cleaning import clean_increment_data
from DataClean.utils.utils import remove_gz_suffix, remove_filename, remove_files_in_dir, \
    remove_gz_suffix_for_condo
from DataClean.validate.validation import validate


class DataUpdator:
    realmaster_bucket = s3config.real_price_prediction_bucket
    internal_realmaster_bucket = s3config.internal_real_price_prediction_bucket
    time_stamp = None
    raw_data_suffix = '.csv.gz'
    shard_bucket_prefix = 'shard/'
    slice_bucket_prefix = 'slice/'
    pred_bucket_prefix = 'pred/'
    sold_condo_bucket_prefix = 'condo/'
    delima = '#'
    time_window = 20

    def __init__(self):
        self.s3client = boto3.client('s3', aws_access_key_id=s3config.aws_access_key_id,
                                     aws_secret_access_key=s3config.aws_secret_access_key)

    def retrieve_file(self, prefix):
        print("Start to retrieve ", prefix, "file")
        response = self.s3client.list_objects(Bucket=self.realmaster_bucket, Prefix=prefix)
        if 'Contents' in response.keys():
            for file in response['Contents']:
                name = file['Key'].rsplit('/', 1)[0]
                full_name = DataCleanServiceConfig.FILE_LOCATION + name
                if name.endswith(self.raw_data_suffix):
                    self.s3client.download_file(self.realmaster_bucket, file['Key'], full_name)
                    print("GZ file : ", name)
                    validate_result = validate(DataCleanServiceConfig.FILE_LOCATION, name)
                    if validate_result:
                        print("Valid file: ", name)
                        try:
                            self.s3client.upload_file(full_name, self.internal_realmaster_bucket, name)
                            df_cd, df_hs = clean_increment_data(full_name, prefix)
                            if df_hs is not None:
                                if prefix is not 'Sold':
                                    # upload clean data to pred/
                                    self.s3client.upload_file(remove_gz_suffix(full_name),
                                                              self.internal_realmaster_bucket,
                                                              self.pred_bucket_prefix + remove_gz_suffix(name))

                                else:
                                    # upload increment house clean data to shard/
                                    self.s3client.upload_file(remove_gz_suffix(full_name),
                                                              self.internal_realmaster_bucket,
                                                              self.shard_bucket_prefix + remove_gz_suffix(name))
                                remove_filename(remove_gz_suffix(full_name))
                            if df_cd is not None:
                                if prefix is not 'Sold':
                                    # upload clean data to pred/
                                    self.s3client.upload_file(remove_gz_suffix_for_condo(full_name),
                                                              self.internal_realmaster_bucket,
                                                              self.pred_bucket_prefix + remove_gz_suffix_for_condo(
                                                                  name))
                                else:
                                    # upload increment condo clean data to condo/
                                    self.s3client.upload_file(remove_gz_suffix_for_condo(full_name),
                                                              self.internal_realmaster_bucket,
                                                              self.sold_condo_bucket_prefix + remove_gz_suffix_for_condo(
                                                                  name))

                                remove_filename(remove_gz_suffix_for_condo(full_name))
                            print("Delete s3 copy : ", file)
                            self.s3client.delete_object(Bucket=self.realmaster_bucket, Key=name)
                        except Exception:
                            self.s3client.upload_file(full_name, self.realmaster_bucket, 'Error' + self.delima + name)
                            self.s3client.delete_object(Bucket=self.realmaster_bucket, Key=name)

                    else:
                        self.s3client.upload_file(full_name, self.realmaster_bucket, 'Error' + self.delima + name)
                        self.s3client.delete_object(Bucket=self.realmaster_bucket, Key=name)
                remove_filename(full_name)
        else:
            print('No ', prefix, ' file')

    def check_and_combine_shards(self):
        print("Checking shards")
        response_shards = self.s3client.list_objects(Bucket=self.internal_realmaster_bucket,
                                                     Prefix=self.shard_bucket_prefix + 'Sold')
        min = -1
        max = -1
        if 'Contents' in response_shards.keys():
            for file in response_shards['Contents']:
                name = file['Key']
                date = name.rsplit(self.delima)[-1][:-4]
                if min is -1 or min > date:
                    min = date
                if max is -1 or date > max:
                    max = date
            time_elapsed = self._get_day_elapsed_from_str(min, max)
            print("Time elapsed: ", time_elapsed)
            if time_elapsed > self.time_window:
                for file in response_shards['Contents']:
                    name = file['Key']
                    if name.endswith('.csv'):
                        self.s3client.download_file(self.internal_realmaster_bucket, name,
                                                    (DataCleanServiceConfig.FILE_LOCATION + name))
                        self.s3client.delete_object(Bucket=self.internal_realmaster_bucket, Key=name)
                response_slice = self.s3client.list_objects(Bucket=self.internal_realmaster_bucket,
                                                            Prefix=self.slice_bucket_prefix)
                slice_max = 0
                for file in response_slice['Contents']:
                    slice_num = file['Key'].rsplit('_')
                    if len(slice_num) > 1:
                        if int(slice_num[-1][:-4]) > slice_max:
                            slice_max = int(slice_num[-1][:-4])
                slice_num = slice_max + 1
                slice_name = 'slice_' + str(slice_num)
                self._combine_files_in_one_dir_and_remove_files(DataCleanServiceConfig.SHARD_LOCATION,
                                                                slice_name)
                slice_full_name = DataCleanServiceConfig.SHARD_LOCATION + slice_name
                self.s3client.upload_file(slice_full_name, self.internal_realmaster_bucket,
                                          self.slice_bucket_prefix + slice_name + '.csv')
                remove_filename(slice_full_name)
                remove_files_in_dir(DataCleanServiceConfig.SHARD_LOCATION, '*')
        else:
            print("No shard file find")

    def _combine_files_in_one_dir_and_remove_files(self, dir, name):
        os.chdir(dir)
        results = pandas.DataFrame([])
        for counter, file in enumerate(glob.glob('*.csv')):
            namedf = pandas.read_csv(file)
            results = results.append(namedf)
            remove_filename(file)
        results.sort_values(by=['Cd'], ascending=True, inplace=True)
        results.drop_duplicates(subset='_id', keep='last', inplace=True)
        results.to_csv(DataCleanServiceConfig.SHARD_LOCATION + name, index=False)

    def _get_day_elapsed_from_str(self, start, end):
        year_min = int(start[0:4])
        month_min = int(start[4:6])
        day_min = int(start[6:8])
        year_max = int(end[0:4])
        month_max = int(end[4:6])
        day_max = int(end[6:8])
        return (year_max - year_min) * 365 + (month_max - month_min) * 30 + day_max - day_min


if __name__ == '__main__':
    data_updator = DataUpdator()
    while True:
        data_updator.retrieve_file('Sold')
        data_updator.retrieve_file('Listing')
        data_updator.check_and_combine_shards()
        time.sleep(60)
