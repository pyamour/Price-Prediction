import os
import pandas as pd
import glob
import random
from PredictionService.config import PredictionServiceConfig
from PredictionService.config import constants
import re


def merge_train_pairs(end_slice, pairs_file_path=PredictionServiceConfig.PAIR_PATH,
                      merged_file_name=PredictionServiceConfig.TRAIN_SET):
    if end_slice > 20:
        start_slice = end_slice - 20
    else:
        start_slice = 1
    output_file = pairs_file_path + merged_file_name

    n = constants.FIRST_PAIR_SAMPLE_NUM
    pair_num_list = list(range(end_slice, start_slice - 1, -1))
    print(pair_num_list)
    for suffix_num in pair_num_list:
        print(suffix_num)
        pair_with_suffix_file_path = pairs_file_path + PredictionServiceConfig.TRAIN_PAIRS_PREFIX + str(
            suffix_num) + '.csv'
        if check_local_file_exist(pair_with_suffix_file_path):

            with open(str(pair_with_suffix_file_path), mode='r') as fr:
                lines = fr.readlines()
                random.shuffle(lines)
                head = lines[:int(n)]

            with open(str(output_file), mode='a') as fw:
                for line in head:
                    fw.write(line)
            n = int(n / 1.142857)
    return str(output_file)


def combine_files(dir):
    os.chdir(dir)
    results = pd.DataFrame([])
    for counter, file in enumerate(glob.glob('*.csv')):
        namedf = pd.read_csv(file)
        results = pd.concat([namedf, results])

    results.drop_duplicates(inplace=True)

    return results


def remove_files_in_dir(file_path, expression):
    file_list = glob.glob(os.path.join(file_path, expression))
    for file in file_list:
        os.remove(file)

    return


def check_local_file_exist(fname):
    return os.path.isfile(fname)


def remove_local_file(fname):
    exist = os.path.exists(fname)
    if exist:
        os.remove(fname)


def extract_time_stamp(filename):
    numbers = re.findall('\d+', filename)
    if len(numbers) == 0:
        return numbers
    else:
        numbers = map(int, numbers)
        return max(numbers)


if __name__ == "__main__":
    merge_train_pairs(5)
