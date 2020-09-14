import numpy as np
import pandas as pd
from datetime import timedelta
import gc
import os
from multiprocessing import Pool
from PredictionService.pairs_generator import mapper
from PredictionService.config import PredictionServiceConfig
from PredictionService.config import constants
from PredictionService.utils.utils import combine_files


def regenerate_whole_train_pairs(num_slice, input_path=PredictionServiceConfig.SLICE_PATH):
    clean_data = combine_files(input_path)
    df_data = clean_data[clean_data['Cd'] >= '2017-12-01']
    df_data.to_csv(PredictionServiceConfig.CLEAN_HOUSE_DATA, index=None)
    mapper.initial_mappers(PredictionServiceConfig.CLEAN_HOUSE_DATA)
    multiprocess_train_pairs(1, num_slice)
    return


def generate_predict_pairs(train_data, target_data, timestamp, output_path=PredictionServiceConfig.PAIR_PATH):
    # year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
    # cur_date = str(year) + '-' + str(month) + '-' + str(day)
    # target_data['Cd'] = cur_date
    pred_prefix = PredictionServiceConfig.PRED_PAIRS_PREFIX
    generate_pair_dataset(target_data, train_data, timestamp, pred_prefix, output_path, constants.INTERVAL,
                          max_diff_ratio=0.08)
    pred_pair_full_path = output_path + pred_prefix + str(timestamp) + '.csv'
    return pred_pair_full_path


def read_slice_file_as_df(slice, slice_path, cols):
    file = slice_path + PredictionServiceConfig.SLICE_PREFIX + str(slice) + '.csv'
    if os.path.isfile(file):
        df = pd.read_csv(file)
    else:
        df = pd.DataFrame(columns=cols)
    return df


def get_train_data(slice, target_data, slice_path):
    cols = list(target_data.columns.values)

    train1 = read_slice_file_as_df(slice - 2, slice_path, cols)
    train2 = read_slice_file_as_df(slice - 1, slice_path, cols)
    train3 = read_slice_file_as_df(slice + 1, slice_path, cols)
    train4 = read_slice_file_as_df(slice + 2, slice_path, cols)

    train_data = pd.concat([train1, train2, target_data, train3, train4])
    return train_data


def generate_train_pairs(slice, input_path=PredictionServiceConfig.SLICE_PATH,
                         output_path=PredictionServiceConfig.PAIR_PATH):
    target_file = input_path + PredictionServiceConfig.SLICE_PREFIX + str(slice) + '.csv'
    if os.path.isfile(target_file):
        target_data = pd.read_csv(target_file)
    else:
        return
    train_data = get_train_data(slice, target_data, input_path)

    pair_file = output_path + PredictionServiceConfig.TRAIN_PAIRS_PREFIX + str(slice) + '.csv'
    if os.path.isfile(pair_file) and len(train_data) != 0:
        os.remove(pair_file)
    generate_pair_dataset(target_data, train_data, slice, PredictionServiceConfig.TRAIN_PAIRS_PREFIX, output_path)


def multiprocess_train_pairs(start_slice, end_slice, input_path=PredictionServiceConfig.SLICE_PATH,
                             output_path=PredictionServiceConfig.PAIR_PATH):
    slice_range = list(range(start_slice - 2, end_slice + 1))
    pool = Pool(processes=int(constants.CORE * 0.8))
    pool.map(generate_train_pairs, slice_range)
    pool.close()
    pool.join()


def generate_pair_dataset(df_target, df_train, process, prefix, pair_file_path, interval=constants.INTERVAL,
                          max_diff_ratio=constants.RATIO):
    target_copy = df_target[:]
    train_copy = df_train[:]

    train_copy['date_t'] = pd.to_datetime(train_copy['Cd'])
    train_copy.drop(['Cd'], axis=1, inplace=True)

    target_copy['date_p'] = pd.to_datetime(target_copy['Cd'])
    target_copy.drop(['Cd'], axis=1, inplace=True)

    target_copy.index = range(len(target_copy))

    target_copy.apply(generate_group, axis=1,
                      args=(train_copy, interval, process, prefix, pair_file_path, None, max_diff_ratio))
    del train_copy, target_copy
    gc.collect()


def generate_group(main_row, df, interval, process, prefix, pair_file_path, max_diff_value=None, max_diff_ratio=0.1):
    date = main_row['date_p']  # main_row is df_target
    if prefix == PredictionServiceConfig.TRAIN_PAIRS_PREFIX:
        value = main_row['Sp_dol']
    if prefix == PredictionServiceConfig.PRED_PAIRS_PREFIX:
        value = main_row['Lp_dol']

    start = date + timedelta(days=-interval)
    end = date + timedelta(days=interval)
    part = df[(df['date_t'] <= end) & (df['date_t'] >= start)]
    if len(part) == 0:
        return

    # Use dol to select rows
    if max_diff_value is not None:
        part['value'] = part['Sp_dol'] - value
        part = part[np.abs(part['value']) < max_diff_value]

    else:
        part['ratio'] = np.round((part['Sp_dol'] - value) / float(value), decimals=2)
        part = part[np.abs(part['ratio']) <= max_diff_ratio]
    if len(part) == 0:
        return

    part['diff'] = np.log1p(part['Sp_dol']) - np.log1p(value)

    if len(part) == 0:
        return

    frac = 0.8
    part = part.sample(frac=frac, random_state=42)
    main = pd.DataFrame(data=[main_row.values] * part.shape[0], columns=main_row.index)

    if prefix == PredictionServiceConfig.PRED_PAIRS_PREFIX:
        vali_file_path = pair_file_path + PredictionServiceConfig.PAIRS_TMP_PREFIX + str(process) + '.csv'
        part_v = part[['_id', 'Sp_dol']]
        part_v.rename(columns={'_id': 'pair_id', 'Sp_dol': 'pair_price'})
        part_v.index = range(len(part_v))
        main_v = main[['_id', 'date_p']]
        main_v.rename(columns={'_id': 'target_id'})
        main_v.index = range(len(main_v))
        df_vali = pd.concat([part_v, main_v], axis=1)
        df_vali.to_csv(vali_file_path, mode='a', header=False, index=False)
    part.drop(columns=['_id'], inplace=True)
    main.drop(columns=['_id'], inplace=True)

    part_ = mapper.mapper_proc_train(part)

    main_ = mapper.mapper_proc_target(main)
    diff = part['diff']

    append_to_file(part_, main_, diff, process, pair_file_path, prefix)


def append_to_file(part, main, diff, process, pair_file_path, prefix):
    pair_file = pair_file_path + prefix + str(process) + '.csv'

    with open(pair_file, 'a') as f:

        diff_ = diff.values
        num_rows = len(diff_)
        len_part = len(part.columns)
        len_main = len(main.columns)

        for i in range(num_rows):
            line = str(np.round(diff_[i], 4))
            each_part = part.iloc[i].values
            each_main = main.iloc[i].values

            for p in range(len_part):
                if each_part[p] != 0:
                    line += ' ' + "%d:%d:%.3f" % (0, p, each_part[p]) + ' '

            for m in range(len_main):
                if each_main[m] != 0:
                    line += ' ' + "%d:%d:%.3f" % (1, m, each_main[m]) + ' '

            line += '\n'
            f.write(line)

    del part, main, diff
    gc.collect()
    return pair_file

