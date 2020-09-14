from PredictionService.config import PredictionServiceConfig
from PredictionService.pairs_generator.pairs_generator import regenerate_whole_train_pairs, generate_predict_pairs, \
    multiprocess_train_pairs
from PredictionService.fm_model.fm_model import train_fm_model, predict_use_fm
from PredictionService.fm_model.results import convert_to_price, output_normal_distribution
from PredictionService.utils.utils import merge_train_pairs, remove_files_in_dir
import pandas as pd
from PredictionService.listing_price_corrector.listing_price_adjustor import generate_new_listing_file_after_lp_adjustment


# TODO: deal with exception, remove error files if exception happens
def retrain_process(num_slice, input_path=PredictionServiceConfig.SLICE_PATH,
                    output_path=PredictionServiceConfig.MODEL_PATH):
    remove_files_in_dir(PredictionServiceConfig.DATA_PATH, PredictionServiceConfig.CLEAN_HOUSE_DATA_NAME)
    remove_files_in_dir(PredictionServiceConfig.PAIR_PATH, PredictionServiceConfig.TRAIN_PAIRS_PREFIX + '*.csv')

    regenerate_whole_train_pairs(num_slice, input_path)
    train_set_path = merge_train_pairs(PredictionServiceConfig.PAIR_PATH, PredictionServiceConfig.TRAIN_SET)
    model_full_path = train_fm_model(train_set_path, output_path)

    remove_files_in_dir(PredictionServiceConfig.PAIR_PATH, PredictionServiceConfig.TRAIN_SET)
    remove_files_in_dir(PredictionServiceConfig.PAIR_PATH, '*.bin')
    return model_full_path


def train_process(start_slice, end_slice, input_path=PredictionServiceConfig.SLICE_PATH,
                  output_path=PredictionServiceConfig.MODEL_PATH):
    multiprocess_train_pairs(start_slice, end_slice, input_path, output_path=PredictionServiceConfig.PAIR_PATH)
    train_set_path = merge_train_pairs(end_slice, PredictionServiceConfig.PAIR_PATH, PredictionServiceConfig.TRAIN_SET)
    model_full_path = train_fm_model(train_set_path, output_path)
    remove_files_in_dir(PredictionServiceConfig.PAIR_PATH, PredictionServiceConfig.TRAIN_SET)
    remove_files_in_dir(PredictionServiceConfig.PAIR_PATH, '*.bin')
    return model_full_path


def predict_process(df_slice, df_pred, timestamp, model_path=PredictionServiceConfig.MODEL_PATH,
                    output_path=PredictionServiceConfig.RESULT_PATH):
    predict_pairs_full_path = generate_predict_pairs(df_slice, df_pred, timestamp, PredictionServiceConfig.PAIR_PATH)
    output_pre_file = predict_use_fm(predict_pairs_full_path, timestamp, model_path, output_path)

    pairs_tmp_file = PredictionServiceConfig.PAIR_PATH + PredictionServiceConfig.PAIRS_TMP_PREFIX + str(
        timestamp) + '.csv'

    df_pairs_result = convert_to_price(output_pre_file, pairs_tmp_file)
    final_result_file_path, df_fm = output_normal_distribution(df_pairs_result, timestamp, output_path)

    # save pairs result file to internal_result folder
    df_pairs_result.to_csv(PredictionServiceConfig.INTERNAL_RESULT_PATH + 'pair_results' + str(timestamp) + '.csv',
                           index=None)

    remove_files_in_dir(PredictionServiceConfig.PAIR_PATH, PredictionServiceConfig.PRED_PAIRS_PREFIX + '*.csv')
    remove_files_in_dir(PredictionServiceConfig.PAIR_PATH, PredictionServiceConfig.PAIRS_TMP_PREFIX + '*.csv')
    remove_files_in_dir(PredictionServiceConfig.PAIR_PATH, '*.bin')
    remove_files_in_dir(PredictionServiceConfig.RESULT_PATH, '*.txt')

    return df_fm


if __name__ == "__main__":
    predict_file = PredictionServiceConfig.DATA_PATH + 'Listing#20190114000044.csv'
    clean_house_data = PredictionServiceConfig.DATA_PATH + 'clean_house_data.csv'
    slice_19 = PredictionServiceConfig.SLICE_PATH + 'slice_19.csv'
    slice_18 = PredictionServiceConfig.SLICE_PATH + 'slice_18.csv'
    df_s19 = pd.read_csv(slice_19)
    df_s18 = pd.read_csv(slice_18)
    df_slice = pd.concat(([df_s18, df_s19]))
    df_data = pd.read_csv(clean_house_data)
    df_pred = pd.read_csv(predict_file)
    df_list, id = generate_new_listing_file_after_lp_adjustment(df_list=df_pred, df_house=df_data)
    predict_process(df_slice, df_list, 'from20190113_test')
