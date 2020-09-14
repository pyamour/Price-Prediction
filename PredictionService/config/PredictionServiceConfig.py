DATA_PATH = '/var/csv_file/'
FILE_LOCATION = '/var/csv_file/download/'

PAIR_PATH = '/var/csv_file/pair/'
SLICE_PATH = '/var/csv_file/slice/'
SHARD_PATH = '/var/csv_file/shard/'
MODEL_PATH = '/var/csv_file/model/'
RESULT_PATH = '/var/csv_file/result/'
INTERNAL_RESULT_PATH = '/var/csv_file/internal_result/'
CONDO_PATH = '/var/csv_file/condo/'
LP_PATH = '/var/csv_file/lp_corrector/'
LP_MODEL_PATH = LP_PATH + 'reg_model/'
CNN_PATH = '/var/csv_file/cnn/'
SOLD_PATH = '/var/csv_file/sold/'

# TODO
# previous weights
# CNN_WEIGHTS = CNN_PATH + 'weights-selfstacking-p1.hdf5'
# SELF_STACKING_WEIGHTS = CNN_PATH + 'weights-selfstacking-p2.hdf5'

# new weights
# CNN_WEIGHTS = CNN_PATH + 'weights-cnn-20190522-1439.hdf5'
# SELF_STACKING_WEIGHTS = CNN_PATH + 'weights-selfstacking-dnn-20190522-1439.hdf5'

# test weights
CNN_WEIGHTS = CNN_PATH + 'weights-cnn-20190531-1743.hdf5'
SELF_STACKING_WEIGHTS = CNN_PATH + 'weights-selfstacking-dnn-20190531-1743.hdf5'
#
CNN_MAPPER = CNN_PATH + 'mapperCNN'
CNN_COL_ORDER = CNN_PATH + 'cnn_columns_order'
STACK_COL_ORDER = CNN_PATH + 'stack_columns_order'

PRED_DATA_FOLDER = DATA_PATH + 'pred/'
CLEAN_HOUSE_DATA_NAME = 'clean_house_data.csv'
CLEAN_HOUSE_DATA = DATA_PATH + CLEAN_HOUSE_DATA_NAME

MAPPER_PATH = '/var/csv_file/mapper/'
PICKLE_FM_TRAIN = MAPPER_PATH + 'mapper_train.pkl'
PICKLE_FM_TARGET = MAPPER_PATH + 'mapper_target.pkl'
XGB_MAPPER = MAPPER_PATH + 'xgb_mapper.pkl'

PICKLE_REG_TRAIN = DATA_PATH + 'train_regression_mapper.pkl'
PICKLE_REG_PREDICT = DATA_PATH + 'predict_regression_mapper.pkl'

SLICE_PREFIX = 'slice_'
TRAIN_PAIRS_PREFIX = 'train_pairs_'
PRED_PAIRS_PREFIX = 'pred_pairs_'
PAIRS_TMP_PREFIX = 'pairs_tmp_'
OUTPUT_PRE_PREFIX = 'output_'
RESULT_PREFIX = 'Predict#'

GB_PREFIX = 'gb-'
ADA_PREFIX = 'ada-'
SHM_PREFIX = 'shm-'
RF_PREFIX = 'rf-'
COMM_COUNT_PREFIX = 'comm_count_'

TRAIN_SET = 'training_set.csv'
MODEL_NAME = 'ffm_model.out'

CONDO_QUANTILE_FILE = DATA_PATH + 'condo_quantile_file.csv'
FULL_NEIGHBORS_FILE = DATA_PATH + 'full_neighbors.txt'
ALS_NEIGHBORS_FILE = DATA_PATH + 'als_neighbors.txt'
PCA_NEIGHBORS_FILE = DATA_PATH + 'pca_neighbors.txt'
IMPLICIT_ALS_NEIGHBORS_FILE = DATA_PATH + 'implicit_als_neighbors.txt'
NMF_NEIGHBORS_FILE = DATA_PATH + 'nmf_neighbors.txt'
