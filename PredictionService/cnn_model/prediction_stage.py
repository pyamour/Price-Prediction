# ===================================================================================================
#
# ===date: 04-11-2019
#
# ===name: CNN, FM, Xgboost ensembled by DNN
#
# ===env: python3.7
#
# ===the following codes will do 3 jobs:
#      1. prepare dataframe from listing files
#      2. load predictions from CNN, FM, Xgboost...
#      3. load stacking DNN weights and predict and save results
#
# ===file description:
# 1. 'clean_house_data.csv' is our data and we use the last 7412 rows as testing data
# 2. 'columns_order_752' is a list to make sure that the order of new columns from new tesing data
#    are consistant with the previous
# 3. 'mapperCNN' is the mapper used for transforming new data
# 4. 'weights-CNN-FM-0411-20190305-4784.hdf5' is the stacking weights
#              |  |    |      |      |
#            modeles date ending accuracy
#
# ===HOW TO USE ???
#   df_CNN = pd.read_csv('CNN_pred.csv')
#   df_FM = pd.read_csv('FM_data.csv')
#   time_stamp = 2019
#
#   fds_stcking(df_CNN=df_CNN, df_FM=df_FM, time_stamp=time_stamp)
# ===================================================================================================

# ===================================================================================================
# 1st job:
#       <1>. acquire data from Listing file and inputs
#
#       <2>. process data
# ===================================================================================================
import sys
import pandas as pd
import numpy as np
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import MinMaxScaler as MM
from sklearn.preprocessing import LabelBinarizer as LB
import pickle
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras.layers import Convolution2D, MaxPooling2D, Flatten, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.regularizers import l1, l2
from datetime import datetime
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from PredictionService.config import PredictionServiceConfig
from PredictionService.config import constants
from PredictionService.cnn_model.cnn_mapper import process_predict_data
from PredictionService.cnn_model.dl_models import cnnModel, dnnModel
from PredictionService.cnn_model.utils import reshape_X


def cnn_predict(df_listing, timestamp):
    if (len(df_listing) == 0) or not timestamp:
        print('Error: check the input')
        sys.exit()
    df = process_predict_data(df_listing)
    # ---------------<6> 0~100 shrink 'Taxes'
    scale_min = 0
    scale_max = 100
    # the following values were previously acquired from the whole dataset(train + validate + test)
    # with new data comin, update them
    min_tax = 0.009950330853168083
    max_tax = 13.680836798801778
    # min_sp = 11.119897691565697
    min_sp = 11.026808713825016
    max_sp = 16.719675692787295
    print(df.columns)
    df['Taxes'] = (df['Taxes'] - min_tax) / (max_tax - min_tax) * (scale_max - scale_min) + scale_min

    # !!! make sure the order of cloumns are consistant with before
    with open(PredictionServiceConfig.CNN_COL_ORDER, 'rb') as fp:
        columns_order = pickle.load(fp)
    df = df[columns_order]

    arr = df.values
    splits = 0
    xTest, yTest = arr[splits:, 1:], arr[splits:, -1]
    # TODO: num_dim, num_zero
    num_dim, num_zero = constants.NUM_DIM, constants.NUM_ZERO_COL
    xTest = reshape_X(xTest, num_dim, num_zero)
    print('xTest and yTest shapes are: ', xTest.shape, yTest.shape)

    # ---------------<2> build the CNN model
    # set random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # ---------------<3> load weights
    weight_file = PredictionServiceConfig.CNN_WEIGHTS
    lr = 0.001
    model = cnnModel(0.0, 0.01, num_dim)
    model.summary()
    model.load_weights(weight_file)
    model.compile(loss='mae', optimizer=Adam(lr=lr, decay=0))

    # ---------------<4> predict and evaluate
    yPredict = model.predict(xTest)

    Sp_pred_list = []
    Lp_dol_list = []
    output = pd.DataFrame(columns=['_id', 'Lp_dol', 'CNN_pred'])
    # recover the data and save into lists
    for i in range(len(yPredict)):
        Sp_pred = int(np.expm1((yPredict[i][0] - scale_min) / (scale_max - scale_min) * (max_sp - min_sp) + min_sp))
        Sp_pred_list.append(Sp_pred - 1)
        Lp_dol = int(np.expm1(arr[i, 6]) - 1)
        Lp_dol_list.append(Lp_dol)

    output['_id'] = arr[:, 0]
    output['Lp_dol'] = Lp_dol_list
    output['CNN_pred'] = Sp_pred_list

    output.to_csv(PredictionServiceConfig.INTERNAL_RESULT_PATH + 'Predict#' + str(timestamp) + '_cnn.csv', index=False)
    print('prediction saved to ' + PredictionServiceConfig.INTERNAL_RESULT_PATH + 'Predict#' + str(
        timestamp) + '_cnn.csv')
    return output


def prediction_stack(**kwargs):
    print('Stacking is running lol lol... ...')
    print('Data preparing... ...')

    # import predictions from CNN, FM, Xgboost... and get the time_stamp val
    for key, val in kwargs.items():
        if key == 'df_CNN':
            df_CNN = val
        if key == 'df_FM':
            df_FM = val
        if key == 'df_Xgboost':
            df_Xgboost = val
        if key == 'timestamp':
            timestamp = val
        if key == 'df_listing':
            df_listing = val

    df = process_predict_data(df_listing)
    print(df.head(10))

    # 0~100 shrink 'Taxes'
    scale_min = 0
    scale_max = 100

    # the following values were previously acquired from the whole dataset(train + validate + test)
    # with new data coming, update them
    # TODO: Keep it. Don't need to change.
    min_tax = 0.009950330853168083
    max_tax = 13.680836798801778
    min_sp = 11.026808713825016
    # min_sp = 11.119897691565697
    max_sp = 16.719675692787295
    df['Taxes'] = (df['Taxes'] - min_tax) / (max_tax - min_tax) * (scale_max - scale_min) + scale_min

    # ===================================================================================================
    # 2nd job:
    #       <1> load predictions from base mdoels (CNN, Xgboost, FM...)
    #
    #       <2> merge
    # ===================================================================================================
    # test whether using GPUs or not
    print(K.tensorflow_backend._get_available_gpus())

    # allocate stacking data from here :)
    df_stack = df

    print(df.head(10))
    print(df_stack.head(10))
    # process and merge
    print(df_CNN['CNN_pred'])
    # CNN_pred calculate log and do scaler
    df_CNN["CNN_pred"] = np.log1p((df_CNN["CNN_pred"].astype(float)))
    df_CNN['CNN_pred'] = (df_CNN['CNN_pred'] - min_sp) / (max_sp - min_sp) * (scale_max - scale_min) + scale_min
    print("**********************************************")
    print(df_CNN.head(10))

    for ele in df_CNN.columns.tolist():
        if ele not in ['_id', 'CNN_pred']:
            df_CNN = df_CNN.drop([ele], axis=1)

    print(len(set(df_CNN['_id'].tolist()) & set(df_stack['_id'].tolist())))

    if len(df_CNN) != 0:
        df_stack = pd.merge(df_stack, df_CNN, on='_id')

    print(df_stack.head(10))

    # reorder

    with open(PredictionServiceConfig.STACK_COL_ORDER, 'rb') as fp:
        stack_columns_order = pickle.load(fp)
    df_stack = df_stack[stack_columns_order]
    # get Lp_dol and -id locations
    print('index of Lp_dol is: ', df_stack.columns.get_loc("Lp_dol"))
    print('index of _id is: ', df_stack.columns.get_loc("_id"))

    # ===================================================================================================
    # 3rd job:
    #       <1> build DNN model
    #
    #       <2> load stacking model weights
    #
    #       <3> predict and save results
    # ===================================================================================================
    # build DNN model and split test data
    # split data into X and Y
    stack_arr = df_stack.values
    print(df_stack.head(10))

    totalNum, totalFeature = stack_arr.shape
    splits = 0
    print('...', totalNum, totalFeature)
    xTest, yTest = stack_arr[splits:, 1:], stack_arr[splits:, -1]
    print(xTest[0])
    # load stacking weights and predict
    file_stacking = PredictionServiceConfig.SELF_STACKING_WEIGHTS
    lr = 0.001
    model_stacking = dnnModel()
    model_stacking.summary()
    model_stacking.load_weights(file_stacking)

    # predict and evaluate
    yPredict_stacking = model_stacking.predict(xTest)

    # recover the data and save into dataframe
    Sp_pred_list = []
    output_stacking = pd.DataFrame(columns=['_id', 'mean_price', 'std'])
    for i in range(len(stack_arr)):
        # TODO
        Sp_pred = int(
            np.expm1((yPredict_stacking[i][0] - scale_min) / (scale_max - scale_min) * (max_sp - min_sp) + min_sp))
        Sp_pred_list.append(Sp_pred)
    print(Sp_pred_list)
    output_stacking['_id'] = stack_arr[:, 0]
    output_stacking['mean_price'] = Sp_pred_list
    output_stacking['std'] = (output_stacking['mean_price'] * 0.04).astype(int)

    # show some results
    print(output_stacking.head(20))
    output_stacking.to_csv(PredictionServiceConfig.RESULT_PATH + 'Predict#' + str(timestamp) + '_house.csv',
                           index=False)
    return output_stacking


if __name__ == "__main__":
    df_listing = pd.read_csv(PredictionServiceConfig.DATA_PATH + 'sold522_cnn.csv')
    df_listing.drop_duplicates(inplace=True)
    tamp = '201905312333'
    df_CNN = cnn_predict(df_listing=df_listing, timestamp=tamp)
    prediction_stack(df_CNN=df_CNN, df_listing=df_listing, timestamp=tamp)
