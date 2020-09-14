import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import tensorflow
import random
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import ensemble
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.regularizers import l1, l2
from PredictionService.cnn_model.show_accuracy import Show_Accuracy
from PredictionService.config import PredictionServiceConfig, s3config
from PredictionService.config import constants
from PredictionService.cnn_model.cnn_mapper import process_train_data
from PredictionService.cnn_model.dl_models import cnnModel, dnnModel
from PredictionService.cnn_model.utils import reshape_X, calculate_input_dims_and_zero_cols
from DataClean.data_cleaning.data_cleaning import clean_whole_data
import os
import boto3
from PredictionService.utils.utils import check_local_file_exist
import glob


class TrainingStage:

    def __init__(self):
        # name cnn weights and dnn weights with nowtime
        self.s3client = boto3.client('s3', aws_access_key_id=s3config.aws_access_key_id,
                                     aws_secret_access_key=s3config.aws_secret_access_key)
        self.datalake_bucket = s3config.datalake_qindom_bucket
        self.sold_bucket_prefix = 'realestate/Online-Results/internalrealmaster/sold/'
        self.nowtime = datetime.now().strftime("%Y%m%d-%H%M")
        # TODO: for test use
        # self.df_clean_data = pd.read_csv(clean_house_data_file)
        # self.df_clean_data = pd.read_csv(RealMasterServiceConfig.DATA_PATH + 'clean_data_20190605.csv')
        return

    def training_flow(self, from_date=None, to_date=None):
        # ------------------------------------------------------------------------------------------------------------
        # =====Step 1: fetch and clean data
        df_clean_data = self.get_training_data(from_date=from_date, to_date=to_date)

        # ------------------------------------------------------------------------------------------------------------
        # =====Step 2: convert data into cnn-read format
        df = process_train_data(df_clean_data, timenow=self.nowtime)
        # df = pd.read_csv(self.ml_test_file, index_col=0)
        # Scale Taxes & Sp_dol
        scaler = MinMaxScaler(feature_range=(0, 100))
        for ele in ['Taxes', 'Sp_dol']:
            df[[ele]] = scaler.fit_transform(df[[ele]])

        # ------------------------------------------------------------------------------------------------------------
        # =====Step 3: split data into cnn-train and cnn-test, split cnn-test into dnn-train and dnn-test
        # cnn and stacking data distribution ratio
        # num_stack = constants.NUM_STACK
        num_stack = int(len(df) * 0.1)
        df_stack = df.tail(num_stack)
        cnn_split = len(df) - num_stack
        dnn_split = int(num_stack * 0.9)

        # ------------------------------------------------------------------------------------------------------------
        # =====Step 4: train cnn model
        cnn_X, cnn_y, cnn_xTest, cnn_yTest = self._split_data(df, splits=cnn_split)
        print(cnn_X)
        num_dim, num_zero = calculate_input_dims_and_zero_cols(len(cnn_X[0]))
        X = reshape_X(cnn_X, num_dim, num_zero)
        cnn_xTest = reshape_X(cnn_xTest, num_dim, num_zero)
        cnn_xTrain, _, cnn_yTrain, _ = train_test_split(X, cnn_y, test_size=0, random_state=42, shuffle=False)

        # set random seed for reproducibility
        # seed = 7
        # np.random.seed(seed)

        cnn_sp_pred_list, cnn_sp_true_list, cnn_yPredcit = self.build_cnn_model(cnn_xTrain, cnn_yTrain, cnn_xTest,
                                                                                cnn_yTest, scaler, num_dim)
        self._show_accuracy(df.values, cnn_split, cnn_sp_pred_list, cnn_sp_true_list)

        # ------------------------------------------------------------------------------------------------------------
        # =====Step5: train dnn stacking model
        df_stack['CNN_pred'] = cnn_yPredcit
        df_stack = self._save_dnn_column_order(df_stack)

        dnn_X, dnn_y, dnn_xTest, dnn_yTest = self._split_data(df_stack, splits=dnn_split)
        dnn_xTrain, _, dnn_yTrain, _ = train_test_split(dnn_X, dnn_y, test_size=0, random_state=42, shuffle=False)
        stack_sp_pred_list, stack_sp_true_list = self.build_dnn_model(dnn_xTrain, dnn_yTrain, dnn_xTest, dnn_yTest,
                                                                      scaler)
        self._show_accuracy(df_stack.values, dnn_split, stack_sp_pred_list, stack_sp_true_list)
        return

    def build_cnn_model(self, xTrain, yTrain, xTest, yTest, scaler, num_dim):

        # traing by Adam and loss func is mae
        model = cnnModel(0.0, 0.01, num_dim)
        model.summary()
        lr = 0.001
        model.compile(loss='mae', optimizer=Adam(lr=lr, decay=0))
        # set checkpoint to save weights with the MIN mae
        file = PredictionServiceConfig.CNN_PATH + "weights-cnn-" + str(self.nowtime) + ".hdf5"
        checkpoint = ModelCheckpoint(file, monitor='val_loss',
                                     verbose=1, save_best_only=True, mode='min')
        earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=80)  # patience=80
        callbacks_list = [checkpoint, earlystopping]
        # start training and use 20% training data to validate
        batch_size = 128
        # TODO: epochs=1000
        model.fit(xTrain, yTrain, validation_split=0.2, epochs=1000, batch_size=batch_size, callbacks=callbacks_list,
                  verbose=2)
        # test the model
        mae = model.evaluate(xTest, yTest, batch_size=batch_size)
        print('---Test scores: ', mae)

        del model
        file = PredictionServiceConfig.CNN_PATH + "weights-cnn-" + str(self.nowtime) + ".hdf5"
        lr = 0.001
        model = cnnModel(0.0, 0.01, num_dim)
        model.load_weights(file)
        model.compile(loss='mae', optimizer=Adam(lr=lr, decay=0))

        yPredict = model.predict(xTest)
        Sp_pred_list, Sp_true_list = self._convert_results_back(scaler, yPredict, yTest)
        cnn_yPredict = list(map(lambda x: x[0], yPredict))

        return Sp_pred_list, Sp_true_list, cnn_yPredict

    def build_dnn_model(self, xTrain, yTrain, xTest, yTest, scaler):

        # traing by Adam and loss func is mae
        model = dnnModel()
        model.summary()
        lr = 0.001
        model.compile(loss='mae', optimizer=Adam(lr=lr, decay=0))
        # set checkpoint to save weights with the MIN mae

        file = PredictionServiceConfig.CNN_PATH + "weights-selfstacking-dnn-" + str(self.nowtime) + ".hdf5"
        checkpoint = ModelCheckpoint(file, monitor='val_loss',
                                     verbose=1, save_best_only=True, mode='min')

        earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)  # patience=200
        callbacks_list = [checkpoint, earlystopping]

        # start training and use 20% training data to validate
        batch_size = 512
        # TODO: epochs = 10000
        model.fit(xTrain, yTrain, validation_split=0.2, epochs=10000,
                  batch_size=batch_size, callbacks=callbacks_list, verbose=2)

        # test the model
        mae = model.evaluate(xTest, yTest, batch_size=batch_size)
        print('---Test scores: ', mae)

        del model
        file = PredictionServiceConfig.CNN_PATH + "weights-selfstacking-dnn-" + str(self.nowtime) + ".hdf5"
        lr = 0.001
        model = dnnModel()
        model.load_weights(file)
        model.compile(loss='mae', optimizer=Adam(lr=lr, decay=0))

        yPredict = model.predict(xTest)
        Sp_pred_list, Sp_true_list = self._convert_results_back(scaler, yPredict, yTest)

        return Sp_pred_list, Sp_true_list

    def get_training_data(self, from_date=None, to_date=None):
        folder_path = PredictionServiceConfig.SOLD_PATH
        self.fetch_from_bucket_to_dir(self.datalake_bucket, self.sold_bucket_prefix, folder_path)
        os.chdir(folder_path)
        df_list = []
        for counter, file in enumerate(glob.glob('*.csv.gz')):
            file_path = folder_path + str(file)
            df_inc = pd.read_csv(file_path, compression='gzip')
            df_list.append(df_inc)
            print(file)
            print(len(df_inc))
        df = pd.concat(df_list, axis=0, ignore_index=True)
        df['Cd'] = pd.to_datetime(df['Cd'])
        if from_date is not None:
            df = df[df['Cd'] >= from_date]
        if to_date is not None:
            df = df[df['Cd'] <= to_date]
        print(df)
        print(len(df))
        df = clean_whole_data(df_raw_data=df)

        return df

    def fetch_from_bucket_to_dir(self, bucket_name, prefix, dest):
        response = self.s3client.list_objects(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' in response.keys():
            for file in response['Contents']:
                fname = file['Key']
                if len(fname) == len(prefix):
                    print(fname)
                    continue
                local_folder_file = fname.split('/')[-1]
                print(local_folder_file)
                download_full_path = dest + local_folder_file
                if not check_local_file_exist(download_full_path):
                    self.s3client.download_file(bucket_name, file['Key'], download_full_path)

    @staticmethod
    def _save_dnn_column_order(df_stack):
        all_features = df_stack.columns.tolist()
        # no Sp_dol when predict
        stacking_top_features = ['_id', 'A_c', 'Kit_plus', 'Lp_dol', 'Pool', 'Taxes', 'Den_fr', 'Depth',
                                 'Garage', '150-to-200', '200-to-250', '250-to-350', 'larger-than350',
                                 'Total_rooms', 'Bsmt1_out_Fin W/O', 'Bsmt1_out_Finished', 'Gar_type_Built-In',
                                 'Heating_Water', 'Style_2 1/2 Storey', 'Style_2-Storey', 'Type_own1_out_Detached',
                                 'Community_Annex', 'Community_Bedford Park-Nortown',
                                 'Community_Bridle Path-Sunnybrook-York Mills',
                                 'Community_Eastlake', 'Community_Forest Hill South', 'Community_Lawrence Park South',
                                 'Community_Lorne Park', 'Community_Patterson', 'Community_Rosedale-Moore Park',
                                 'Community_St. Andrew-Windfields', 'CNN_pred', 'Sp_dol']
        for ele in all_features:
            if ele not in set(stacking_top_features + ['_id']):
                df_stack = df_stack.drop([ele], axis=1)

        price = df_stack['Sp_dol']
        df_stack = df_stack.drop(['Sp_dol'], axis=1)
        # make sure all the columns are in the same order
        with open(PredictionServiceConfig.STACK_COL_ORDER, 'wb') as fp:
            pickle.dump(df_stack.columns.tolist(), fp)

        df_stack.insert(len(df_stack.columns), 'Sp_dol', price)

        print(df_stack.columns.tolist())
        return df_stack

    @staticmethod
    def _split_data(df, splits=None):
        # split data into X and Y
        arr = df.values
        totalNum, totalFeature = arr.shape
        if splits is None:
            splits = int(0.9 * totalNum)
        print('...', totalNum, totalFeature)

        # For X: not choose the first col (_id)  and the last col (Sp_dol)
        # For y: choose the last col (Sp_dol)
        xTest, yTest = arr[splits:, 1: -1], arr[splits:, -1]
        print(len(xTest))
        X, Y = arr[0: splits, 1: -1], arr[0: splits, -1]
        return X, Y, xTest, yTest

    @staticmethod
    def _convert_results_back(scaler, yPredict, yTest):

        Sp_true_list = []
        Sp_pred_list = []
        for i in range(len(yTest)):
            Sp_price = int(np.expm1(scaler.inverse_transform([[yTest[i]]])[0][0]))
            Sp_true_list.append(Sp_price)
            Sp_pred = int(np.expm1(scaler.inverse_transform([[yPredict[i][0]]])[0][0]))
            Sp_pred_list.append(Sp_pred)
        return Sp_pred_list, Sp_true_list

    @staticmethod
    # TODO: change back
    def _show_accuracy(arr, splits, Sp_pred_list, Sp_true_list):
        data = np.c_[arr[splits:, 0], Sp_pred_list, Sp_pred_list, Sp_true_list]
        data = pd.DataFrame({'_id': data[:, 0], 'Lp': data[:, 1], 'Pp': data[:, 2], 'Sp': data[:, 3]})
        # data = data.tail(700)
        accuracy = Show_Accuracy(result_df=data)
        accuracy.show()


if __name__ == "__main__":
    # for reproducibility use
    '''
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # os.environ["PYTHONHASHSEED"] = '0'
    np.random.seed(7)
    random.seed(12345)
    session_conf = tensorflow.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tensorflow.set_random_seed(1234)
    sess = tensorflow.Session(graph=tensorflow.get_default_graph(), config=session_conf)
    K.set_session(sess)
    '''
    # test whether use GPU or not
    print(K.tensorflow_backend._get_available_gpus())
    flow = TrainingStage()
    flow.training_flow(from_date='2018-01-01')
