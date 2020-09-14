import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from PredictionService.cnn_model.show_accuracy import Show_Accuracy
from PredictionService.config import PredictionServiceConfig, s3config
from PredictionService.cnn_model.cnn_mapper import process_train_data
from PredictionService.toronto_area_model.toronto_model import cnnModel, dnnModel
from DataClean.data_cleaning.data_cleaning import clean_whole_data
import os
import boto3
from PredictionService.utils.utils import check_local_file_exist
import glob
import gc
from PredictionService.toronto_area_model.autokeras_training import build_auto_mlp_model
from PredictionService.toronto_area_model.utils import convert_results_back

class TrainingStage:

    def __init__(self):
        # name cnn weights and dnn weights with nowtime
        self.s3client = boto3.client('s3', aws_access_key_id=s3config.aws_access_key_id,
                                     aws_secret_access_key=s3config.aws_secret_access_key)
        self.datalake_bucket = s3config.datalake_price_bucket
        self.sold_bucket_prefix = 'Price/Online-Results/internalpriceprediction/sold/'
        self.nowtime = datetime.now().strftime("%Y%m%d-%H%M")
        # TODO: for test use
        # self.df_clean_data = pd.read_csv(clean_house_data_file)
        self.df_clean_data = pd.read_csv(PredictionServiceConfig.DATA_PATH + 'toronto_house_data.csv')
        return

    def training_flow(self, from_date=None, to_date=None):
        # ------------------------------------------------------------------------------------------------------------
        # =====Step 1: fetch and clean data
        # df_clean_data = self.get_training_data(from_date=from_date, to_date=to_date)

        # ------------------------------------------------------------------------------------------------------------
        # =====Step 2: convert data into cnn-read format
        df = process_train_data(self.df_clean_data, timenow=self.nowtime)
        # Scale Taxes & Sp_dol
        scaler = MinMaxScaler(feature_range=(0, 100))
        for ele in ['Taxes', 'Sp_dol']:
            df[[ele]] = scaler.fit_transform(df[[ele]])
        '''
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

        X, y, xTest, yTest = self._split_data(df, splits=cnn_split)
        num_dim, num_zero = calculate_input_dims_and_zero_cols(len(X[0]))
        cnn_X = reshape_X(X, num_dim, num_zero)
        cnn_xTest = reshape_X(xTest, num_dim, num_zero)
        # test_size=0
        cnn_xTrain, _, cnn_yTrain, _ = train_test_split(cnn_X, y, test_size=None, random_state=42, shuffle=False)

        # set random seed for reproducibility
        seed = 7
        np.random.seed(seed)

        cnn_sp_pred_list, cnn_sp_true_list, cnn_yPredcit, train_cnn_sp_pred_list, train_cnn_sp_true_list, train_cnn_yPredcit = self.build_cnn_model(
            cnn_xTrain, cnn_yTrain, cnn_xTest, yTest, scaler, num_dim)
        # cnn_sp_pred_list, cnn_sp_true_list, cnn_yPredcit, train_cnn_sp_pred_list, train_cnn_sp_true_list, train_cnn_yPredcit = self.build_cnn_model(
        #     cnn_X, cnn_y, cnn_xTest, cnn_yTest, scaler, num_dim)

        print("------Performance on training set------")
        self._show_accuracy(df.values, cnn_split, train_cnn_sp_pred_list, train_cnn_sp_true_list, data_type='train')
        print("------Performance on test set------")
        self._show_accuracy(df.values, cnn_split, cnn_sp_pred_list, cnn_sp_true_list, data_type='test')

        # ------------------------------------------------------------------------------------------------------------
        # =====Step 5: train xgb model
        xgb_model_path = PredictionServiceConfig.MODEL_PATH + 'xgb_model_gpu.pkl'
        '''
        #Train
        '''
        # todo: same data processing as cnn
        gc.collect()
        train_xgboost_with_gpu(X, y, xgb_model_path)
        '''
        #Test
        '''
        xgb_yPredict = predict_use_xgboost(xTest, xgb_model_path)
        xgb_sp_pred_list = []
        xgb_sp_true_list = []
        for i in range(len(yTest)):
            xgb_sp_price = int(np.expm1(scaler.inverse_transform([[yTest[i]]])))
            xgb_sp_true_list.append(xgb_sp_price)
            xgb_sp_pred = int(np.expm1(scaler.inverse_transform([[xgb_yPredict[i]]])))
            xgb_sp_pred_list.append(xgb_sp_pred)
        print(xgb_sp_pred_list)
        # xgb_sp_pred_list = list(map(lambda x: np.expm1(scaler.inverse_transform([[x]])), xgb_yPredict))
        # xgb_sp_true_list = list(map(lambda x: np.expm1(scaler.inverse_transform([[x]])), yTest))
        self._show_accuracy(df.values, cnn_split, xgb_sp_pred_list, xgb_sp_true_list, data_type='test')
        self._show_accuracy(df.values, cnn_split, xgb_sp_pred_list, xgb_sp_true_list, data_type='test')

        # ------------------------------------------------------------------------------------------------------------
        # =====Step 6: train dnn stacking model
        df_stack['cnn_pred'] = cnn_yPredcit
        df_stack['xgb_pred'] = xgb_yPredict
        df_stack = self._save_dnn_column_order(df_stack)
        print(df_stack.head(10))
        df_stack.to_csv(PredictionServiceConfig.DATA_PATH + 'xgb_cnn_stack_file.csv', index=False)
        
        '''
        stack_file = PredictionServiceConfig.DATA_PATH + 'xgb_cnn_stack_file.csv'
        df_stack = pd.read_csv(stack_file)
        dnn_split = int(len(df_stack)*0.9)

        # dnn_X, dnn_y, dnn_xTest, dnn_yTest = self._split_data(df_stack, splits=dnn_split)
        # test_size=0
        # dnn_xTrain, _, dnn_yTrain, _ = train_test_split(dnn_X, dnn_y, test_size=None, random_state=42, shuffle=False)
        # stack_sp_pred_list, stack_sp_true_list = self.build_dnn_model(dnn_xTrain, dnn_yTrain, dnn_xTest, dnn_yTest,
        #                                                               scaler)

        stack_sp_pred_list, stack_sp_true_list = build_auto_mlp_model(df_stack, dnn_split, scaler, self.nowtime)
        print(stack_sp_pred_list)
        print(stack_sp_true_list)
        self._show_accuracy(df_stack.values, dnn_split, stack_sp_pred_list, stack_sp_true_list, data_type='test')

        return

    def build_cnn_model(self, xTrain, yTrain, xTest, yTest, scaler, num_dim):
        # traing by Adam and loss func is mae
        # TODO:
        model = cnnModel(0.0, 0.01, num_dim)
        # num_features = len(xTrain[0])
        # model = fully_connected_Model(num_features)
        model.summary()
        # TODO: lr=0.001
        lr = 0.001
        # TODO: mae
        model.compile(loss='mae', optimizer=Adam(lr=lr, decay=0))
        # set checkpoint to save weights with the MIN mae
        file = PredictionServiceConfig.CNN_PATH + "weights-cnn-" + str(self.nowtime) + ".hdf5"
        checkpoint = ModelCheckpoint(file, monitor='val_loss',
                                     verbose=1, save_best_only=True, mode='min')
        earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)  # patience=80
        callbacks_list = [checkpoint, earlystopping]
        # start training and use 20% training data to validate
        batch_size = 128
        # TODO: epochs=1000
        model.fit(xTrain, yTrain, validation_split=0.2, epochs=1000, batch_size=batch_size, callbacks=callbacks_list,
                  verbose=2)
        # test the model
        train_mae = model.evaluate(xTrain, yTrain, batch_size=batch_size)
        test_mae = model.evaluate(xTest, yTest, batch_size=batch_size)
        print('---Train scores: ', train_mae)
        print('---Test scores: ', test_mae)
        # # TODO:
        # # ------------------------------------------------------------------------------------------------------------
        # # predict xTest
        # yPredict = model.predict(xTest)
        # Sp_pred_list, Sp_true_list = self._convert_results_back(scaler, yPredict, yTest)
        # cnn_yPredict = list(map(lambda x: x[0], yPredict))
        #
        # # ------------------------------------------------------------------------------------------------------------
        # # predict xTrain
        # train_yPredict = model.predict(xTrain)
        # train_Sp_pred_list, train_Sp_true_list = self._convert_results_back(scaler, train_yPredict, yTrain)
        # train_cnn_yPredict = list(map(lambda x: x[0], train_yPredict))
        #
        # ------------------------------------------------------------------------------------------------------------
        # load model
        del model

        file = PredictionServiceConfig.CNN_PATH + "weights-cnn-" + str(self.nowtime) + ".hdf5"
        # TODO: lr = 0.001
        lr = 0.001

        model = cnnModel(0.0, 0.01, num_dim)
        # model = fully_connected_Model(num_features)
        model.load_weights(file)
        # TODO: mae
        model.compile(loss='mae', optimizer=Adam(lr=lr, decay=0))
        # ------------------------------------------------------------------------------------------------------------
        # predict xTest
        yPredict = model.predict(xTest)
        Sp_pred_list, Sp_true_list = convert_results_back(scaler, yPredict, yTest)
        cnn_yPredict = list(map(lambda x: x[0], yPredict))

        # ------------------------------------------------------------------------------------------------------------
        # predict xTrain
        train_yPredict = model.predict(xTrain)
        train_Sp_pred_list, train_Sp_true_list = convert_results_back(scaler, train_yPredict, yTrain)
        train_cnn_yPredict = list(map(lambda x: x[0], train_yPredict))
        return Sp_pred_list, Sp_true_list, cnn_yPredict, train_Sp_pred_list, train_Sp_true_list, train_cnn_yPredict

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
        Sp_pred_list, Sp_true_list = convert_results_back(scaler, yPredict, yTest)

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
                                 'Community_St. Andrew-Windfields', 'cnn_pred', 'xgb_pred', 'Sp_dol']
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
    # TODO: change back
    def _show_accuracy(arr, splits, Sp_pred_list, Sp_true_list, data_type):
        if data_type == 'test':
            data = np.c_[arr[splits:, 0], Sp_pred_list, Sp_pred_list, Sp_true_list]
        elif data_type == 'train':
            data = np.c_[arr[0:len(Sp_pred_list), 0], Sp_pred_list, Sp_pred_list, Sp_true_list]
        else:
            print("wrong data_type!")
        data = pd.DataFrame({'_id': data[:, 0], 'Lp': data[:, 1], 'Pp': data[:, 2], 'Sp': data[:, 3]})
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
    flow.training_flow()
