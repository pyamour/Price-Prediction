import pandas as pd
from PredictionService.src.main.config import PredictionServiceConfig
from PredictionService.cnn_model.cnn_mapper import process_predict_data
from sklearn.preprocessing import MinMaxScaler
from PredictionService.src.main.toronto_area_model.toronto_model import cnnModel
from PredictionService.src.main.cnn_model.utils import reshape_X
from keras.optimizers import Adam
import numpy as np
import pickle


class PredictionStage:
    def __init__(self):
        self.df_predict_data = pd.read_csv(PredictionServiceConfig.DATA_PATH + 'toronto_test_file.csv')
        return

    def prediction_flow(self):
        df = process_predict_data(self.df_predict_data)
        # ---------------<6> 0~100 shrink 'Taxes'
        scale_min = 0
        scale_max = 100
        # the following values were previously acquired from the whole dataset(train + validate + test)
        # For toronto house only
        min_tax = 0.01
        max_tax = 12.612
        min_sp = 11.648338835562749
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
        num_dim, num_zero = 15, 5
        xTest = reshape_X(xTest, num_dim, num_zero)
        print('xTest and yTest shapes are: ', xTest.shape, yTest.shape)

        # ---------------<2> build the CNN model
        # set random seed for reproducibility
        seed = 7
        np.random.seed(seed)

        # ---------------<3> load weights
        weight_file = PredictionServiceConfig.CNN_PATH + 'weights-cnn-20190611-2246.hdf5'
        lr = 0.001
        model = cnnModel(0.0, 0.01, num_dim)
        model.summary()
        model.load_weights(weight_file)
        model.compile(loss='mae', optimizer=Adam(lr=lr, decay=0))

        # ---------------<4> predict and evaluate
        yPredict = model.predict(xTest)

        Sp_pred_list = []
        Lp_dol_list = []
        output = pd.DataFrame(columns=['_id', 'CNN_pred'])
        # recover the data and save into lists
        for i in range(len(yPredict)):
            Sp_pred = int(np.expm1((yPredict[i][0] - scale_min) / (scale_max - scale_min) * (max_sp - min_sp) + min_sp))
            Sp_pred_list.append(Sp_pred)

        output['_id'] = arr[:, 0]
        output['CNN_pred'] = Sp_pred_list

        output.to_csv(PredictionServiceConfig.RESULT_PATH + 'cnn_results_611.csv', index=False)


if __name__ == "__main__":
    flow = PredictionStage()
    flow.prediction_flow()
