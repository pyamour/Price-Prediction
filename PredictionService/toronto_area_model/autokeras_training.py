from autokeras import MlpModule
from autokeras import ImageRegressor
from autokeras.backend.torch.loss_function import regression_loss
from autokeras.nn.metric import MSE
from autokeras.backend.torch import DataTransformerMlp
import numpy as np
from PredictionService.src.main.config import PredictionServiceConfig
from autokeras.utils import pickle_from_file


def auto_train_mlp_model(x_train, train_data, test_data, nowtime):
    mlpModule = MlpModule(loss=regression_loss, metric=MSE, searcher_args={}, verbose=True,
                          path=PredictionServiceConfig.MODEL_PATH + 'autokeras_model_' + str(nowtime))
    fit_args = {
        "n_output_node": 1,
        "input_shape": x_train.shape,
        "train_data": train_data,
        "test_data": test_data
    }
    mlpModule.fit(n_output_node=fit_args.get("n_output_node"),
                  input_shape=fit_args.get("input_shape"),
                  train_data=fit_args.get("train_data"),
                  test_data=fit_args.get("test_data"),
                  time_limit=1 * 60)
    # mlpModule.final_fit(train_data, test_data, retrain=False)

    return mlpModule


def build_auto_mlp_model(df, split, scaler, nowtime):
    arr = df.values
    train_data_all, test_data = split_train_test(arr, split)
    train_data, validate_data = split_train_test(train_data_all, int(len(train_data_all) * 0.9))
    x_train, y_train = split_xy(train_data)
    x_validate, y_validate = split_xy(validate_data)
    x_test, y_test = split_xy(test_data)

    x_train = np.squeeze(x_train.reshape((x_train.shape[0], -1)))
    x_validate = np.squeeze(x_validate.reshape((x_validate.shape[0], -1)))
    x_test = np.squeeze(x_test.reshape((x_test.shape[0], -1)))

    data_transformer = DataTransformerMlp(x_train)
    train_arr = data_transformer.transform_train(x_train, y_train)
    validate_arr = data_transformer.transform_test(x_validate, y_validate)
    test_arr = data_transformer.transform_test(x_test)

    # mlp_model = auto_train_mlp_model(x_train, train_arr, validate_arr, nowtime)
    mlp_model = pickle_from_file(PredictionServiceConfig.MODEL_PATH + 'autokeras_model_20190618-0233/55.graph')
    y_predict = mlp_model.predict(test_arr)
    print(y_predict)
    sp_pred_list = []
    sp_true_list = []
    for i in range(len(y_test)):
        ak_sp_price = int(np.expm1(scaler.inverse_transform([[y_test[i]]])))
        sp_true_list.append(ak_sp_price)
        ak_sp_pred = int(np.expm1(scaler.inverse_transform([[y_predict[i][0]]])))
        sp_pred_list.append(ak_sp_pred)

    return sp_pred_list, sp_true_list


def split_train_test(arr, split):
    train = arr[:split]
    test = arr[split:]
    return train, test


def split_xy(arr):
    x = np.array(arr[:, 1:-1], dtype=np.float32)
    y = np.array(arr[:, -1], dtype=np.float32)
    return x, y
