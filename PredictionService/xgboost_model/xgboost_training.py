from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from PredictionService.xgboost_model import xgb_mapper
from sklearn.model_selection import RandomizedSearchCV
from PredictionService.config import PredictionServiceConfig
import pickle
import os
import pandas as pd
import numpy as np
from PredictionService.utils.utils import remove_local_file
from PredictionService.cnn_model.show_accuracy import Show_Accuracy
from sklearn.model_selection import train_test_split
import time


def train_xgboost_model(df, xgb_path):
    xgb_mapper.init_xgb_mapper(df)
    df_X = xgb_mapper.proc_xgb_mapper(df)
    df_y = np.log1p(df['Sp_dol'])
    param = {
        'objective': 'reg:squarederror',
        'tree_method': 'gpu_hist',
        "predictor": 'gpu_predictor'
    }
    xgb_reg = XGBClassifier(**param)
    param_dist = {'max_depth': [1, 3, 5, 10],
                  'n_estimators': [50, 100, 200, 500, 1000],
                  'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
                  'subsample': [0.3, 0.5, 0.8, 1.0],
                  'colsample_bytree': [0.3, 0.5, 0.8, 1.0],
                  'reg_lambda': [0.01, 0.1, 1.0, 10]
                  }
    xgb_model = RandomizedSearchCV(xgb_reg, param_distributions=param_dist, n_iter=10, cv=5,
                                   scoring='neg_mean_absolute_error')  # n_jobs=constants.CORE,

    xgb_model.fit(df_X.values, df_y.values)
    remove_local_file(xgb_path)
    pickle.dump(xgb_model, open(xgb_path, 'wb'))
    return


def train_xgboost_with_gpu(X, y, xgb_path):
    # xgb_mapper.init_xgb_mapper(df)
    # df_X = xgb_mapper.proc_xgb_mapper(df)
    # df_X = df
    # df_y = np.log1p(df['Sp_dol'])
    # X = df_X.values
    # y = df_y.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)
    # TODO: num_round = 1000
    num_round = 1000
    params = {'objective': 'reg:squarederror',  # Specify multiclass classification
              'tree_method': 'gpu_hist',  # Use GPU accelerated algorithm
              'predictor': 'gpu_predictor',
              'eval_metric': 'mae'
              }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    gridsearch_params1 = [
        (max_depth, min_child_weight)
        for max_depth in range(5, 12)
        for min_child_weight in range(1, 8)
    ]
    gridsearch_params2 = [
        (subsample, colsample)
        for subsample in [i / 10. for i in range(3, 11)]
        for colsample in [i / 10. for i in range(3, 11)]
    ]
    gridsearch_params3 = [0.5, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001]
    tmp = time.time()
    max_depth, min_child_weight = xgb_param_grid_search(params, dtrain, num_round, gridsearch_params1,
                                                        ['max_depth', 'min_child_weight'])
    subsample, colsample = xgb_param_grid_search(params, dtrain, num_round, gridsearch_params2,
                                                 ['subsample', 'colsample_bytree'])
    eta = xgb_param_grid_search(params, dtrain, num_round, gridsearch_params3, ['eta'])

    best_params = {
        'tree_method': 'gpu_hist',
        'colsample_bytree': colsample,
        'eta': eta,
        'eval_metric': 'mae',
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'objective': 'reg:squarederror',
        'subsample': subsample
    }
    print(best_params)
    gpu_res = {}
    bst = xgb.train(best_params, dtrain, num_round, evals=[(dtest, 'test')], evals_result=gpu_res)
    print("GPU Training Time: %s seconds" % (str(time.time() - tmp)))
    remove_local_file(xgb_path)
    bst.save_model(xgb_path)
    return


def xgb_param_grid_search(params, dtrain, num_boost_round, gridsearch_params, param_list):
    min_mae = float("Inf")
    best_params = None
    if len(param_list) == 2:
        for ele1, ele2 in gridsearch_params:
            print("CV with {}={}, {}={}".format(param_list[0], ele1, param_list[1], ele2))
            # Update our parameters
            params[param_list[0]] = ele1
            params[param_list[1]] = ele2
            # Run CV
            cv_results = xgb_cv(params, dtrain, num_boost_round)
            # Update best MAE
            mean_mae = cv_results['test-mae-mean'].min()
            boost_rounds = cv_results['test-mae-mean'].argmin()
            print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
            if mean_mae < min_mae:
                min_mae = mean_mae
                best_params = (ele1, ele2)
        print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
        return best_params[0], best_params[1]

    if len(param_list) == 1:
        for eta in gridsearch_params:
            print("CV with eta={}".format(eta))
            params[param_list[0]] = eta
            # Run and time CV
            cv_results = xgb_cv(params, dtrain, num_boost_round)
            # Update best score
            mean_mae = cv_results['test-mae-mean'].min()
            boost_rounds = cv_results['test-mae-mean'].argmin()
            print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
            if mean_mae < min_mae:
                min_mae = mean_mae
                best_params = eta
        print("Best params: {}, MAE: {}".format(best_params, min_mae))
        return best_params


def xgb_cv(params, dtrain, num_boost_round):
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics=['mae'],
        early_stopping_rounds=100
    )
    return cv_results


def predict_use_xgboost(xTest, xgb_path):
    # df_test = xgb_mapper.proc_xgb_mapper(df)
    print(xgb_path)
    if os.path.isfile(xgb_path):
        '''
        for sklearn
        '''
        # xgb_model = pickle.load(open(xgb_path, 'rb'))
        # pred_log_results = xgb_model.predict(df_test.values)
        '''
        for xgboost
        '''
        xgb_model = xgb.Booster()
        xgb_model.load_model(xgb_path)
        dtest = xgb.DMatrix(xTest)
        pred_log_results = xgb_model.predict(dtest)

        # results_arr = list(map(np.expm1, pred_log_results))
        # print(results_arr)

        return pred_log_results
    else:
        return None


def predict_xgb_and_show_accuracy(df_test, xgb_model_path):
    # df_test = pd.read_csv(test_file)
    df_test.index = range(len(df_test))

    results_arr = predict_use_xgboost(df_test, xgb_model_path)
    df_pred = pd.DataFrame(results_arr, columns=['xgb_pred'])
    df_results = pd.concat([df_test[['_id', 'Lp_dol', 'Sp_dol']], df_pred], axis=1)
    df_output = df_results.rename({'_id': '_id', 'Lp_dol': 'Lp', 'Sp_dol': 'Sp', 'xgb_pred': 'Pp'}, axis='columns')
    accuracy = Show_Accuracy(result_df=df_output)
    accuracy.show()
    # df_output.to_csv(PredictionServiceConfig.RESULT_PATH + 'xgb_result_0611.csv', index=False)
    # print(df_output)
    return results_arr


if __name__ == "__main__":
    model_path = PredictionServiceConfig.MODEL_PATH + 'xgb_model_gpu.pkl'
    '''
    Train
    '''
    data_file = PredictionServiceConfig.DATA_PATH + 'toronto_house_data.csv'
    df_data = pd.read_csv(data_file)
    train_xgboost_with_gpu(df_data, model_path)
    # train_xgboost_model(df_data, model_path)

    '''
    Test
    '''
    # test_file = PredictionServiceConfig.DATA_PATH + 'toronto_test_file.csv'
    # generate_predict_file(test_file, model_path)

    '''
    Validate
    '''
    # result_file = PredictionServiceConfig.RESULT_PATH + 'xgb_result_0611.csv'
    # df = pd.read_csv(result_file)
    # accuracy = Show_Accuracy(result_df=df)
    # accuracy.show()
