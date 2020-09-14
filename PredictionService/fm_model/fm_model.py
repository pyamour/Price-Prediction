import xlearn as xl
import os
from PredictionService.config import PredictionServiceConfig
from PredictionService.config import constants


def train_fm_model(train_pairs, model_path=PredictionServiceConfig.MODEL_PATH):
    ffm_model = xl.create_fm()
    ffm_model.setTrain(str(train_pairs))
    # TODO: setValidate
    thread_num = int(constants.CORE * 0.8)
    param = {'task': 'reg', 'lr': 0.045, 'k': 4, 'lambda': 0.0000005, 'metric': 'mape', 'fold': 5, 'epoch': 66,
             'opt': 'adagrad', 'nthread': thread_num}
    model_full_path = model_path + PredictionServiceConfig.MODEL_NAME
    if os.path.isfile(model_full_path):
        os.remove(model_full_path)
    ffm_model.fit(param, model_full_path)

    return model_full_path


def predict_use_fm(predict_pairs, timestamp, model_path=PredictionServiceConfig.MODEL_PATH,
                   output_path=PredictionServiceConfig.RESULT_PATH):
    model_full_path = model_path + PredictionServiceConfig.MODEL_NAME
    if os.path.isfile(model_full_path):
        output = output_path + PredictionServiceConfig.OUTPUT_PRE_PREFIX + str(timestamp) + '.txt'
        ffm_model = xl.create_fm()
        ffm_model.setTest(predict_pairs)
        ffm_model.predict(model_full_path, output)
        return output
    else:
        print("Error! No ffm_model.out found!")
        return None

