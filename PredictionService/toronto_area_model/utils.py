import numpy as np
def convert_results_back(scaler, yPredict, yTest):
    Sp_true_list = []
    Sp_pred_list = []
    for i in range(len(yTest)):
        Sp_price = int(np.expm1(scaler.inverse_transform([[yTest[i]]])[0][0]))
        Sp_true_list.append(Sp_price)
        Sp_pred = int(np.expm1(scaler.inverse_transform([[yPredict[i][0]]])[0][0]))
        Sp_pred_list.append(Sp_pred)
    return Sp_pred_list, Sp_true_list