import pandas as pd
import statistics
from PredictionService.config import PredictionServiceConfig
from PredictionService.config import constants
from PredictionService.condo_prediction_manager.condo_index_files_preparation import generate_condo_price_index


def load_similar_condos(similarity_path):
    i = 0
    neighbours_addr = {}
    neighbours_sim = {}
    with open(similarity_path) as f:
        for line in f:
            cols = line.split('\t')
            neighbors = cols[1].split(',')
            if i % 2 == 0:
                neighbours_addr[cols[0]] = neighbors
            else:
                neighbours_sim[cols[0]] = [float(v) for v in neighbors]

            i += 1
    return neighbours_addr, neighbours_sim


def get_price_from_neighbors(df_quantile, key, lp, implicit_neighbours_addr, pca_neighbours_addr, als_neighbours_addr,
                             nmf_neighbours_addr):
    cols = key.split(':', 1)
    addr = cols[0]
    ctype = cols[1]

    common_neighbors = get_reliable_neighbor(addr, implicit_neighbours_addr, pca_neighbours_addr, als_neighbours_addr,
                                             nmf_neighbours_addr)
    if len(common_neighbors) == 0:
        return int(lp), 0, 0, 0, 0

    neighbors_q0_prices = []
    neighbors_q2_prices = []
    neighbors_q3_prices = []

    neighbors_q2_ratio = []
    for addr in common_neighbors:
        addr = addr + ':' + ctype
        if addr in df_quantile.index:
            neighbors_q0_prices.append(df_quantile.loc[addr, 'lp_q0'])
            neighbors_q2_prices.append(df_quantile.loc[addr, 'lp_q2'])
            neighbors_q3_prices.append(df_quantile.loc[addr, 'lp_q3'])
            neighbors_q2_ratio.append(df_quantile.loc[addr, 'ratio_q2'])

    if len(neighbors_q0_prices) == 0:
        return int(lp), 0, 0, 0, 0

    sum_q0_price = int(statistics.mean(neighbors_q0_prices))
    sum_q2_price = int(statistics.mean(neighbors_q2_prices))
    sum_q3_price = int(statistics.mean(neighbors_q3_prices))

    sum_q2_ratio = round(statistics.mean(neighbors_q2_ratio), 3)

    if (sum_q0_price != 0) and (lp > sum_q0_price * 0.6) and (lp < sum_q0_price * 0.8):
        lp = lp * 1.2
    if (sum_q3_price != 0) and (lp > sum_q3_price * 1.2) and (lp < sum_q3_price * 1.5):
        lp = lp * 0.9

    return int(lp), sum_q2_ratio, sum_q0_price, sum_q2_price, sum_q3_price


def get_reliable_neighbor(addr, implicit_neighbours_addr, pca_neighbours_addr, als_neighbours_addr,
                          nmf_neighbours_addr):
    neighbors1 = []
    neighbors2 = []
    neighbors3 = []
    neighbors4 = []
    common_neighbors = []
    if addr in implicit_neighbours_addr:
        neighbors1 = implicit_neighbours_addr[addr]
    if addr in pca_neighbours_addr:
        neighbors2 = pca_neighbours_addr[addr]
    if addr in als_neighbours_addr:
        neighbors3 = als_neighbours_addr[addr]
    if addr in nmf_neighbours_addr:
        neighbors4 = nmf_neighbours_addr[addr]

    all_neighbors = set(neighbors1 + neighbors2 + neighbors3 + neighbors4)
    for k in all_neighbors:
        count = 0
        if k in neighbors1:
            count += 1
        if k in neighbors2:
            count += 1
        if k in neighbors3:
            count += 1
        if k in neighbors4:
            count += 1

        if count >= 3:
            common_neighbors.append(k)

    return common_neighbors


def predict_condo_price(df_condo, timestamp, quantile_file=PredictionServiceConfig.CONDO_QUANTILE_FILE):
    # version 2 use only
    # implicit_neighbours_addr, neighbours_sim1 = load_similar_condos(
    #                                                   RealMasterServiceConfig.IMPLICIT_ALS_NEIGHBORS_FILE)
    # pca_neighbours_addr, neighbours_sim2 = load_similar_condos(RealMasterServiceConfig.PCA_NEIGHBORS_FILE)
    # als_neighbours_addr, neighbours_sim3 = load_similar_condos(RealMasterServiceConfig.ALS_NEIGHBORS_FILE)
    # nmf_neighbours_addr, neighbours_sim2 = load_similar_condos(RealMasterServiceConfig.NMF_NEIGHBORS_FILE)
    test_condo_dic = generate_condo_price_index(df_condo, purpose='predict')
    if test_condo_dic is None:
        return None
    df_quantile = pd.read_csv(quantile_file)
    keys = df_quantile['key'].values
    df_quantile.set_index('key', inplace=True)
    manual_rules = constants.MANUAL_RULES

    df_prediction = pd.DataFrame(columns=['_id', 'mean_price', 'std'])
    fn = 0

    for k in test_condo_dic:
        records = test_condo_dic[k]
        _id = [t[0] for t in records]
        lprice = [float(t[1]) for t in records]
        taxes = [float(t[3]) for t in records]
        num = len(lprice)
        if k in manual_rules:
            pp = manual_rules[k]
            for i in range(num):
                df_prediction.loc[-1] = [_id[i], int(pp), pp * 0.03]
                df_prediction.index = df_prediction.index + 1  # shifting index
                df_prediction = df_prediction.sort_index()
            continue

        if k in keys and (df_quantile.loc[k]['num'] >= 2):
            quantile_row = df_quantile.loc[k]
            q0p = quantile_row['lp_q0']
            q1p = quantile_row['lp_q1']
            q3p = quantile_row['lp_q3']
            for i in range(num):
                lp = lprice[i]
                tax = taxes[i]
                if (lp <= q1p * 0.85) and (lp > q0p):
                    if (quantile_row['tax_q1'] != 0) and (tax >= quantile_row['tax_q1'] * 0.9 or tax == 0):
                        lp = lp * 1.2

                if lp <= q0p * 0.95:
                    lp = lp * 1.1

                if (lp > q3p) and (lp < q3p * 1.1):
                    lp = lp  # more

                if lp > q3p * 1.1:
                    pp = lp * quantile_row['ratio_q1']
                else:
                    pp = lp * quantile_row['ratio_q2']  # prediction

                if pp / quantile_row['sp_q1'] < 0.7:
                    pp = quantile_row['sp_q1'] * 0.8

                df_prediction.loc[-1] = [_id[i], int(pp), pp * 0.0315]
                df_prediction.index = df_prediction.index + 1  # shifting index
                df_prediction = df_prediction.sort_index()
        else:
            for i in range(num):
                # version 1.0
                fn = fn + 1
                lp = lprice[i]
                pp = lp * 0.98

                # version 2.0
                # lp = lprice[i]
                # lp, *op = get_price_from_neighbors(df_quantile, k, lp, implicit_neighbours_addr, pca_neighbours_addr,
                #                                    als_neighbours_addr, nmf_neighbours_addr)
                # pp = lp * 0.98
                df_prediction.loc[-1] = [_id[i], int(pp), pp * 0.0315]
                df_prediction.index = df_prediction.index + 1  # shifting index
                df_prediction = df_prediction.sort_index()
    df_prediction.to_csv(
        PredictionServiceConfig.RESULT_PATH + PredictionServiceConfig.RESULT_PREFIX + str(timestamp) + '_condo.csv',
        index=False)
    return df_prediction


if __name__ == "__main__":
    quantile_file = PredictionServiceConfig.CONDO_QUANTILE_FILE
    test_file = '/var/csv_file/pred/Listing#20190111184501_condo.csv'
    df_test = pd.read_csv(test_file)
    predict_condo_price(df_test, '20190111184501', quantile_file)
