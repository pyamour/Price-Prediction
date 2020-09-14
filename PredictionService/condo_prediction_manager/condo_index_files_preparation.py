import re
import statistics
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import implicit
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
import pandas as pd
from PredictionService.config import PredictionServiceConfig


def generate_condo_quantile_file(df_condo):
    condo_price_dic = generate_condo_price_index(df_condo, purpose='train')
    if condo_price_dic is None:
        return None
    compute_condo_quatiles(condo_price_dic)


def generate_condo_price_index(df_condo, purpose):  # train / predict
    condo_dic = dict()
    for i, row in df_condo.iterrows():
        addr = str(row['Addr'])
        br = str(row['Br'])
        brp = '' if row['Br_plus'] == '0' or row['Br_plus'] == '0.0' else str(row['Br_plus'])
        bath = str(row['Bath_tot'])
        key = addr + ":" + br + ":" + brp + ":" + bath

        try:
            tax = float(row['Taxes'])
        except:
            print(row['Taxes'])

        records = condo_dic.get(key, [])
        if purpose == 'train':
            if len(records) <= 4:
                records.append(
                    (row['_id'], row['Lp_dol'], row['Sp_dol'], float(row['Sp_dol']) / float(row['Lp_dol']), tax))
        elif purpose == 'predict':
            records.append((row['_id'], row['Lp_dol'], row['Taxes'], tax))
        condo_dic[key] = records

    return condo_dic


def compute_condo_quatiles(condo_price_dic):
    df_quantile = pd.DataFrame(
        columns=['key', 'lp_q0', 'lp_q1', 'lp_q2', 'lp_q3', 'sp_q1', 'ratio_q1', 'ratio_q2', 'tax_q1', 'num'])

    i = 0
    for k in condo_price_dic:
        records = condo_price_dic[k]
        lprices = [float(t[1]) for t in records]
        sprices = [float(t[2]) for t in records]
        ratios = [t[3] for t in records]
        taxes = [t[4] for t in records if t[3] > 0.0]

        num = len(lprices)
        lp_q0 = np.min(lprices)
        lp_q1 = np.quantile(lprices, 0.35)
        lp_q2 = np.quantile(lprices, 0.5)
        lp_q3 = np.quantile(lprices, 0.75)

        sp_q1 = np.quantile(sprices, 0.3)

        ratio_q1 = np.quantile(ratios, 0.2)
        ratio_q2 = np.quantile(ratios, 0.45)
        tax_q1 = 0

        if ratio_q2 >= 1.3:
            ratio_q2 = 1.05

        if ratio_q1 >= 1.0:
            ratio_q1 = 0.95

        if len(taxes) != 0:
            tax_q1 = np.quantile(taxes, 0.35)

        df_quantile.loc[i] = [k, lp_q0, lp_q1, lp_q2, lp_q3, sp_q1, ratio_q1, ratio_q2, tax_q1, num]

        i += 1
    df_quantile.to_csv(RealMasterServiceConfig.CONDO_QUANTILE_FILE, index=False)

    return df_quantile


def get_condo_similarity_index(df_condo):  # train use only
    condo_dic = {}
    condo_types = {}
    for i, row in df_condo.iterrows():
        if float(row['Br']) == 0:
            continue

        addr = re.sub('\s+', ' ', row['Addr']).strip()
        br = str(row['Br'])
        brp = '' if row['Br_plus'] == '0' or row['Br_plus'] == '0.0' else str(row['Br_plus'])
        bath = str(row['Bath_tot'])
        key = addr + ":" + br + ":" + brp + ":" + bath

        ctype = br + ":" + brp + ":" + bath
        count = condo_types.get(ctype, 0)
        count += 1
        condo_types[ctype] = count
        records = condo_dic.get(key, [])
        records.append((float(row['Sp_dol'])))
        condo_dic[key] = records

    return condo_dic, condo_types


def get_condo_price_matrix(condo_dic, condo_types):
    ctype_num_dic = {k: condo_types[k] for k in condo_types if condo_types[k] > 30}
    key_median_price_dic = {k: int(statistics.median(condo_dic[k])) for k in condo_dic}
    df_condo_price_matrix = pd.DataFrame([])
    for key in key_median_price_dic:
        key_part = key.split(":", 1)
        addr = key_part[0]
        ctype = key_part[1]
        if ctype in ctype_num_dic.keys():
            df_condo_price_matrix.loc[addr, ctype] = key_median_price_dic[key]
    df_condo_price_matrix.fillna(0, inplace=True)
    condo_price_matrix = df_condo_price_matrix.values
    df_condo_price_matrix['addr'] = df_condo_price_matrix.index
    df_condo_price_matrix.index = range(len(df_condo_price_matrix))
    addr_dic = df_condo_price_matrix['addr'].to_dict()

    return condo_price_matrix, addr_dic


def get_neighbours(addr_dic, similarities, save_to=None):
    neighbours_addr = {}
    neighbours_sim = {}
    for row in range(similarities.shape[0]):
        similar_indices = np.argsort(similarities[row, :])[-1:-21:-1]
        # print(similar_indices)
        for idx in similar_indices:
            if idx != row:
                addrs = neighbours_addr.get(addr_dic[row], [])
                addrs.append(addr_dic[idx])
                neighbours_addr[addr_dic[row]] = addrs

                sims = neighbours_sim.get(addr_dic[row], [])
                sims.append(similarities[row, idx])
                neighbours_sim[addr_dic[row]] = sims

    if save_to:
        with open(save_to, 'w') as f:
            for k in neighbours_addr:
                f.write(k + "\t")
                f.write(','.join(neighbours_addr[k]))
                f.write('\n')

                f.write(k + "\t")
                # f.write(','.join([str(round( float(v)/sum(neighbours_sim[k]), 5) ) for v in neighbours_sim[k]]))
                f.write(','.join([str(round(v, 6)) for v in neighbours_sim[k]]))
                f.write('\n')

    return neighbours_addr, neighbours_sim


# ====================== ALS + Implicit
def get_als_neighbours(addr_dic, model, size, save_to=None):
    neighbours_addr = {}
    neighbours_sim = {}
    for row in range(size):
        pairs = model.similar_users(row, N=20)
        for i in range(len(pairs)):
            idx = pairs[i][0]
            v = pairs[i][1]
            if idx != row:
                addrs = neighbours_addr.get(addr_dic[row], [])
                addrs.append(addr_dic[idx])
                neighbours_addr[addr_dic[row]] = addrs

                sims = neighbours_sim.get(addr_dic[row], [])
                sims.append(round(v, 4))
                neighbours_sim[addr_dic[row]] = sims

    if save_to:
        with open(save_to, 'w') as f:
            for k in neighbours_addr:
                f.write(k + "\t")
                f.write(','.join(neighbours_addr[k]))
                f.write('\n')

                f.write(k + "\t")
                f.write(','.join([str(round(v, 5)) for v in neighbours_sim[k]]))
                f.write('\n')

    return neighbours_addr, neighbours_sim


def do_als(prices_matrix, addr_dic):
    A_sparse = sparse.csr_matrix(prices_matrix)
    model = implicit.als.AlternatingLeastSquares(factors=10, iterations=50)
    model.fit(A_sparse.T)
    neighbours_addr, neighbours_sim = get_als_neighbours(addr_dic, model, len(prices_matrix),
                                                         save_to=PredictionServiceConfig.IMPLICIT_ALS_NEIGHBORS_FILE)
    return neighbours_addr, neighbours_sim


# ===================== Sparse matrix cosine similarity
def sparse_cos(prices_matrix, addr_dic):
    print(prices_matrix)
    A_sparse = sparse.csr_matrix(prices_matrix)
    similarities = cosine_similarity(A_sparse)
    neighbours_addr, neighbours_sim = get_neighbours(addr_dic, similarities,
                                                     PredictionServiceConfig.FULL_NEIGHBORS_FILE)
    return neighbours_addr, neighbours_sim


# ============= ALS + cosine_similarity
def do_als_cos(prices_matrix, addr_dic):
    A_sparse = sparse.csr_matrix(prices_matrix)
    model = implicit.als.AlternatingLeastSquares(factors=10, iterations=50)
    model.fit(A_sparse.T)
    condo_vecs = model.user_factors
    als_similarities = cosine_similarity(condo_vecs)
    neighbours_addr, neighbours_sim = get_neighbours(addr_dic, als_similarities,
                                                     PredictionServiceConfig.ALS_NEIGHBORS_FILE)
    return neighbours_addr, neighbours_sim


# ============ PCA + cosine_similarity
def do_pca_cos(prices_matrix, addr_dic):
    pca = PCA(n_components=10)
    pca_matrix = pca.fit_transform(prices_matrix)
    # pca.explained_variance_ratio_

    print("Sum of variance ratio:", sum(pca.explained_variance_ratio_))
    pca_similarities = cosine_similarity(pca_matrix)
    neighbours_addr, neighbours_sim = get_neighbours(addr_dic, pca_similarities,
                                                     PredictionServiceConfig.PCA_NEIGHBORS_FILE)
    return neighbours_addr, neighbours_sim


# ============== NMF + cosine_similarity
def do_nmf_cos(prices_matrix, addr_dic):
    nmf = NMF(n_components=10, init='random', random_state=42)
    W = nmf.fit_transform(prices_matrix)
    nmf_similarities = cosine_similarity(W)
    neighbours_addr, neighbours_sim = get_neighbours(addr_dic, nmf_similarities,
                                                     PredictionServiceConfig.NMF_NEIGHBORS_FILE)
    return neighbours_addr, neighbours_sim


def init_condo_index_files(df_condo):
    generate_condo_quantile_file(df_condo)
    condo_dic, condo_types = get_condo_similarity_index(df_condo)
    condo_prices_matrix, addr_dic = get_condo_price_matrix(condo_dic, condo_types)
    sparse_cos(condo_prices_matrix, addr_dic)
    do_als(condo_prices_matrix, addr_dic)
    do_als_cos(condo_prices_matrix, addr_dic)
    do_pca_cos(condo_prices_matrix, addr_dic)
    do_nmf_cos(condo_prices_matrix, addr_dic)


if __name__ == "__main__":
    condo_file = '/var/csv_file/sold_condo_from_2018.csv'
    df_condo = pd.read_csv(condo_file)
    # init_condo_index_files(df_condo)
    generate_condo_quantile_file(df_condo)
