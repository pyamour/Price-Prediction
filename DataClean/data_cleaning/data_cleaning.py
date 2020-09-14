import pandas as pd
import numpy as np
import math
import os
import time
from DataClean.utils.utils import remove_gz_suffix, remove_gz_suffix_for_condo
from DataClean.config import constants, DataCleanServiceConfig
import glob


# TODO: data format exception (str, float...)
def select_related_rows(df, prefix):
    df = df[df['Taxes'] != 0]
    if prefix == 'Sold':
        df.dropna(subset=['Cd'], inplace=True)
        df = df[df['Sp_dol'] > 50000]
        # TODO: Remove this constraint
        # df['lp/sp'] = abs(df['Lp_dol'] - df['Sp_dol']) / df['Sp_dol']
        # df = df[df['lp/sp'] <= 0.3]
        # df.drop(columns=['lp/sp'], inplace=True)

    if prefix == 'Listing':
        df = df[df['Lp_dol'] > 50000]
        df.drop(columns=['Sp_dol', 'Cd'], inplace=True)
        year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
        cur_date = str(year) + '-' + str(month) + '-' + str(day)
        df['Cd'] = cur_date

    df.index = range(len(df))
    return df


def complement_null(df, depth_median, front_median):
    df[constants.CMPLMT_NONE_COL] = df[constants.CMPLMT_NONE_COL].fillna(value='None')
    df[constants.CMPLMT_ZERO_COL] = df[constants.CMPLMT_ZERO_COL].fillna(value=0)
    df['Den_fr'] = df['Den_fr'].fillna(value='N')

    # Depth / Front_ft:    Condo related cols -> 0     House-related cols -> median
    df_cdhs = df[df['Type_own1_out'].isin(constants.CDHS_LABEL)][['Depth', 'Front_ft']]
    df_part_hs = df[~df['Type_own1_out'].isin(constants.CDHS_LABEL)][['Depth', 'Front_ft']]
    df_cdhs['Depth'] = df_cdhs['Depth'].fillna(value=0)
    df_cdhs['Front_ft'] = df_cdhs['Front_ft'].fillna(value=0)

    if (depth_median == 0) & (front_median == 0):
        depth_median = df_part_hs['Depth'].median()
        front_median = df_part_hs['Front_ft'].median()
        median = [[depth_median, front_median]]
        df_median = pd.DataFrame(median, columns=['depth_median', 'front_median'])
        df_median.to_csv(DataCleanServiceConfig.CLEAN_DATA_MEDIAN_FILE, index=None)

    df_part_hs['Depth'] = df_part_hs['Depth'].fillna(value=depth_median)
    df_part_hs['Front_ft'] = df_part_hs['Front_ft'].fillna(value=front_median)
    depth_front = pd.concat([df_cdhs, df_part_hs], ignore_index=False)
    df = df.join(depth_front, lsuffix='_x', rsuffix='')
    df.drop(columns=['Depth_x', 'Front_ft_x'], inplace=True)

    return df


def process_cols(df, comm_list):
    # Process Area code
    df.Area_code = df.Area_code.astype(str)

    df['Area_code'] = df.Area_code.str.extract('(\d+)', expand=True).astype(float)

    # Process Garage
    df['Garage'] = df['Gar'] + df['Gar_spaces']
    df.drop(columns=['Gar', 'Gar_spaces'], inplace=True)

    # Process lat & lng
    df['lng'] = df['lng'].apply(lambda x: x * (-1))

    # Process Community
    if comm_list is None:
        cm_count = df.Community.value_counts()
        cm_h = {cm_count.index[i]: cm_count.values[i] for i in range(len(cm_count.values)) if
                cm_count.values[i] > constants.COMM_TH}
        selected_cm = [*(cm_h.keys())]

        df_comm = pd.DataFrame(selected_cm, columns=['Comm'])
        df_comm.to_csv(DataCleanServiceConfig.COMM_FILE, index=None)
    else:
        selected_cm = comm_list

    df.Community.where(df['Community'].isin(selected_cm), 'Other', inplace=True)

    return df


def process_date(df):
    df['Cd'] = pd.to_datetime(df['Cd'])
    df['month'] = df.Cd.dt.month
    df.index = range(len(df))
    month_dic = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct',
                 11: 'Nov', 12: 'Dec'}
    df_month = pd.DataFrame(0, index=np.arange(len(df)), columns=constants.MONTH)
    df = pd.concat([df, df_month], axis=1)
    for i, month in enumerate(df['month']):
        df.loc[i, month_dic[month]] = 1
    df.drop(columns='month', inplace=True)
    return df


def rooms_total_area(df):
    df['area'] = 0
    for i in range(1, 13):
        df['area'] += df['Rm' + str(i) + '_len'] * df['Rm' + str(i) + '_wth']
    double_rm = (df[constants.RM_LEN_WTH] != 0).sum(axis=1)
    df['rm_num'] = double_rm / 2.0
    df = df[df['rm_num'] != 0]
    df['ave_area'] = df['area'] / df['rm_num']

    # Reset index
    df.index = range(len(df))
    for i, area in enumerate(df['ave_area']):
        if (area > 1) & (area < 100):
            df.loc[i, 'Rooms_total_area'] = df.loc[i, 'area']
        elif (area >= 100) & (area < 700):
            df.loc[i, 'Rooms_total_area'] = df.loc[i, 'area'] / 10.7584  # 3.28 * 3.28
        elif (area >= 700) & (area < 8000):
            df.loc[i, 'Rooms_total_area'] = df.loc[i, 'area'] / 100.0
        elif (area >= 8000) & (area < 22500):
            df.loc[i, 'Rooms_total_area'] = df.loc[i, 'area'] / 1075.84  # 32.8 * 32.8
        else:
            df.loc[i, 'Rooms_total_area'] = df.loc[i, 'rm_num'] * 25.0

    df.drop(columns=['area', 'rm_num', 'ave_area'], inplace=True)
    df.index = range(len(df))
    df_area = pd.DataFrame(0, index=np.arange(len(df)), columns=constants.DISCRETE_ROOM_AREA)
    df = pd.concat([df, df_area], axis=1)

    for i, area in enumerate(df['Rooms_total_area']):
        if area < 50:
            df.loc[i, 'less-than50'] = 1
        elif (area >= 50) & (area < 100):
            df.loc[i, '50-to-100'] = 1
        elif (area >= 100) & (area < 150):
            df.loc[i, '100-to-150'] = 1
        elif (area >= 150) & (area < 200):
            df.loc[i, '150-to-200'] = 1
        elif (area >= 200) & (area < 250):
            df.loc[i, '200-to-250'] = 1
        elif (area >= 250) & (area < 350):
            df.loc[i, '250-to-350'] = 1
        else:
            df.loc[i, 'larger-than350'] = 1

    df.drop(columns='Rooms_total_area', inplace=True)
    df.drop(columns=constants.RM_LEN_WTH, inplace=True)
    return df


def drop_cols(df):
    df.drop(columns=['Lsc', 'S_r'], inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(keep='last', inplace=True)
    return df


def clean_whole_data(df_raw_data=None, raw_data_file=None, lsc='Sld', s_r='Sale'):
    if df_raw_data is not None:
        df_data = df_raw_data
    elif raw_data_file is not None:
        df_data = pd.read_csv(raw_data_file, sep=',')
    else:
        print("No data / data file to clean!")
    # Select house records: type is house & house-related columns
    df_hs = \
        df_data.loc[
            (df_data['Type_own1_out'].isin(constants.HS_LABEL)) & (df_data['Lsc'] == lsc) & (df_data['S_r'] == s_r)][
            constants.COLUMNS_HS]
    if os.path.isfile(DataCleanServiceConfig.COMM_FILE):
        os.remove(DataCleanServiceConfig.COMM_FILE)
    if os.path.isfile(DataCleanServiceConfig.CLEAN_DATA_MEDIAN_FILE):
        os.remove(DataCleanServiceConfig.CLEAN_DATA_MEDIAN_FILE)
    print("Start select_related_rows...")
    df_hs = select_related_rows(df_hs, prefix='Sold')

    print("Start complement null...")
    df_hs = complement_null(df_hs, 0, 0)

    print("Start process_cols...")
    df_hs = process_cols(df_hs, None)

    print("Start process_date...")
    df_hs = process_date(df_hs)

    print("Start calculate rooms_total_area...")
    df_hs = rooms_total_area(df_hs)

    drop_cols(df_hs)
    print("Sorting date...")
    df_hs['Cd'] = pd.to_datetime(df_hs.Cd)
    df_hs.sort_values(by=['Cd'], ascending=True, inplace=True)

    print(len(df_hs))
    # TODO:
    # Change file name
    df_hs.to_csv(DataCleanServiceConfig.CLEAN_HOUSE_DATA, index=False)
    # df_hs.to_csv(DataCleanServiceConfig.DATA_PATH + 'clean_data_515.csv', index=False)
    return df_hs


def clean_house_increment_data(df_hs, prefix):
    if len(df_hs) == 0:
        return None

    print("Start select_related_rows...")
    df_hs = select_related_rows(df_hs, prefix)
    if len(df_hs) == 0:
        return None

    if os.path.isfile(DataCleanServiceConfig.CLEAN_DATA_MEDIAN_FILE):
        print("Start complement null...")
        df_median = pd.read_csv(DataCleanServiceConfig.CLEAN_DATA_MEDIAN_FILE)
        depth_median = df_median['depth_median'].values[0]
        front_median = df_median['front_median'].values[0]
        print(depth_median, front_median)
        df_hs = complement_null(df_hs, depth_median, front_median)
    else:
        print("Error! No median file found!")
        return

    if os.path.isfile(DataCleanServiceConfig.COMM_FILE):
        comm_list = pd.read_csv(DataCleanServiceConfig.COMM_FILE)['Comm'].values.tolist()
        df_hs = process_cols(df_hs, comm_list)
    else:
        print('Error! No community file found!')

    df_hs = process_date(df_hs)

    df_hs = rooms_total_area(df_hs)
    drop_cols(df_hs)

    if len(df_hs) == 0:
        return None

    print("Sorting date...")
    df_hs['Cd'] = pd.to_datetime(df_hs.Cd)
    df_hs.sort_values(by=['Cd'], ascending=True, inplace=True)
    return df_hs


def clean_condo_increment_data(df_cd, prefix):
    df_cd['Cd'] = pd.to_datetime(df_cd['Cd'])
    df_cd.sort_values(by=['Cd'], ascending=True, inplace=True)
    df_cd.drop_duplicates(keep="last", inplace=True)
    if len(df_cd) == 0:
        return None

    if prefix == 'Sold':
        df_cd = df_cd[df_cd['Sp_dol'] > 50000]
        if len(df_cd) == 0:
            return None
        df_cd = df_cd[df_cd.Sp_dol.notna()]
        if len(df_cd) == 0:
            return None
    elif prefix == 'Listing':
        df_cd = df_cd[df_cd['Lp_dol'] > 50000]
        if len(df_cd) == 0:
            return None
        df_cd.drop(columns=['Sp_dol'], inplace=True)
    return df_cd


def clean_increment_data(raw_data, prefix, s_r='Sale'):
    df_data = pd.read_csv(raw_data, sep=',', compression='gzip')
    # Pre-process Condo
    df_cd = df_data.loc[
        (df_data['Type_own1_out'].isin(constants.CD_LABEL)) & (df_data['S_r'] == s_r)][
        constants.COLUMNS_CD]
    if len(df_cd) == 0:
        df_cd = None
    else:
        df_cd = clean_condo_increment_data(df_cd, prefix)
        if df_cd is not None:
            df_cd.to_csv(remove_gz_suffix_for_condo(raw_data), index=False)

    # Pre-process house
    df_hs = df_data.loc[
        (df_data['Type_own1_out'].isin(constants.HS_LABEL)) & (df_data['S_r'] == s_r)][
        constants.COLUMNS_HS]
    if len(df_hs) == 0:
        df_hs = None
    else:
        df_hs = clean_house_increment_data(df_hs, prefix)
        if df_hs is not None:
            # convert file to csv and remove suffix '.gz'
            df_hs.to_csv(remove_gz_suffix(raw_data), index=False)
    return df_cd, df_hs


if __name__ == "__main__":
    '''
    CASE 1
    '''
    # folder_path = '/Users/kristyx/Downloads/'
    # os.chdir(folder_path)
    # for counter, file in enumerate(glob.glob('*.csv.gz')):
    #     file_path = folder_path + str(file)
    #     print(type(file_path))
    #     clean_increment_data(file_path, 'Sold')

    '''
    CASE 2
    '''
    # file_path = '/var/qindom/realmaster/csv_file/Sold#20190313000001.csv.gz'
    # clean_increment_data(file_path, 'Sold')
    '''
    CASE 3
    '''
    # previous_data_path = DataCleanServiceConfig.DATA_PATH + 'part_5_Sold#20190314110046.csv'
    # increment_data_path = DataCleanServiceConfig.DATA_PATH + 'realmaster_data.csv'
    # df_part = pd.read_csv(previous_data_path)[constants.COLUMNS_HS]
    # df_inc = pd.read_csv(increment_data_path, sep=';')[constants.COLUMNS_HS]
    # print("**************************************")
    # print(len(df_inc))
    # df = pd.concat([df_part, df_inc])
    # df['Cd'] = pd.to_datetime(df.Cd)
    # df.dropna(subset=['Cd'], inplace=True)
    # df.sort_values(by=['Cd'], ascending=True, inplace=True)
    # print(df['Cd'])
    whole_raw_data_path = DataCleanServiceConfig.DATA_PATH + 'raw_data_190514.csv'
    # df.to_csv(whole_raw_data_path, index=False)

    clean_whole_data(whole_raw_data_path)
