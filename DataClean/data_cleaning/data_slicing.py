from DataClean.config import constants
import pandas as pd
from datetime import datetime
from DataClean.config import DataCleanServiceConfig


def slice_data(rawdata, interval=constants.INTERVAL):
    rawdata['Cd'] = pd.to_datetime(rawdata['Cd'])
    # TODO
    rawdata = rawdata[rawdata['Cd'] >= '2018-01-01']
    rawdata.sort_values(by=['Cd'], ascending=True, inplace=True)  # long long ago -> now
    start_date = datetime.strptime('2000-01-01', '%Y-%m-%d')
    cols = rawdata.columns.values.tolist()
    cur_df = pd.DataFrame(columns=cols)
    i = 0
    for index, row in rawdata.iterrows():
        cur_date = row['Cd']
        delta = cur_date - start_date
        if delta.days >= interval:
            if int(cur_df.shape[0]) > 0:
                i += 1
                cur_df.to_csv(DataCleanServiceConfig.SLICE_DATA + str(i) + '.csv', index=None)
            start_date = cur_date
            cur_df = pd.DataFrame(columns=cols)
        cur_df = pd.concat([cur_df, pd.DataFrame(row).transpose()], ignore_index=True)
    i += 1
    cur_df.to_csv(DataCleanServiceConfig.SLICE_DATA + str(i) + '.csv', index=None)
    return i


if __name__ == "__main__":
    data_file = DataCleanServiceConfig.DATA_PATH + 'clean_house_data.csv'
    df_data = pd.read_csv(data_file)
    df_data = df_data[(df_data['Cd'] >= '2018-01-01') & (df_data['Cd'] < '2019-01-13')]
    slice_data(df_data)
