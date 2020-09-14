import pandas
from DataClean.config import constants


def validate(path, filename):
    fname = path + filename
    df = pandas.read_csv(fname, compression='gzip')
    columns = df.columns.values
    if len(df) < 1:
        return False
    for column in constants.COLUMNS_HS:
        if column not in columns:
            return False
    for column in constants.COLUMNS_CD:
        if column not in columns:
            return False
    return True
