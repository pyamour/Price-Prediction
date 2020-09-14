import glob
import os


def remove_filename(fname):
    exists = os.path.isfile(fname)
    if exists:
        os.remove(fname)


def remove_local_file(path, fname):
    remove_filename(path + fname)


def remove_gz_suffix(name):
    if name.endswith('.gz'):
        return name[:-3]
    else:
        return name


def remove_gz_suffix_for_condo(name):
    if name.endswith('.gz'):
        new_name = name[:-7] + '_condo.csv'
        return new_name
    else:
        new_name = name[:-4] + '_condo.csv'
        return new_name


def remove_files_in_dir(file_path, expression):
    file_list = glob.glob(os.path.join(file_path, expression))
    for file in file_list:
        os.remove(file)
