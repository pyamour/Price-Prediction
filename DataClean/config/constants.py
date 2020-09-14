from multiprocessing import cpu_count

COLUMNS_HS = ['_id', 'Dom', 'A_c', 'Area_code', 'Bath_tot', 'Br', 'Br_plus', 'Bsmt1_out', 'Community',
              'Depth', 'Front_ft', 'Gar', 'Gar_spaces', 'Park_spcs', 'Gar_type', 'Heating', 'Kit_plus',
              'Lp_dol', 'Sp_dol', 'Lsc', 'Pool', 'Rms', 'Rooms_plus', 'S_r', 'Style', 'Type_own1_out', 'Cd', 'lat',
              'lng', 'Taxes', 'Den_fr',
              'Rm1_len', 'Rm1_wth', 'Rm2_len', 'Rm2_wth', 'Rm3_len', 'Rm3_wth', 'Rm4_len', 'Rm4_wth', 'Rm5_len',
              'Rm5_wth', 'Rm6_len', 'Rm6_wth', 'Rm7_len', 'Rm7_wth', 'Rm8_len', 'Rm8_wth', 'Rm9_len', 'Rm9_wth',
              'Rm10_len', 'Rm10_wth', 'Rm11_len', 'Rm11_wth', 'Rm12_len', 'Rm12_wth']

COLUMNS_CD = ['_id', 'Br', 'Br_plus', 'Lp_dol', 'Sp_dol', 'Cd', 'Apt_num', 'Addr', 'Bath_tot', 'Taxes', 'Yr_built',
              'lat', 'lng']

HS_LABEL = ['Detached', 'Semi-Detached', 'Duplex', 'Triplex', 'Fourplex', 'Link', 'Att/Row/Twnhouse', 'Det Condo',
            'Semi-Det Condo', 'Condo Townhouse']

CD_LABEL = ['Condo Apt', 'Comm Element Condo', 'Co-Ownership Apt', 'Co-Op Apt', 'Leasehold Condo', 'Vacant Land Condo',
            'Condo Apartment', 'Phased Condo']

# TODO: condo house
CDHS_LABEL = ['Condo Townhouse', 'Semi-Det Condo', 'Det Condo']

CMPLMT_NONE_COL = ['A_c', 'Pool', 'Bsmt1_out', 'Gar_type']

CMPLMT_ZERO_COL = ['Br_plus', 'Kit_plus', 'Rooms_plus', 'Gar', 'Gar_spaces', 'Park_spcs', 'Rm1_len', 'Rm1_wth',
                   'Rm2_len', 'Rm2_wth', 'Rm3_len',
                   'Rm3_wth', 'Rm4_len', 'Rm4_wth', 'Rm5_len', 'Rm5_wth', 'Rm6_len', 'Rm6_wth', 'Rm7_len', 'Rm7_wth',
                   'Rm8_len', 'Rm8_wth', 'Rm9_len', 'Rm9_wth', 'Rm10_len', 'Rm10_wth', 'Rm11_len', 'Rm11_wth',
                   'Rm12_len', 'Rm12_wth', 'Dom']

RM_LEN_WTH = ['Rm1_len', 'Rm1_wth', 'Rm2_len', 'Rm2_wth', 'Rm3_len', 'Rm3_wth', 'Rm4_len', 'Rm4_wth', 'Rm5_len',
              'Rm5_wth', 'Rm6_len', 'Rm6_wth', 'Rm7_len', 'Rm7_wth', 'Rm8_len', 'Rm8_wth', 'Rm9_len', 'Rm9_wth',
              'Rm10_len', 'Rm10_wth', 'Rm11_len', 'Rm11_wth', 'Rm12_len', 'Rm12_wth']

MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

DISCRETE_ROOM_AREA = ['less-than50', '50-to-100', '100-to-150', '150-to-200', '200-to-250', '250-to-350',
                      'larger-than350']

DATA_PATH = '/var/csv_file/download/'  # file_location

INTERVAL = 20
RATIO = 0.1
N_THREAD = 4
RAND = 5
COMM_TH = 12
CORE = cpu_count()
KFOLD = 10
