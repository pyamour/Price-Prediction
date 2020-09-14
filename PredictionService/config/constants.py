from multiprocessing import cpu_count

MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

DISCRETE_ROOM_AREA = ['less-than50', '50-to-100', '100-to-150', '150-to-200', '200-to-250', '250-to-350',
                      'larger-than350']

MANUAL_RULES = {'373 Front St W:1.0:1.0:1.0': 575000, '560 Front St W:1.0::1.0': 590000,
                '257 Hemlock St:1.0::1.0': 340000}


INTERVAL = 20
RATIO = 0.1
KFOLD = 10
FIRST_PAIR_SAMPLE_NUM = 1428570
CORE = cpu_count()


# TODO: should get it automatically
# NUM_DIM = 27
# NUM_DIM = 29

# NUM_ZERO_COL = 6
NUM_ZERO_COL = 9

# NUM_STACK = 500
NUM_STACK = 7000
