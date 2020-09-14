from configparser import ConfigParser

config = ConfigParser()
config.read('/var/DataCleanService.conf')

aws_access_key_id = config.get('s3Section', 'aws_access_key_id')
aws_secret_access_key = config.get('s3Section', 'aws_secret_access_key')
test_price_prediction_bucket = config.get('s3Section', 'test_price_prediction_bucket')
real_price_prediction_bucket = config.get('s3Section', 'real_price_prediction_bucket')
internal_price_prediction_bucket = config.get('s3Section', 'internal_price_prediction_master_bucket')
