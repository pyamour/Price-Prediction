from configparser import ConfigParser

config = ConfigParser()
config.read('/var/PredictionService.conf')

aws_access_key_id = config.get('s3Section', 'aws_access_key_id')
aws_secret_access_key = config.get('s3Section', 'aws_secret_access_key')
test_real_master_bucket = config.get('s3Section', 'test_price_prediction_bucket')
real_price_prediction_bucket = config.get('s3Section', 'real_price_prediction_bucket')
internal_price_prediction_bucket = config.get('s3Section', 'internal_price_prediction_bucket')
price_prediction_results_bucket = config.get('s3Section', 'price_prediction_results_bucket')

datalake_price_bucket = 'datalake-price'
