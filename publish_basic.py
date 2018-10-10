#! /usr/bin/env python
import sys
import warnings
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', 
					'--today', 
					nargs   = '?',
					default = 'today', 
					help    = 'Determine the timestamp to query and process data.')
parser.add_argument('-s', 
					'--storage', 
					nargs   = '?', 
					default = None, 
					help    = 'Specifiy storage, either s3 or local.')
parser.add_argument('module', 
					nargs   = '?', 
					default = 'evergreen', 
					help    = 'Specify the module to run.')
args = parser.parse_args()


if not sys.warnoptions:
    warnings.simplefilter("ignore")

import config
import pandas as pd
import MySQLdb
import datetime
from   storage import create_storage

host = config.load_config(today   = args.today, 
                          module  = args.module, 
						  storage = args.storage)

# get names of modules whose 'publishable' is true from the configuration file
publishable_modules = [name for name in host.modules.keys() 
                                 if name != '_default' and not host.modules[name].publishable]

# load output of publishable modules to a single DataFrame
fs     = create_storage(host)
result = None
trends = []
print('Process module output(s)...')
for module in publishable_modules:
    module_config = config.load_config(today   = args.today, 
	                                   module  = module, 
									   storage = args.storage)
    filename      = module_config.paths.classification_output_data.format(module)
    if not fs.exists(filename):
        print('\tERROR: output of the module {} is not found: {}'.format(module, filename))
        quit()
		
    trends        = trends + host.modules[module].trends
    module_output = fs.read_csv(filename)
    del module_output['Other']
    if result is None:
        result    = module_output
    else:
        result    = pd.merge(result, 
		                     module_output, 
							 on = 'sku')
    print('\tModule {} processed: {} row(s) loaded.'.format(module, len(module_output)))

# apply filter for monochrome
# Monochrome sku should not be classified as other trends
monochrome         = 'Monochrome'
monochrome_idx     = trends.index(monochrome)
if monochrome_idx >= 0:
    mono_exclusive_trends = [monochrome, 'Floral', 
	                                     'Stripes', 
										 #'Denim', 
										 #'Athleisure', 
										 'Nude and Neutrals']
    print('Apply Monochrome restriction...')
    result[monochrome]    = result[mono_exclusive_trends].apply(lambda r: 1 
																    if r[0] == 1 and sum(r) == 1 
																    else 0, 
														        raw  = True, 
															    axis = 1)
 
print('Detect skus without trend...')
# create column Others
result['Other'] = result[trends].apply(lambda r: 0 
                                            if sum(r) > 0 
											else 1, 
									   raw  = True, 
									   axis = 1)

# load the unlabled data
print('Read sku information: {}'.format(host.paths.unlabeled_data))

sku_info       = fs.read_csv(host.paths.unlabeled_data)
print('\t{} row(s) loaded'.format(len(sku_info)))

del sku_info['summary']
published_data = pd.merge(sku_info, result, on='sku')

if monochrome in trends:
   print('Apply Monochrome color filter...')
   MONOCHROME_COLORS = ['black', 'white', 
						'white/black', 'black/white', 
						'black/black', 'black / black', 
						'white/white', 'white / white', 
						'natural white', 
						'off white', 
						'ivory']
   published_data[monochrome] = published_data.apply(lambda r: 1 
														if (str(r[monochrome]) == '1') 
															and (str(r['color']).lower() in MONOCHROME_COLORS ) 
														else 0, 
													 raw  = True, 
													 axis = 1)
   print('Updating skus without trend...')
   published_data['Other']    = published_data[trends].apply(lambda r: 0 
                                                                  if sum(r) > 0 
																  else 1, 
														     raw  = True, 
															 axis = 1)

print('Serializing data...')

from sqlalchemy import create_engine

timestamp =  datetime.date.today().strftime('%Y-%m-%d')

def publish_data_to_frame() -> pd.DataFrame:
    trends_with_other = trends + ['Other']
    total_rows        = len(published_data)
    exported_rows     = 0

    result            = pd.DataFrame(columns = ['sku', 
												'trend', 
												'updated_at', 
												'country', 
												'description', 
												'age', 
												'image'])

    for idx in range(0, total_rows):
        row         = published_data.iloc[idx]
        description = '' if row.short_description is None 
		                 else str(row.short_description)
						 
        for trend in [x for x in trends_with_other if row[x] == 1]:            
            args      = { 'sku': row.sku, 
			              'trend': trend, 
					      'updated_at': timestamp,
                          'country': row.country, 
						  'description': description,
                          'age': row.days_since_activation_date, 
						  'image': row.image_url 
					    }
            result    = result.append(args, 
			                          ignore_index = True)
        exported_rows = exported_rows + 1
        print('\r\t{:d}/{:d} ({:6.1f}%)....'.format(exported_rows, 
		                                            total_rows, 
													exported_rows * 100.0 / total_rows), 
													end = ''
													)
    print('\r\t{} rows are generated.'.format(len(result)))
    return result

exported_rows = 0

def create_trend_columns():
    trends_with_other = trends + ['Other']
    total_rows        = len(published_data)   
    def calc_trend(row):
        global exported_rows
        trend         = ', '.join([x for x in trends_with_other 
									    if row[x] == 1])
        exported_rows = exported_rows + 1
        print('\r\t{:d}/{:d} ({:6.1f}%)....'.format(exported_rows, 
		                                            total_rows, 
													exported_rows * 100.0 / total_rows), 
			  end  = '' 
			  )
        return trend

    published_data['trend'] = published_data.apply(calc_trend, axis = 1)
    print('')
    return published_data
 
# combine all classified trends to a column
# published_data = create_trend_columns()

print('Publishing data to temporary table...', end = '', flush = True)

published_data.reset_index(drop = True)
dbconfig = host.server.outdb
engine   = create_engine('mysql://{}:{}@{}/{}?charset=utf8'.format(dbconfig.userid, 
                                                                   dbconfig.password, 
																   dbconfig.host, 
																   dbconfig.schema)
																   )
fs.write_csv(published_data, 'data/published_data.csv')


if False:
    published_data.to_sql('temp_published_data', engine, if_exists='replace')

    print('\nTransfer data to database...', end='')

    with open('config/cleanup_data.sql', 'r') as f:
        clean_up_sql = f.read()

    with open('config/publish_data.sql', 'r') as f:
        sql          = f.read()

    with engine.connect() as con:
        print("\nCleanup target data on date: {}...".format(timestamp), end='')
        con.execute(clean_up_sql.format(timestamp))
        trends_with_other = trends + ['Other']
        for trend in trends_with_other:
            print('\nPublish data for trend: {}...'.format(trend), end='')
            con.execute(sql.format(trend, timestamp))

print('\nComplete')
