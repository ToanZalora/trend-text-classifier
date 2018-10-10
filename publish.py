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
parser.add_argument('-p', 
					'--push',
					action  = 'store_true', 
					default = False, 
					help    = 'Push published data to database.')
parser.add_argument('-c', 
					'--color', 
					nargs   = '?', 
					default = None, 
					help    = 'Path to the CSV define the SKU color.')
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
import os
from   storage       import create_storage
from   publish_utils import resolve_color

host = config.load_config(today   = args.today, 
						  module  = args.module, 
						  storage = args.storage)


# get names of modules whose 'publishable' is true from the configuration file
publishable_modules = [name for name in host.modules.keys() 
								if name != '_default' 
									and host.modules[name].publishable]

# load output of publishable modules to a single DataFrame
fs 		 = create_storage(host)
result 	 = None
sku_info = None
trends   = []

print('Process module output(s)...')
for module in publishable_modules:
    module_config = config.load_config(today   = args.today, 
									   module  = module, 
									   storage = args.storage)
    
    filename 	  = module_config.paths.classification_output_data.format(module)
    if not fs.exists(filename):
        print('\tERROR: output of the module {} is not found: {}'.format(module, filename))
        quit()
    
    trends        = trends + host.modules[module].trends
    module_output = fs.read_csv(filename)
    del module_output['Other']

    if result is None:
        result    = module_output
    else:
        # result  = result.merge(module_output, how='outer', on='sku')
        result    = result.merge(module_output, 
							     how = 'outer', 
								 on  = 'sku')
								 
    print('\tModule {} processed: {} row(s) loaded.'.format(module, len(module_output)))
	
    module_section = module_config.modules[module]
    if sku_info is None or 'sku_info' in module_section.keys():
        x          = fs.read_csv(module_config.paths.unlabeled_data)
        del x['summary']
        print('\t\t {} sku(s) loaded.'.format(len(x)))
        if sku_info is None:
            sku_info = x
        elif module_section.sku_info:
            new_sku_selector = [not sku in sku_info.sku.values for sku in x.sku]
            sku_info         = sku_info.append(x.loc[new_sku_selector], ignore_index = True)
            pass

print('Total: {} row(s) loaded.'.format(len(result)))
print('       {} sku(s) loaded.'.format(len(sku_info)))

result = result.fillna(0)
 
# apply trend exclusion
for trend in host.exclusive_trends.keys():
    if trend in result.keys():
        exclusive_trends = [ trend for trend in host.exclusive_trends[trend] if trend in result.keys()]
        print('Apply {} restriction...'.format(trend))
        result[trend]    = result[exclusive_trends].apply(lambda r: 1 
																if r[0] == 1 and sum(r) == 1 
																else 0, 
														  raw  = True , 
														  axis = 1    )

print('Detect skus without trend...')
# create column Others
result['Other'] = result[trends].apply(lambda r: 0 
											if sum(r) > 0 
											else 1, 
				                       raw  = True, 
									   axis = 1)
published_data  = pd.merge(sku_info, 
						   result, 
						   on = 'sku')

# apply monochrome color restriction
monochrome        = 'Monochrome'
monochrome_module = host.modules.monochrome

if (monochrome in trends) and monochrome_module.color_restriction:
   print('Apply Monochrome color filter...')
   MONOCHROME_COLORS = monochrome_module.accepted_colors
   external_color    = None
   if args.color and fs.exists(args.color):
       print('\tFound external color file: {}'.format(args.color))
       external_color         = fs.read_csv(args.color)
       external_color.index   = external_color.sku

   published_data[monochrome] = published_data.apply(lambda r: 1 
														if (str(r[monochrome]) == '1') 
															and (resolve_color(r, external_color) in MONOCHROME_COLORS ) 
														else 0  , 
													 raw  = True, 
												   	 axis = 1   )
   print('Updating skus without trend...')
   published_data['Other'] = published_data[trends].apply(lambda r: 0 
															 if sum(r) > 0 
															 else 1  , 
														  raw  = True, 
														  axis = 1   )

print('Serializing data...')

from sqlalchemy import create_engine

timestamp =  datetime.date.today().strftime('%Y-%m-%d')

def publish_data_to_frame() -> pd.DataFrame:
    trends_with_other = trends + ['Other']
    total_rows        = len(published_data)
    exported_rows     = 0

    result 			  = pd.DataFrame(columns = ['sku', 
												 'trend', 
												 'updated_at', 
												 'country', 
												 'description', 
												 'age', 
												 'image']
									)
    for idx in range(0, total_rows):
        row           = published_data.iloc[idx]
        description   = '' if       row.short_description is None else str(row.short_description)
        for trend in [x for x in trends_with_other if row[x] == 1]:            
            args      = { 'sku'	     : row.sku, 
						  'trend'      : trend, 
						  'updated_at' : timestamp,
						  'country'    : row.country, 
						  'description': description,
						  'age'        : row.days_since_activation_date, 
					      'image'      : row.image_url 
						 }
            result    = result.append(args, 
									  ignore_index = True)
        exported_rows = exported_rows + 1
        print( '\r\t{:d}/{:d} ({:6.1f}%)....'.format( exported_rows, 
													  total_rows, 
													  exported_rows * 100.0 / total_rows
													 ),
			   end = '' 
			  )
			 
    print(      '\r\t{} rows are generated.'.format(len(result)))

    return result

exported_rows = 0


def create_trend_columns():
    trends_with_other = trends + ['Other']
    total_rows        = len(published_data)   
    def calc_trend(row):
        global exported_rows
        trend         = ', '.join([x 
									for x in trends_with_other 
									   if row[x] == 1]
								  )
        exported_rows = exported_rows + 1
        print('\r\t{:d}/{:d} ({:6.1f}%)....'.format( exported_rows, 
												     total_rows, 
													 exported_rows * 100.0 / total_rows
													), 
			  end = ''
			  )
        return trend

    published_data['trend'] = published_data.apply(calc_trend, 
												   axis = 1)
    print('')
    return published_data

print('Create description...')
published_data['short_description'] = published_data.apply( lambda r: 'color: {}, color_family: {}, {}'.format( r['color'], 
																						  r['color_family'], 
																						  r['short_description']
																						 ), 
														    raw  = True, 
														    axis = 1
														   ) 

# combine all classified trends to a column
# published_data = create_trend_columns()

print( 'Publishing data to temporary table...', 
       end   = '', 
	   flush = True
	  )

published_data.reset_index(drop = True)
dbconfig     = host.server.outdb
engine       = create_engine('mysql://{}:{}@{}/{}?charset=utf8'.format( dbconfig.userid, 
																		dbconfig.password, 
																		dbconfig.host, 
																		dbconfig.schema )
							)
fs.write_csv( published_data, 
             'data/published_data.csv')


if True:
    published_data.to_sql( 'temp_published_data', 
	                       engine, 
						   if_exists = 'replace')

    print('\nTransfer data to database...', 
	      end = '')

    with open('config/cleanup_data.sql', 'r') as f:
        clean_up_sql = f.read()

    with open('config/publish_data.sql', 'r') as f:
        sql          = f.read()

    with engine.connect() as con:
        print( "\nCleanup target data on date: {}...".format(timestamp), 
		       end = ''
			 )
        con.execute(clean_up_sql.format(timestamp))
        trends_with_other = trends + ['Other']
        for trend in trends_with_other:
            print( '\nPublish data for trend: {}...'.format(trend), 
			       end    = ''
				  )
            con.execute(sql.format(trend, timestamp))

print('\nComplete')
