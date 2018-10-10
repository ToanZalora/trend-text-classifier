import config 
import pandas as pd
import io
import psycopg2
# import mysql.connector
import storage
import re
import numpy as np

from   nltk        import word_tokenize
# from nltk.stem.snowball import SnowballStemmer
from   nltk.corpus import stopwords
from   objdict     import ObjDict

CHARS_TO_REMOVE     = [ '=-', '<b>', ';', '<', '-', '/', '@', '>', '.', '\"', 
                        '&nbsp', ',', '(', ')', '?', ':', '%', '{', '}', '&', '\u2122', '\u2019']
IGNORE_WORDS        = [ 'size', 'side','print', 'prints', 'printed', 'printing','pocket', 'pockets','casual',
                         'sleeve', 'sleeveless', 'ruffle', 'ruffled', 'embroidery', 'embroidered',  
                         'graphic', 'graphics', 'tone', 'toned', 'pattern', 'patterned', 'zip','fastening',
                         'detail', 'details', 'regular', 'fit', 'solid', 
                         'b', 's', 'y', 'x', 'c', 'd', 'e', 'f', 'v' ]
FEATURE_COLUMNS     = ['brand', 
                       'department',
                       'buying_planning_cat_type',
                       'sub_cat_type', 
                       'occasion',
                       'color', 
                       'color_family',
                       'product_name', 
                       'short_description',
                       'catalog_attribute_set_name']

# stemmer             = SnowballStemmer("english", ignore_stopwords=False)
STOP_WORDS            = set(stopwords.words('english'))
NON_CHARACTER_PATTERN = re.compile('[^a-zA-Z,\s\r\n]')

def filter_data(text):
    s  = str(text)
    for char in CHARS_TO_REMOVE:
        s = s.replace(char, ' ')
    return s

def stemSentences(sentence: str):
    words = [w for w in word_tokenize(sentence.lower()) 
                   if w not in STOP_WORDS]
    return ' '.join(list(set([w for w in words 
                                    if w not in IGNORE_WORDS])))

def pitbull(cfg):
    return psycopg2.connect(host     = cfg.host, 
                            database = cfg.schema, 
                            user     = cfg.userid,
                            password = cfg.password, 
                            port     = cfg.port)

# def mysqldb(cfg):
#    return mysql.connector.connect(user=cfg.userid, password=cfg.password, 
#                                   host=cfg.host, database=cfg.schema)

def identity(x):
    return x

def strip_trend_tagging(x: str) -> str:
    try:
        idx = x.index('/* TREND_TAGGING_FILTER')
        return x[:idx]
    except ValueError:
        return x

def exec_sql_from_file(filename, 
                       s3, 
                       cache        = None, 
                       server       = pitbull, 
                       cfg          = None, 
                       sql_formater = identity):
    # check cache file exist
    if ((cache is not None) and s3.exists(cache)):
        print('\tCache file found: {}'.format(cache))
        return s3.read_csv(cache)
    
    with open(filename, 'r') as f:
        sql = sql_formater(f.read())
    conn    = server(cfg)
    df      = pd.read_sql_query(sql, conn)
    conn.close()
    if (cache is not None):
        s3.write_csv(df, cache)
    return df

def prepare_data(config: ObjDict, 
                 args):
    paths = config.paths
    s3    = storage.create_storage(config)    
    s3.makedirs(paths.raw_data_folder)
    s3.makedirs(paths.input_folder)
    s3.makedirs(paths.output_folder)
    ignore_existing     = args.force_regenerate_data
    classification_file = args.classify

    suggestion_mode     = config.suggestion_mode

    module        = config.runtime.module
    module_config =  config.modules[module]
    trends        = module_config.trends 

    if 'query' in module_config.keys():
        print('Classification data is taken from query...: {}'.format(module_config.query))
        exec_sql_from_file(module_config.query, 
                           s3, 
                           paths.unlabeled_data, 
                           cfg = config.server.pitbull)
        return        

    if (not ignore_existing) and s3.exists(paths.sanitized_data) and s3.exists(paths.unlabeled_data):
        print('Sanitized data found at: {0}'.format(paths.sanitized_data))
        print('Unlabeled data found at: {0}'.format(paths.unlabeled_data))
        return

    # print("Query suggestion data...")
    # suggestion_data = exec_sql_from_file(paths.suggestion_data_query,
    #                                   s3,
    #                                   None if suggestion_mode else paths.cached_suggestion_data ,
    #                                   server=mysqldb,
    #                                   cfg = config.server.mysql,
    #                                   sql_formater= strip_trend_tagging if suggestion_mode else identity)
    # print("\t{} rows loaded".format(len(suggestion_data)))

    if not suggestion_mode:
        print("Query trend data...")
        labeled_data = exec_sql_from_file(paths.trend_data_query,
                                          s3,
                                          paths.cached_trend_data,
                                          # server=mysqldb,
                                          server = None,
                                          cfg    = config.server.mysql)
        # perform trend-name remapping
        mapping           = config.trend_mapping
        labeled_data.name = labeled_data.name.apply(lambda n: mapping[n])

        print("\t{} rows loaded".format(len(labeled_data)))

        print('Connect trend data with suggestion data...')

    #    labeled_data = labeled_data.append(suggestion_data, ignore_index= True)
    else:
        print('Suggestion-only mode is enabled, trend-data is skipped.')
    #    labeled_data = suggestion_data

    labeled_data.reset_index()
    print("\t{} rows after concatination".format(len(labeled_data)))

    print('Active module: {}'.format(module))
    labeled_data.name = labeled_data.name.apply(lambda x: x 
                                                   if x in trends 
                                                   else 'Other')

    all_trends        = labeled_data.name.drop_duplicates()
    for col in all_trends:
        labeled_data[col] = labeled_data.name.apply(lambda x: 1 
                                                        if x == col 
                                                        else 0)
    
    del labeled_data['name']

    labeled_data          = labeled_data.groupby('sku').aggregate(max)
    labeled_data['sku']   = labeled_data.index.values
    
    labeled_data['Other'] = labeled_data[trends].apply(lambda x: 0 
                                                           if sum(x) > 0 
                                                           else 1, 
                                                       raw  = True, 
                                                       axis = 1)

    print("\tLabeled data processing is completed")

    print("Query sku data...")
    sku_data = exec_sql_from_file(paths.sku_data_query,
                                  s3,
                                  paths.cached_sku_data,
                                  server = pitbull,
                                  cfg    = config.server.pitbull)
    print("\t{} rows loaded".format(len(sku_data)))

    if args.color is not None and s3.exists(args.color):
        print("Recoloring sku data...")
        color_data = s3.read_csv(args.color)[['sku', 'colors']]
        sku_data = pd.merge(sku_data, color_data, how='left', on='sku')
        s3.write_csv(sku_data, './data/test.csv')
        recolor_count = sum(sku_data.apply(lambda r: 1 if r.colors is not None and r.colors != r.color else 0, axis=1))
        print("\t{} skus to recolor".format(recolor_count))
        print(sku_data.colors.unique())
        
        sku_data['color'] = sku_data.apply(lambda r: r['colors'] if isinstance(r.colors, str) else r['color'], axis=1 )
        recolor_count = sum(sku_data.apply(lambda r: 1 if r.colors is not None and r.colors != r.color else 0, axis=1))
        print("\t{} skus after recoloring".format(len(sku_data)))
        print("\t{} skus recolored".format(recolor_count))


    print("Generate sku information (text) from sku data...")
    columns = module_config.feature_columns  if 'feature_columns' in module_config  else FEATURE_COLUMNS
         
    sku_data['summary'] = sku_data[columns].apply(lambda x: filter_data(' '.join(x.apply(str))), 
                                                  axis=1
                                                  )
    sku_data.summary    = sku_data.summary.apply(stemSentences)

    sku_data.set_index('sku')

    output = pd.merge(sku_data[['sku', 'summary']], labeled_data, on='sku')

    print("Sanitize information...")
    # output['summary']   = output['summary'].apply(stemSentences)

    print("Write sanitized data...{}".format(paths.sanitized_data))
    s3.write_csv(output, paths.sanitized_data)

    print("Generate data for classification...")
    for_classification = sku_data[~sku_data.sku.isin(labeled_data.sku)][['sku', 
                                                                            'product_type',
                                                                            'brand',
                                                                            'color',
                                                                            'color_family',
                                                                            'sub_cat_type',
                                                                            'country', 
                                                                            'days_since_activation_date', 
                                                                            'image_url', 
                                                                            'short_description', 
                                                                            'summary']]
    print("\t{} rows generated".format(len(for_classification)))

    if classification_file:
        print("Restrict SKU in provided classification file")
        
        classification_file_data = s3.read_csv(classification_file)
        print("\tClassification file: {} rows".format(len(classification_file_data)))
        
        classification_file_sku  = np.asarray(classification_file_data.sku)
        selected_skus            = [sku in classification_file_sku 
                                            for sku in for_classification.sku ]
        for_classification       = for_classification.loc[selected_skus]
        print("\t{} rows after restriction".format(len(for_classification)))

    print("Filter skus lasting for more than 180 days...")
    for_classification           = for_classification.loc[for_classification.days_since_activation_date <= 180]
    
    print("\t{} rows after filtered".format(len(for_classification))) 

    print("Filter suggested-sku from classification...")
    # suggested_skus = np.asarray(suggestion_data.sku)
    # unsuggested_skus = [sku not in suggested_skus for sku in for_classification.sku]
    # unsuggested_skus = [sku not in suggested_skus for sku in for_classification.sku]

    # for_classification = for_classification[unsuggested_skus]
    print("\t{} rows after filtered".format(len(for_classification))) 

    if args.exclude:
        exclusion_data     = s3.read_csv(args.exclude)
        print("Exclude SKUs from {}".format(args.exclude))
        excluded_skus      = np.asarray(exclusion_data.sku)
        included_skus      = [sku not in excluded_skus 
                                        for sku in for_classification.sku]
        for_classification = for_classification[included_skus]
        print("\t{} rows after exclusion".format(len(for_classification))) 

    for_classification['summary'] = for_classification['summary'].apply(stemSentences)
    
    print('Write data for classification...{}'.format(paths.unlabeled_data))
    s3.write_csv(for_classification, paths.unlabeled_data)

    print('Data preprocessing is completed')
 
