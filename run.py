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
parser.add_argument('-m', 
                    '--mode', 
                    nargs   = '?', 
                    default = 'default', 
                    help    = 'Determine the running mode: default or suggestion.')
parser.add_argument('--color', 
                    nargs   = '?', 
                    default = None, 
                    help    = 'Specify the CSV file containing color of skus.')
parser.add_argument('--config', 
                    nargs   = '?', 
                    default = 'config/config.yml', 
                    help    = 'Specify the configuration file, default is config/config.yml.')
parser.add_argument('-n', 
                    '--name', 
                    nargs   = '?', 
                    default = None, 
                    help    = 'Specifiy the experiment/production name for the outcome.')
parser.add_argument('-s', 
                    '--storage', 
                    nargs   = '?', 
                    default = None, 
                    help    = 'Specifiy storage, either s3 or local.')
parser.add_argument('-c', 
                    '--classify', 
                    nargs   = '?', 
                    default = None, 
                    help    ='Specify CSV file containing list of SKU for classification.')
parser.add_argument('-e', 
                    '--exclude', 
                    nargs   = '?', 
                    default = None, 
                    help    = 'Specify CSV file containing list of SKU to exclude from classification.')
parser.add_argument('-f', 
                    '--force-regenerate-data', 
                    action  = 'store_true', 
                    default = False, 
                    help    = 'Regenerate input data even if they exist.')
parser.add_argument('module', 
                    nargs   = '?', 
                    default = 'evergreen', 
                    help    = 'Specify the module to run.')
args = parser.parse_args()

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import config
import classifier
import models 
import experiments
from   objdict      import ObjDict 
from   prepare_data import prepare_data

print('today = {}, module = {}, storage = {}'.format(args.today, 
                                                     args.module, 
                                                     args.storage))

cfg = config.load_config(filename = args.config, 
                         today    = args.today, 
                         module   = args.module, 
                         storage  = args.storage)
cfg.suggestion_mode = args.mode.startswith('s')

name                = args.name or args.module
module              = cfg.modules[args.module]

# copy settings from _default module for non-existed setting keys
default_module      = ObjDict(cfg.modules._default) 
module_keys         = module.keys()
for key in default_module.keys():
    if not key in module_keys:
        module[key] = default_module[key]
# makeup the data
prepare_data(cfg, args)

trend_classifier = classifier.create_classifier(cfg, args.module)

if trend_classifier is None:
    quit()


args = {}
if 'conv_output_dimension' in module.keys():
    args     = ObjDict( conv_output_dimension = module.conv_output_dimension,
                        kernel_size           = module.kernel_size,
                        pool_size             = module.pool_size,
                        dense_layer_count     = module.layer_count,
                        dense_layer_size      = module.layer_size,
                        dropout               = module.dropout,
                        optimizer             = 'adam',
                        layer_activation      = 'relu',
                        output_activation     = 'sigmoid'
                       )

model, score = trend_classifier.build_model(name, 
                                            models.model3, 
                                            **args)
trend_classifier.classify(name, model)

## Experient mode
""" elif args.mode.startswith('e'):     # experimental mode
    model_args = experiments.model3_arg_gen(dense_layer_size      = [128,256,512,768,1024,1152],
                                            kernel_size           = [5, 7, 10, 13, 15, 17, 20],
                                            pool_size             = [3,4,5,6,7,8],
                                            dense_layer_count     = [2,3,4,5,6],
                                            dropout               = [0.2, 0.3, 0.4, 0.5],
                                            conv_output_dimension = [20, 30, 40, 50],
                                            optimizer             = ['adam', 'nadam'])
    result = experiments.experiment("model3", 
                                    trend_classifier, 
                                    models.model3, 
                                    model_args)
    print(result)
 """
    # 'adam', 'adadelta', 'RMSprop', 'Adagrad', 'adamax', 'nadam': best are adam, nadam
