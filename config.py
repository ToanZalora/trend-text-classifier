"""
Utility to read configuration file into memory
"""
from   datetime import datetime
import json
import re
import os
import yaml
from   objdict  import ObjDict

def expand_keys(section, value: str):
    pattern   = re.compile('({(\w+)})')
    start_pos = 0
    while True:
        m     = pattern.search(value, 
		                       pos  =  start_pos)
        if m is None: 
            break
        key = m.group(2)
        if key in section.keys():
            value = value.replace(m.group(0), 
			                      str(section[key]))
        else:
            start_pos = m.endpos
    return value

def load_config(filename = './config/config.yml', 
                today    = None, 
				module   = 'evergreen', 
				storage  = None):
    if not os.path.isfile(filename):
        raise Exception('Configuration file not found: {}'.format(filename))
    with open(filename) as file:
        config =  yaml.load(file)
    config     = ObjDict(json.dumps(config))

    runtime    = config.runtime
    runtime.module  = module or 'default'
    runtime.storage = storage or runtime.storage
    
    if today:
        config.runtime.timestamp = today 
    if config.runtime.timestamp == 'today':
        config.runtime.timestamp = datetime.today().strftime('%Y.%m.%d')
		
    timestamp  = config.runtime.timestamp
    runtime.folder_timestamp = timestamp.replace(".", "")

    paths      = config.paths
    paths.root = paths[runtime.storage + '_root']
    for key in paths.keys():
        paths[key] = expand_keys(runtime, expand_keys(paths, paths[key]))
    return config
