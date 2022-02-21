import configparser
import os

root_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(root_dir, 'configs')
base_config_file = os.path.join(config_dir, 'global.cfg')

global_config = configparser.ConfigParser()
with open(base_config_file, 'r') as f:
    global_config.read_file(f)

global_config['directories']['root'] = os.path.dirname(os.path.abspath(__file__))
# check directories, convert to relative paths if necessary
for key, value in global_config['directories'].items():
    if not os.path.isabs(value):
        global_config['directories'][key] = os.path.join(global_config['directories']['root'], value)

def parse_config(config_path):
    config = configparser.ConfigParser()
    with open(config_path, 'r') as f:
        config.read_file(f)
    return config
