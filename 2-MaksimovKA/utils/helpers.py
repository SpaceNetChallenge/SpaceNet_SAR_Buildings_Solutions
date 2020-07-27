import os.path as osp
import sys
from importlib import import_module


def get_config(filename):
    module_name = osp.basename(filename)[:-3]
    
    config_dir = osp.dirname(filename)
    
    sys.path.insert(0, config_dir)
    mod = import_module(module_name)
    sys.path.pop(0)
    cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith('__')
            }
    return cfg_dict

