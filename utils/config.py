from easydict import EasyDict as edict
import yaml
import os
import argparse

def get_config(config_file, args=None):
    conf = edict()
    if args is not None:
        for key in vars(args):
            setattr(conf, key, getattr(args, key))

    f = open(config_file, encoding="utf-8").read()
    yml = yaml.safe_load(f)

    for seg, param_list in yml.items():
        if isinstance(param_list, list):
            conf[seg] = {}
            for key_value in param_list:
                for key, value in key_value.items():
                    conf[seg][key] = value
        else:
            setattr(conf, seg, param_list)

    # # ------------------- conf init set ---------------------
    if hasattr(conf, "DEBUG"):
        conf["print_freq"] = 10 if conf["DEBUG"] else 100
    return conf

