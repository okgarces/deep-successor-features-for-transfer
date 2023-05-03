# -*- coding: UTF-8 -*-
from ast import literal_eval
from collections import defaultdict
import configparser
from pathlib import Path

global_settings = {}

def parse_config_file(name):
    global global_settings

    raw_path = Path(__file__).parent.parent
    config_path = raw_path / 'configs' / name

    config = configparser.RawConfigParser()
    config.optionxform = str
    config.read(config_path)

    section_dict = defaultdict()
    for section in config.sections():
        section_dict[section] = {k: literal_eval(v) for k, v in config.items(section)}

    global_settings = section_dict
    return section_dict

def check_settings():
    if len(global_settings) == 0:
        raise Exception('Global settings is not initialized')

# TODO this could be removed
def use_torch():
    check_settings()
    return global_settings.get('GENERAL', {}).get('use_torch')

def use_logger():
    check_settings()
    return global_settings.get('GENERAL', {}).get('use_torch')
