import argparse
from typing import Optional, Dict
from pyhocon import ConfigFactory, ConfigTree
from pyhocon.converter import HOCONConverter

TRAIN_DEFAULT_CONFIG_FILE = "config/default_gpt2.conf"


def parse_config(config_dict=None,
                 config_file: Optional[str] = None,
                 default_file: str = TRAIN_DEFAULT_CONFIG_FILE):
    if config_dict is None:
        config_dict = dict()
    conf = ConfigFactory.from_dict(config_dict)
    conf = conf.with_fallback(config_file) if config_file else conf.with_fallback(default_file)
    return conf


def define_parser(parser):
    parser.add_argument("-f", "--config_file", type=str, default=None,
                        help="configuration file if presents overwrite default")
    parser.add_argument("-c", "--config", type=str, nargs="*",
                        help="individual config key=value, if presents overwrite config file values")


def parse_args():
    parser = argparse.ArgumentParser(description="nanGPT")
    define_parser(parser)
    args = parser.parse_args()
    return args


def config_args_to_dict(args):
    config_dict = {}
    if args.config:
        for item in args.config:
            kvs = item.split("=")
            assert (len(kvs) == 2)
            config_dict[kvs[0]] = kvs[1]
    return {"config": config_dict}


def load_config(default_file: str = TRAIN_DEFAULT_CONFIG_FILE) -> ConfigTree:
    args = parse_args()
    config_dict = config_args_to_dict(args)
    conf = parse_config(config_dict, args.config_file, default_file)
    return conf.get_config("config")


def _convert_conf_item(conf_item):
    result = {}
    if isinstance(conf_item, ConfigTree):
        if len(conf_item) > 0:
            for key, item in conf_item.items():
                new_key = key.strip('"')  # for dotted keys enclosed with "" to not be interpreted as nested key
                new_value = _convert_conf_item(item)
                result[new_key] = new_value
    elif isinstance(conf_item, list):
        if len(conf_item) > 0:
            result = [_convert_conf_item(item) for item in conf_item]
        else:
            result = []
    elif conf_item is True:
        return True
    elif conf_item is False:
        return False
    else:
        return conf_item

    return result


def to_dict(config: ConfigTree) -> Dict:
    """Convert HOCON input into a Dict"""
    return _convert_conf_item(config)


def to_json(config: ConfigTree) -> str:
    """Convert HOCON input into a JSON string"""
    return HOCONConverter.to_json(config)
