"""
Middle Class Man's Configurator. Probably a bad idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals(), using argparse to parse
arguments from the command line and a config file.

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""

from ast import literal_eval
import argparse
import types

current_globals = globals().copy()

parser = argparse.ArgumentParser(formatter_class=argparse.MetavarTypeHelpFormatter)
parser.add_argument('config_file', nargs='?', type=str, default=None, help='Python config file to override defaults')


# Custom type for tuples, which aren't conventionally supported by argparse
def tuple_arg(val):
    val = val.replace("(", "").replace(")", "")
    parsed_val = val.split(",")
    return tuple([literal_eval(x.strip()) for x in parsed_val])

type_ignore_list = [types.FunctionType, types.ModuleType, type]

for glob in current_globals:
    if not glob.startswith('__') and not any(isinstance(current_globals[glob], t) for t in type_ignore_list):
        value = current_globals[glob]
        actual_type = type(current_globals[glob])
        tuple_helper_text = f",  to pass surround in quotes e.g. --{glob}='{value}'" if actual_type == tuple else ""
        parser_val_type = actual_type if actual_type != tuple else tuple_arg
        parser.add_argument(f'--{glob}', type=parser_val_type, help=f"default: {value}" + tuple_helper_text, default=None)

args = parser.parse_args()

if args.config_file is not None:
    print(f"Overriding config with {args.config_file}:")
    with open(args.config_file) as f:
        print(f.read())
    exec(open(args.config_file).read())

for k, v in vars(args).items():
    if k != 'config_file' and v is not None:
        print(f"Overriding: {k} = {v}")
        globals()[k] = v
