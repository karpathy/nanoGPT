"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.

IMPLICIT CONTRACT / RISK (Hokmah architectural audit):
    This file is exec()'d inside the caller's global scope (train.py, sample.py,
    bench.py). It mutates the caller's globals() directly. Two silent failure modes:

    1. GHOST VARIABLE: a typo in a config file (e.g. "learing_rate = 1e-4" instead
       of "learning_rate = 1e-4") introduces a brand-new global that overrides nothing
       and raises no error. train.py now detects and warns about this pattern.

    2. TYPE MISMATCH: a --key=value override whose inferred type doesn't match the
       default will now raise a descriptive ValueError instead of a bare AssertionError.
"""

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        # assume it's the name of a config file
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match — raise a descriptive error instead of a bare
            # AssertionError so the caller knows exactly what went wrong
            if type(attempt) != type(globals()[key]):
                raise ValueError(
                    f"Type mismatch for config key '{key}': "
                    f"expected {type(globals()[key]).__name__} "
                    f"(current value: {globals()[key]!r}), "
                    f"got {type(attempt).__name__} (override value: {attempt!r}). "
                    f"Fix your --{key}= argument or config file."
                )
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
