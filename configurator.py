import argparse
import inspect
from dataclasses import dataclass, fields
import tokenize
import io

# The previous version of this file was a "Poor Man's Configurator" and was described
# as "probably a terrible idea". This is also probably a terrible idea, but it
# brings some argparse sanity, while preserving the requirements outlined in the
# previous version:
# - avoiding configuration complexity
# - not having to prepend config

def arguments(cls):
    parser = argparse.ArgumentParser( formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = inspect.getsource(cls)
    source_bytes = source.encode("utf-8")
    source_file = io.BytesIO(source_bytes)
    tokens = tokenize.tokenize(source_file.readline)
    tokens = {t.string:t.line for t in tokens}
    for field in fields(cls):
        comment = None
        try:
            comment = tokens[field.name].split('#')[1].strip()
        except:
            pass
        parser.add_argument(f"--{field.name}", type=field.type, default=field.default, help=comment)
    args = parser.parse_args()
    cls.__init__ = lambda self : self.__dict__.update(args.__dict__)
    return cls
