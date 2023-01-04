import json

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False

def guess_config(d):
    f = dict(filter(lambda e: not str(e[0]).startswith("__") and is_jsonable(e[1]), d.items()))
    return f 