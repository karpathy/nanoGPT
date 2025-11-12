try:
    from contextlib import nullcontext
except ImportError:
    class nullcontext:
        def __enter__(self):
            return None
        def __exit__(self, *args):
            pass
