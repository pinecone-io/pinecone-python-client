try:
    # Use the notebook-friendly auto selection if tqdm is installed.
    from tqdm.auto import tqdm
except ImportError:
    # Fallback: define a dummy tqdm that supports the same interface.
    class tqdm:
        def __init__(self, total=None, desc=None, disable=False, *args, **kwargs):
            self.total = total
            self.desc = desc
            self.disable = disable
            self.current = 0  # to keep track of progress

        def update(self, n=1):
            # Simply increment the internal counter.
            self.current += n

        def __enter__(self):
            # When entering the context, just return self.
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            # Nothing special to do when exiting the context.
            pass
