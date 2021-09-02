from typing import Callable
import time
import os

import tqdm.notebook
if tqdm.notebook.IProgress is None:
    # Not notebook compatible
    from tqdm import tqdm
else:
    from tqdm.auto import tqdm


class ProgressBar:
    DISABLE_PROGRESS_BAR = bool(os.getenv("PINECONE_DISABLE_PROGRESS_BAR"))

    def __init__(
        self,
        total: int,
        get_current_fn: Callable = None,
        get_remaining_fn: Callable = None,
        interval: float = 2,
        timeout: int = 3600,
        disable: bool = False,
        **kwargs
    ):
        """Progess Bar

        :param total: the total number of iterations
        :type total: int
        :param get_current_fn: get the current progress
        :type get_current_fn: Callable
        :param get_remaining_fn: get the remaining number of iterations
        :type get_remaining_fn: Callable
        :param interval: wait time between progress updates
        :type interval: float
        :param timeout: how long do we wait. Set to ``0`` to never time out.
        :type timeout: int
        :param disable: whether to disable the progress bar. Defaults to ``False``
        :type disable: bool
        :param kwargs: key word arguments tqdm
        :type kwargs: dict
        """
        self.total = total
        self.get_current_fn = get_current_fn
        self.get_remaining_fn = get_remaining_fn
        self.interval = interval
        self.timeout = timeout
        self.disable = disable or self.DISABLE_PROGRESS_BAR
        self.kwargs = kwargs
        self.progress = 0

    def update_progress(self):
        progress = None
        if self.get_current_fn:
            progress = self.get_current_fn()
        elif self.get_remaining_fn:
            progress = self.total - self.get_remaining_fn()
        else:
            raise RuntimeError("Unable to get current progress.")
        # NOTE: round float
        self.progress = round(progress, 4)

    def watch(self):
        tick = time.perf_counter()
        with tqdm(total=self.total, initial=self.progress, disable=self.disable, **self.kwargs) as pbar:
            while self.progress < self.total:
                time.sleep(self.interval)
                tock = time.perf_counter()
                if 0 < self.timeout < tock - tick:
                    raise RuntimeError("Time out when waiting for updates.")
                # Update progress only if it has changed.
                prev_progress = self.progress
                self.update_progress()
                pbar.update(round(self.progress - prev_progress, 4))

    @classmethod
    def iter(cls, iterable=None, **kwargs):
        """Returns vanilla tqdm."""
        disable = kwargs.pop("disable", cls.DISABLE_PROGRESS_BAR)
        return tqdm(iterable, disable=disable, **kwargs)
