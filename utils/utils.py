import os
import time
import functools

from datetime import datetime
from functools import partial, update_wrapper

from .logging import log_to_file


def str2bool(s):
    if s.lower() in ("1", "true", "yes"):
        return True
    elif s.lower() in ("0", "false", "no"):
        return False
    else:
        raise ValueError("Invalid boolean value: " + s)


def str2boolorstrfun(allowed_strings):
    if not isinstance(allowed_strings, (tuple, list)):
        allowed_strings = [allowed_strings]

    def str2boolorstr(s):
        if allowed_strings is not None and s in allowed_strings:
            return s

        return str2bool(s)
    return str2boolorstr



def partial_wrap(func, *args, **kwargs):
    p = partial(func, *args, **kwargs)
    p = update_wrapper(p, func)

    return p


def time_fun(fun):
    @functools.wraps(fun)
    def time_wrapper(*args, **kwargs):
        t_start = time.process_time()
        t_real_start = time.time()

        ret_val = fun(*args, **kwargs)

        t_end = time.process_time()
        t_real_end = time.time()

        print("Took {} seconds CPU time ({} seconds realtime)".format(t_end - t_start, t_real_end - t_real_start))

        return ret_val

    return time_wrapper


class Timer:
    def __init__(self, verbose=True):
        self.t_start = None
        self.t_real_start = None
        self.t_end = None
        self.t_real_end = None
        self.t_diff = None
        self.t_real_diff = None

        self.t_last = None
        self.t_real_last = None

        self.verbose = verbose

    def __enter__(self):
        self.start()

        return self

    def start(self):
        self.t_start = time.process_time()
        self.t_real_start = time.time()

        self.t_last = self.t_start
        self.t_real_last = self.t_real_start

    def measure(self, log_text=None):
        tp = time.process_time()
        tr = time.time()

        diff_p = tp - self.t_last
        diff_r = tr - self.t_real_last

        if log_text is None:
            log_text = ""
        else:
            log_text = " " + log_text

        print(f"Timing{log_text}: {diff_p} process, {diff_r} real")

        self.t_last = tp
        self.t_real_last = tr

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t_end = time.process_time()
        self.t_real_end = time.time()

        self.t_diff = self.t_end - self.t_start
        self.t_real_diff = self.t_real_end - self.t_real_start

        if self.verbose:
            print("Took {} seconds CPU time ({} seconds realtime)".format(self.t_diff, self.t_real_diff))


class LineUpdater:
    def __init__(self):
        self.first_run = True
        self.last_text = ""

    def print(self, text):
        if not self.first_run:
            print("\b" * len(self.last_text), end="")

        print(text, end="\r")

        self.last_text = text

    def __enter__(self):
        self.first_run = True
        self.last_text = ""

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print()