from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stdwarn = sys.stdwarn
        sys.stdwarn = devnull
        try:
            yield
        finally:
            sys.stdwarn = old_stdwarn