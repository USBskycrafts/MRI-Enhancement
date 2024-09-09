import time
import functools


def splitter(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        print(f"--------{f.__name__}--------")
        f(*args, **kwargs)
        print("--------end---------\n\n")
    return wrapper


def timer(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        f(*args, **kwargs)
        print(f"{time.time() - start} sec")
    return wrapper
