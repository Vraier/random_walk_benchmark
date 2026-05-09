"""Call-wrapping memory metric via psutil RSS delta."""
import os
import psutil

_proc = psutil.Process(os.getpid())


def measure(fn, *args, **kwargs):
    """Run fn(*args, **kwargs), return (result, rss_delta_mb)."""
    before = _proc.memory_info().rss
    result = fn(*args, **kwargs)
    after = _proc.memory_info().rss
    return result, (after - before) / 1024 / 1024
