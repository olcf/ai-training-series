"""Small logging helpers for tutorial-friendly stage output."""

from functools import wraps


VERBOSE = False


def set_verbose(enabled: bool) -> None:
    global VERBOSE
    VERBOSE = enabled


def log_function(function_name: str, summary: str) -> None:
    if VERBOSE:
        print(f"[verbose] {function_name}: {summary}")


def verbose_step(summary: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log_function(func.__name__, summary)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def log_stage(step: str, message: str) -> None:
    print(f"[{step}] {message}")


def log_substep(step: str, message: str) -> None:
    print(f"[{step}] -> {message}")
