import contextlib
import dataclasses
import time
from typing import Generator

import termcolor


@contextlib.contextmanager
def stopwatch(label: str = "unlabeled block") -> Generator[None, None, None]:
    start_time = time.time()
    print(f"\n========")
    print(f"Running ({label})")
    yield
    print(f"{termcolor.colored(str(time.time() - start_time), attrs=['bold'])} seconds")
    print(f"========")


def immutable_dataclass(cls):
    """Decorator for defining immutable dataclasses."""

    # Hash based on object ID, rather than contents
    cls.__hash__ = object.__hash__

    return dataclasses.dataclass(cls, frozen=True)
