"""Minimal cross-process file locks.

One mechanism, reused wherever a read/check/write cycle must be serialized
across processes: annotation document saves and catalog rebuilds. The lock
lives in a ``<name>.lock`` sibling file of the protected target. Blocking
acquisition is safe against holder crashes: the OS releases the lock when
the process dies.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

__all__ = ["file_lock"]


@contextmanager
def file_lock(target: Path) -> Iterator[None]:
    """Cross-process mutex guarding ``target`` via a sibling lock file."""
    lock_path = target.with_name(target.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        if sys.platform == "win32":
            handle.seek(0)
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if sys.platform == "win32":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
