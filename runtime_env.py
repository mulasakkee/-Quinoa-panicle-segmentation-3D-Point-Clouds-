"""Prepare Windows DLL lookup paths before importing NumPy/SciPy/Open3D.

Usage must precede imports of binary packages::

    from runtime_env import configure_runtime_environment
    configure_runtime_environment()

    import numpy as np

The handles returned by :func:`os.add_dll_directory` are intentionally kept at
module scope.  Closing or garbage-collecting them would remove the directories
from the process DLL search path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


_DLL_DIRECTORY_HANDLES: dict[str, object] = {}


def _path_key(path: str | os.PathLike[str]) -> str:
    """Return a case-insensitive, normalized key suitable for PATH de-duping."""

    return os.path.normcase(os.path.normpath(os.fspath(path)))


def configure_runtime_environment() -> tuple[Path, ...]:
    """Prepend the current Python environment's Windows runtime directories.

    Returns the ordered directories placed at the front of ``PATH``.  On
    non-Windows systems no change is made and an empty tuple is returned.  The
    function is idempotent and safe to call more than once.
    """

    if os.name != "nt":
        return ()

    environment_root = Path(sys.executable).resolve().parent
    runtime_directories = (
        environment_root,
        environment_root / "Library" / "mingw-w64" / "bin",
        environment_root / "Library" / "usr" / "bin",
        environment_root / "Library" / "bin",
        environment_root / "Scripts",
        environment_root / "bin",
    )

    current_entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    runtime_keys = {_path_key(path) for path in runtime_directories}
    remaining_entries = [entry for entry in current_entries if _path_key(entry) not in runtime_keys]
    os.environ["PATH"] = os.pathsep.join(
        [*(str(path) for path in runtime_directories), *remaining_entries]
    )

    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        for directory in runtime_directories:
            key = _path_key(directory)
            if directory.is_dir() and key not in _DLL_DIRECTORY_HANDLES:
                _DLL_DIRECTORY_HANDLES[key] = add_dll_directory(str(directory))

    return runtime_directories


__all__ = ["configure_runtime_environment"]
