"""Cross-platform path utilities — lessons from V1."""

import platform
from pathlib import Path

# Windows reserved device names that break git and filesystem operations
WINDOWS_RESERVED_NAMES = frozenset({
    "con", "prn", "aux", "nul",
    "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9",
    "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
})


def to_posix(path: str | Path) -> str:
    """Convert a path to forward slashes.

    V1 lesson: Windows backslashes break Jest/Vitest regex-based path matching.
    Path.as_posix() is a no-op on Linux/macOS.
    """
    return Path(path).as_posix()


def is_reserved_filename(name: str) -> bool:
    """Check if a filename is a Windows reserved device name.

    V1 lesson: Claude CLI sometimes creates a file named 'nul' on Windows,
    which breaks git add -A because 'nul' is a reserved device name.
    """
    stem = Path(name).stem.lower()
    return stem in WINDOWS_RESERVED_NAMES


def is_windows() -> bool:
    """Check if running on Windows."""
    return platform.system() == "Windows"
