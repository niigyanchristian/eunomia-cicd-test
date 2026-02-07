"""Git utilities with Windows safety — lessons from V1.

Provides safe wrappers for git init, add, and commit operations.
All functions handle the case where git is not installed gracefully.
"""

import logging
import subprocess
from pathlib import Path

from eunomia_v2.utils.paths import is_reserved_filename

logger = logging.getLogger(__name__)

# Default .gitignore contents (V1 lesson: must exist BEFORE git add -A)
DEFAULT_GITIGNORE = """\
node_modules/
.venv/
venv/
__pycache__/
dist/
build/
.next/
.nuxt/
coverage/
*.pyc
.env
.env.local
.DS_Store
nul
"""


def ensure_gitignore(project_path: str | Path) -> bool:
    """Create .gitignore if it doesn't exist.

    V1 lesson: Without .gitignore, git add -A stages node_modules/
    (thousands of files) and hangs for minutes. Also, 'nul' file
    on Windows breaks git entirely.

    Returns:
        True if .gitignore was created, False if it already existed.
    """
    gitignore_path = Path(project_path) / ".gitignore"
    if gitignore_path.exists():
        return False

    gitignore_path.write_text(DEFAULT_GITIGNORE, encoding="utf-8")
    logger.info(f"Created .gitignore at {gitignore_path}")
    return True


def git_init(project_path: str | Path) -> bool:
    """Initialize a git repository if not already initialized.

    Returns:
        True if repo was initialized (or already existed), False on error.
    """
    project_path = Path(project_path)
    git_dir = project_path / ".git"

    if git_dir.exists():
        logger.debug(f"Git repo already exists at {project_path}")
        return True

    try:
        result = subprocess.run(
            ["git", "init"],
            cwd=str(project_path),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if result.returncode == 0:
            logger.info(f"Initialized git repo at {project_path}")
            return True
        else:
            logger.warning(f"git init failed: {result.stderr}")
            return False
    except FileNotFoundError:
        logger.warning("git not found on PATH — skipping git init")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("git init timed out")
        return False


def git_add_and_commit(
    project_path: str | Path,
    message: str,
    add_all: bool = True,
) -> bool:
    """Stage files and create a git commit.

    Handles Windows-specific issues:
    - Removes reserved filenames (nul, con, etc.) before staging
    - Uses utf-8 encoding for subprocess output
    - Gracefully handles missing git

    Args:
        project_path: Absolute path to the project directory.
        message: Commit message.
        add_all: If True, stages all untracked/modified files.

    Returns:
        True if commit succeeded, False otherwise.
    """
    project_path = Path(project_path)

    if not (project_path / ".git").exists():
        logger.warning("No .git directory — skipping commit")
        return False

    # V1 lesson: clean up Windows reserved filenames before git add
    _cleanup_reserved_files(project_path)

    try:
        # Stage files
        if add_all:
            add_result = subprocess.run(
                ["git", "add", "-A"],
                cwd=str(project_path),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
            if add_result.returncode != 0:
                logger.warning(f"git add failed: {add_result.stderr}")
                return False

        # Check if there's anything to commit
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(project_path),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if not status_result.stdout.strip():
            logger.info("Nothing to commit — working tree clean")
            return True  # Not an error, just nothing to do

        # Commit
        commit_result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=str(project_path),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        if commit_result.returncode == 0:
            # Extract short hash from output
            short_info = commit_result.stdout.split("\n")[0] if commit_result.stdout else ""
            logger.info(f"Git commit: {short_info}")
            return True
        else:
            logger.warning(f"git commit failed: {commit_result.stderr}")
            return False

    except FileNotFoundError:
        logger.warning("git not found on PATH — skipping commit")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("git operation timed out")
        return False


def _cleanup_reserved_files(project_path: Path) -> None:
    """Remove Windows reserved device name files that break git.

    V1 lesson: Claude CLI sometimes creates a file named 'nul' on Windows.
    'nul' is a reserved device name and causes git add -A to fail silently
    or hang. We delete these files before staging.
    """
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv"}

    for item in project_path.rglob("*"):
        if item.is_file():
            parts = item.relative_to(project_path).parts
            if any(part in skip_dirs for part in parts):
                continue
            if is_reserved_filename(item.name):
                try:
                    item.unlink()
                    logger.info(f"Removed reserved filename: {item.relative_to(project_path)}")
                except OSError as e:
                    logger.warning(f"Could not remove {item.name}: {e}")
