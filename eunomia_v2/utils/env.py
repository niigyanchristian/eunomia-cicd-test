"""Environment detection for Deep Agent shell backends.

LocalShellBackend has a minimal PATH that lacks python, pip, git, node, etc.
This module pre-detects absolute paths to key executables so agents can use
them directly without needing PATH resolution.
"""

import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def detect_tool_paths(project_path: Path | str = ".") -> dict[str, str]:
    """Detect absolute paths to key development tools.

    Searches for executables in this order:
    1. Project .venv (Scripts/python.exe on Windows, bin/python on Unix)
    2. Parent directory .venv (common for monorepos)
    3. Current Python interpreter (sys.executable)
    4. System PATH (shutil.which)

    Args:
        project_path: Project directory to check for local venvs.

    Returns:
        Dict mapping tool names to their absolute paths.
        Keys: python, pip, git, node, npm. Values are empty string if not found.
    """
    project_path = Path(project_path).resolve()
    tools: dict[str, str] = {
        "python": "",
        "pip": "",
        "git": "",
        "node": "",
        "npm": "",
    }

    # --- Python ---
    # Try 1: .venv in project or parent directories
    for search_dir in [project_path, project_path.parent, project_path.parent.parent]:
        venv_python = search_dir / ".venv" / "Scripts" / "python.exe"
        if not venv_python.exists():
            venv_python = search_dir / ".venv" / "bin" / "python"
        if venv_python.exists():
            tools["python"] = str(venv_python)
            tools["pip"] = f'"{venv_python}" -m pip'
            logger.info("Found venv python: %s", venv_python)
            break

    # Try 2: Current Python interpreter
    if not tools["python"]:
        current = Path(sys.executable)
        if current.exists():
            tools["python"] = str(current)
            tools["pip"] = f'"{current}" -m pip'
            logger.info("Using current python: %s", current)

    # Try 3: System PATH
    if not tools["python"]:
        for name in ("python", "python3", "py"):
            found = shutil.which(name)
            if found:
                tools["python"] = found
                tools["pip"] = f'"{found}" -m pip'
                logger.info("Found system python: %s", found)
                break

    # --- Git ---
    git_path = shutil.which("git")
    if git_path:
        tools["git"] = git_path
    else:
        # Common Windows install locations
        for candidate in [
            Path("C:/Program Files/Git/cmd/git.exe"),
            Path("C:/Program Files (x86)/Git/cmd/git.exe"),
            Path("C:/Program Files/Git/bin/git.exe"),
        ]:
            if candidate.exists():
                tools["git"] = str(candidate)
                break

    # --- Node / npm ---
    tools["node"] = shutil.which("node") or ""
    tools["npm"] = shutil.which("npm") or ""

    return tools


def build_env_info(tools: dict[str, str], project_path: Path | str = "") -> str:
    """Build an environment info block for agent prompts.

    Generates a markdown-formatted section that tells the agent exactly
    which executables to use, avoiding PATH discovery failures.

    Args:
        tools: Dict from detect_tool_paths().
        project_path: Project path for display.

    Returns:
        Formatted environment info string for prompt injection.
    """
    lines = ["## Pre-Detected Environment"]

    if project_path:
        lines.append(f"**Working Directory (absolute)**: `{project_path}`")

    lines.append("")

    # --- PATH RULES ---
    lines.append("### PATH RULES (CRITICAL — read carefully)")
    lines.append("")
    lines.append("Your tools have **two different path systems**:")
    lines.append("")
    lines.append("**1. File tools** (`write_file`, `read_file`, `edit_file`, `list_files`, "
                 "`search_files`):")
    lines.append("   - Use **virtual paths** starting with `/` relative to the project root")
    lines.append("   - Example: `/.gitignore`, `/app/main.py`, `/tests/test_main.py`")
    lines.append("   - **NEVER** use Windows absolute paths like `C:/Users/...` with file tools")
    lines.append("   - **NEVER** use bare filenames like `main.py` — always start with `/`")
    lines.append("")
    lines.append("**2. Shell tool** (`execute` / `run_command`):")
    lines.append("   - The shell's working directory is already the project root")
    lines.append("   - Use **relative paths** for project files: `app/main.py`")
    lines.append("   - Use **full absolute paths** for executables (see below)")
    lines.append("")

    # --- TOOL PATHS ---
    lines.append("### Pre-Detected Tool Paths")
    lines.append("")
    lines.append("Use these exact paths in **shell commands** instead of bare command names:")
    lines.append("")

    if tools.get("python"):
        lines.append(f'- **Python**: `"{tools["python"]}"`')
        lines.append(f'- **pip**: `{tools["pip"]}`')
        lines.append(f'- **pytest**: `"{tools["python"]}" -m pytest`')
    else:
        lines.append("- **Python**: NOT FOUND (use shell `echo` for file writes if needed)")

    if tools.get("git"):
        lines.append(f'- **git**: `"{tools["git"]}"`')
    else:
        lines.append("- **git**: NOT FOUND (skip git commands)")

    if tools.get("node"):
        lines.append(f'- **node**: `"{tools["node"]}"`')
    if tools.get("npm"):
        lines.append(f'- **npm**: `"{tools["npm"]}"`')

    lines.append("")
    lines.append("**CRITICAL**: Do NOT use bare `python`, `pip`, `git`, `node`, or `npm` "
                 "in shell commands. They will fail because PATH is not configured. "
                 "Always use the full quoted paths shown above.")

    return "\n".join(lines)
