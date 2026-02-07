"""DevOps Agent — handles infrastructure setup and deployment.

Uses Deep Agents with filesystem + shell tools.
Responsible for project scaffolding, dependency installation,
git init, CI/CD config, and other infrastructure tasks.

Architecture:
    1. Creates a Deep Agent with LocalShellBackend (file I/O + shell)
    2. Sends infrastructure task prompt
    3. Agent executes autonomously (creates dirs, writes configs, runs installs)
    4. Captures output: files created, commands run, success/failure
"""

import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from eunomia_v2.models.results import DeveloperOutput
from eunomia_v2.models.task import Task
from eunomia_v2.prompts.devops import DEVOPS_SYSTEM_PROMPT, build_devops_prompt
from eunomia_v2.utils.env import build_env_info, detect_tool_paths
from eunomia_v2.utils.git import ensure_gitignore
from eunomia_v2.utils.paths import to_posix

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "anthropic:claude-sonnet-4-5-20250929"


async def setup_infrastructure(
    task: Task,
    project_path: str | Path,
    model: str = DEFAULT_MODEL,
) -> DeveloperOutput:
    """Execute an infrastructure task using a Deep Agent.

    The agent operates in the project directory with full access to:
    - Filesystem: read, write, edit, search files
    - Shell: run commands (npm install, pip install, git init, etc.)

    Args:
        task: The infrastructure task to execute.
        project_path: Absolute path to the project directory.
        model: LangChain model string.

    Returns:
        DeveloperOutput with files created, commands run, and status.
    """
    from deepagents import create_deep_agent
    from deepagents.backends import LocalShellBackend
    from langgraph.checkpoint.memory import MemorySaver

    project_path = Path(project_path).resolve()
    project_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"DevOps setup [{task.id}] {task.title} in {project_path}")

    # Pre-task: ensure .gitignore exists before any file creation
    ensure_gitignore(project_path)

    # Detect tech stack from task description for the prompt
    tech_stack = _detect_tech_stack(task.description)

    # Pre-detect tool paths (LocalShellBackend has minimal PATH)
    tools = detect_tool_paths(project_path)
    env_info = build_env_info(tools, to_posix(project_path))

    # Build the devops prompt
    devops_prompt = build_devops_prompt(
        task_title=task.title,
        task_description=task.description,
        project_path=to_posix(project_path),
        tech_stack=tech_stack,
        acceptance_criteria=task.acceptance_criteria,
        env_info=env_info,
    )

    # Create backend — LocalShellBackend for file I/O + shell execution
    backend = LocalShellBackend(
        root_dir=str(project_path), virtual_mode=True, inherit_env=True,
    )

    # Create the Deep Agent
    agent = create_deep_agent(
        model=model,
        backend=backend,
        system_prompt=DEVOPS_SYSTEM_PROMPT,
        name=f"devops-task-{task.id}",
        checkpointer=MemorySaver(),
        debug=False,
    )

    logger.info("DevOps Deep Agent created, starting infrastructure setup...")

    # Run the agent
    config = {"configurable": {"thread_id": f"devops-task-{task.id}"}}
    input_msg = {"messages": [HumanMessage(content=devops_prompt)]}

    all_messages: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    try:
        async for event in agent.astream(input_msg, config=config, stream_mode="updates"):
            for node, update in event.items():
                if update is None:
                    continue
                if not isinstance(update, dict):
                    continue
                if "messages" in update:
                    msgs = update["messages"]
                    # Handle LangGraph Overwrite wrapper
                    if hasattr(msgs, "value"):
                        msgs = msgs.value
                    if not isinstance(msgs, list):
                        msgs = [msgs]
                    for msg in msgs:
                        content = getattr(msg, "content", "")
                        if content and isinstance(content, str):
                            all_messages.append(content)
                            logger.info(f"[{node}] {content[:200]}")

                        # Capture tool calls for output tracking
                        msg_tool_calls = getattr(msg, "tool_calls", None)
                        if msg_tool_calls:
                            for tc in msg_tool_calls:
                                tool_calls.append({
                                    "name": tc.get("name", ""),
                                    "args": tc.get("args", {}),
                                })

    except Exception as e:
        logger.error(f"DevOps agent execution failed: {e}")
        return DeveloperOutput(
            success=False,
            error=str(e),
            output="\n".join(all_messages),
        )

    # Parse output using same logic as developer agent
    output = _parse_devops_output(all_messages, tool_calls, project_path)

    logger.info(
        f"DevOps [{task.id}] complete: success={output.success}, "
        f"files={len(output.files_created)}, cmds={len(output.commands_run)}"
    )

    return output


def _parse_devops_output(
    messages: list[str],
    tool_calls: list[dict[str, Any]],
    project_path: Path,
) -> DeveloperOutput:
    """Parse agent messages and tool calls into DeveloperOutput.

    Tracks file writes and shell commands from tool calls,
    plus post-hoc filesystem scanning for files created via shell.
    """
    files_created: list[str] = []
    files_modified: list[str] = []
    commands_run: list[str] = []

    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("args", {})

        if name in ("write_file", "create_file", "write"):
            path = args.get("path", args.get("file_path", ""))
            if path:
                files_created.append(to_posix(path))
        elif name in ("edit_file", "patch_file", "replace_in_file", "edit"):
            path = args.get("path", args.get("file_path", ""))
            if path:
                files_modified.append(to_posix(path))
        elif name in ("run_command", "execute", "shell", "bash"):
            cmd = args.get("command", args.get("cmd", str(args)))
            if cmd:
                commands_run.append(cmd)

    # Scan messages for file creation patterns
    combined_text = "\n".join(messages)
    for match in re.finditer(r"Updated file\s+([^\s]+)", combined_text):
        path = match.group(1)
        posix_path = to_posix(path)
        if posix_path not in files_created:
            files_created.append(posix_path)

    # Post-hoc filesystem scan
    actual_files = _scan_project_files(project_path)

    # Determine success
    last_model_msg = ""
    for msg in reversed(messages):
        if any(kw in msg.lower() for kw in [
            "setup complete", "infrastructure", "successfully",
            "initialized", "installed", "created",
        ]):
            last_model_msg = msg
            break

    success = bool(last_model_msg) or bool(actual_files) or bool(files_created)

    return DeveloperOutput(
        success=success,
        files_created=files_created if files_created else actual_files,
        files_modified=files_modified,
        commands_run=commands_run,
        output=combined_text[-2000:] if len(combined_text) > 2000 else combined_text,
    )


def _scan_project_files(project_path: Path) -> list[str]:
    """Scan the project directory for non-gitignore files."""
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv"}
    files: list[str] = []

    if not project_path.exists():
        return files

    for item in project_path.rglob("*"):
        if item.is_file():
            parts = item.relative_to(project_path).parts
            if any(part in skip_dirs for part in parts):
                continue
            if item.name == ".gitignore":
                continue
            files.append(to_posix(item.relative_to(project_path)))

    return files


def _detect_tech_stack(description: str) -> str:
    """Detect tech stack from task description for the prompt.

    Simple keyword matching — just for providing context to the agent.
    """
    desc_lower = description.lower()
    stacks: list[str] = []

    if any(kw in desc_lower for kw in ["python", "pip", "pyproject", "pytest", "flask", "fastapi", "django"]):
        stacks.append("Python")
    if any(kw in desc_lower for kw in ["node", "npm", "package.json", "typescript", "react", "next", "express"]):
        stacks.append("Node.js")
    if any(kw in desc_lower for kw in ["docker", "dockerfile", "container"]):
        stacks.append("Docker")
    if any(kw in desc_lower for kw in ["rust", "cargo"]):
        stacks.append("Rust")
    if any(kw in desc_lower for kw in ["go ", "golang", "go.mod"]):
        stacks.append("Go")
    if any(kw in desc_lower for kw in ["ci/cd", "cicd", "ci cd", "github actions", "gitlab ci", "jenkins", "pipeline", "workflow"]):
        stacks.append("CI/CD")

    return " + ".join(stacks) if stacks else ""
