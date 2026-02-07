"""Developer Agent — writes code for assigned tasks.

Uses Deep Agents with filesystem + shell tools.
Scope-constrained to only work on the current task.

Architecture:
    1. Creates a Deep Agent with filesystem + shell backends
    2. Sends the task prompt with scope constraints
    3. Agent executes autonomously (reads files, writes code, runs commands)
    4. Captures output: files created, commands run, success/failure
"""

import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from eunomia_v2.models.results import DeveloperOutput
from eunomia_v2.models.task import Task, TaskType
from eunomia_v2.prompts.developer import DEVELOPER_SYSTEM_PROMPT, build_task_prompt
from eunomia_v2.utils.env import build_env_info, detect_tool_paths
from eunomia_v2.utils.git import ensure_gitignore
from eunomia_v2.utils.paths import to_posix

logger = logging.getLogger(__name__)

# Default model for the developer agent
DEFAULT_MODEL = "anthropic:claude-sonnet-4-5-20250929"


async def execute_task(
    task: Task,
    project_path: str | Path,
    model: str = DEFAULT_MODEL,
    existing_files: dict[str, str] | None = None,
) -> DeveloperOutput:
    """Execute a single task using a Deep Agent with filesystem + shell tools.

    The agent operates in the project directory and has full access to:
    - Filesystem: read, write, edit, search files
    - Shell: run terminal commands (npm install, pip install, etc.)

    Args:
        task: The task to execute.
        project_path: Absolute path to the project directory.
        model: LangChain model string (e.g. "anthropic:claude-sonnet-4-5-20250929").
        existing_files: Optional dict of filename → summary for context.

    Returns:
        DeveloperOutput with files created/modified, commands run, and status.
    """
    from deepagents import create_deep_agent
    from deepagents.backends import LocalShellBackend
    from langgraph.checkpoint.memory import MemorySaver

    project_path = Path(project_path).resolve()
    project_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Executing task [{task.id}] {task.title} in {project_path}")
    logger.info(f"Task type: {task.task_type.value}, file: {task.filename or '(none)'}")

    # Pre-task setup: ensure .gitignore exists (V1 lesson)
    if task.task_type == TaskType.INFRASTRUCTURE:
        ensure_gitignore(project_path)

    # Pre-detect tool paths (LocalShellBackend has minimal PATH)
    tools = detect_tool_paths(project_path)
    env_info = build_env_info(tools, to_posix(project_path))

    # Build project context from existing files
    project_context = ""
    if existing_files:
        context_parts = ["These files already exist in the project:"]
        for fname, summary in existing_files.items():
            context_parts.append(f"- `{to_posix(fname)}`: {summary}")
        project_context = "\n".join(context_parts)

    # Build the task-specific prompt with explicit project path
    task_prompt = build_task_prompt(
        task_title=task.title,
        task_description=task.description,
        task_type=task.task_type.value,
        filename=to_posix(task.filename) if task.filename else "",
        acceptance_criteria=task.acceptance_criteria,
        project_context=project_context,
        project_path=to_posix(project_path),
        env_info=env_info,
    )

    # Create backend — LocalShellBackend provides BOTH file I/O and shell execution
    # (read, write, edit, execute, glob, grep, ls) all sandboxed to cwd.
    # Note: CompositeBackend(FilesystemBackend + LocalShellBackend) breaks root_dir
    # sandboxing. LocalShellBackend alone has all capabilities we need.
    backend = LocalShellBackend(
        root_dir=str(project_path), virtual_mode=True, inherit_env=True,
    )

    # Build subagents for delegation (M10)
    from eunomia_v2.agents.subagents import (
        make_multi_file_coder_subagent,
        make_researcher_subagent,
    )
    subagents = [make_multi_file_coder_subagent(), make_researcher_subagent()]

    # Create the Deep Agent with subagent delegation
    agent = create_deep_agent(
        model=model,
        backend=backend,
        system_prompt=DEVELOPER_SYSTEM_PROMPT,
        name=f"developer-task-{task.id}",
        checkpointer=MemorySaver(),
        subagents=subagents,
        debug=False,
    )

    logger.info("Deep Agent created, starting execution...")

    # Run the agent
    config = {"configurable": {"thread_id": f"dev-task-{task.id}"}}
    input_msg = {"messages": [HumanMessage(content=task_prompt)]}

    # Collect all agent messages for output parsing
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
        logger.error(f"Agent execution failed: {e}")
        return DeveloperOutput(
            success=False,
            error=str(e),
            output="\n".join(all_messages),
        )

    # Parse agent output to determine what happened
    output = _parse_agent_output(all_messages, tool_calls, project_path)

    logger.info(
        f"Task [{task.id}] complete: success={output.success}, "
        f"files_created={len(output.files_created)}, "
        f"commands_run={len(output.commands_run)}"
    )

    return output


def _parse_agent_output(
    messages: list[str],
    tool_calls: list[dict[str, Any]],
    project_path: Path,
) -> DeveloperOutput:
    """Parse agent messages and tool calls into a structured DeveloperOutput.

    Scans tool calls for filesystem writes and shell commands to build
    the lists of files_created, files_modified, and commands_run.
    Also does post-hoc filesystem scanning for files the agent may have
    created via shell commands (echo >>, etc.).

    Args:
        messages: All text messages from the agent.
        tool_calls: All tool calls made by the agent.
        project_path: The project root directory.

    Returns:
        Populated DeveloperOutput.
    """
    files_created: list[str] = []
    files_modified: list[str] = []
    commands_run: list[str] = []

    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("args", {})

        # Track file operations — Deep Agents tool names
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

    # Scan messages for "Updated file ..." patterns from Deep Agents SDK
    combined_text = "\n".join(messages)
    for match in re.finditer(r"Updated file\s+([^\s]+)", combined_text):
        path = match.group(1)
        posix_path = to_posix(path)
        if posix_path not in files_created:
            files_created.append(posix_path)

    # Post-hoc filesystem scan: find actual files in project_path
    # (agent may have created files via shell echo/redirect commands)
    actual_files = _scan_project_files(project_path)

    # Determine success from the agent's final message
    # Look at the LAST model message — if it says "completed" or "summary", it succeeded
    last_model_msg = ""
    for msg in reversed(messages):
        if any(kw in msg.lower() for kw in ["summary", "completed", "successfully", "all tasks"]):
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
    """Scan the project directory for non-gitignore files.

    Returns paths relative to project_path in posix format.
    Used as a fallback when tool call tracking misses file creation
    (e.g. agent used shell echo/redirect commands).
    """
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv"}
    files: list[str] = []

    if not project_path.exists():
        return files

    for item in project_path.rglob("*"):
        if item.is_file():
            # Skip files in ignored directories
            parts = item.relative_to(project_path).parts
            if any(part in skip_dirs for part in parts):
                continue
            # Skip .gitignore itself (pre-created by us)
            if item.name == ".gitignore":
                continue
            files.append(to_posix(item.relative_to(project_path)))

    return files
