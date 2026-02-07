"""QA Agent — runs tests and validates implementations.

Uses Deep Agents with filesystem + shell tools.
Detects test framework automatically, runs tests, and parses results
into a structured QAResult.

Architecture:
    1. Creates a Deep Agent with LocalShellBackend (read-only + shell)
    2. Sends QA prompt with test file path
    3. Agent detects framework, runs tests, reports results
    4. Parses output for pass/fail counts and returns QAResult
"""

import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from eunomia_v2.models.results import QAResult
from eunomia_v2.models.task import Task
from eunomia_v2.prompts.qa import QA_SYSTEM_PROMPT, build_qa_prompt
from eunomia_v2.utils.env import build_env_info, detect_tool_paths
from eunomia_v2.utils.paths import to_posix

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "anthropic:claude-sonnet-4-5-20250929"


async def run_tests(
    task: Task,
    project_path: str | Path,
    model: str = DEFAULT_MODEL,
    test_command: str = "",
) -> QAResult:
    """Run tests for a QA validation task using a Deep Agent.

    The agent operates in the project directory, detects the test framework,
    runs the appropriate test command, and reports results.

    Args:
        task: The QA validation task (must reference a test file).
        project_path: Absolute path to the project directory.
        model: LangChain model string.
        test_command: Pre-built test command. If empty, auto-detects based on
                      project structure. Use this when you know the exact
                      python/node executable path.

    Returns:
        QAResult with pass/fail status, test counts, and error details.
    """
    from deepagents import create_deep_agent
    from deepagents.backends import LocalShellBackend
    from langgraph.checkpoint.memory import MemorySaver

    project_path = Path(project_path).resolve()

    logger.info(f"Running QA for task [{task.id}] {task.title} in {project_path}")
    logger.info(f"Test file: {task.filename or '(auto-detect)'}")

    # Auto-detect test command if not provided
    if not test_command:
        test_command = _detect_test_command(project_path, task.filename)
        if test_command:
            logger.info(f"Auto-detected test command: {test_command}")

    # Pre-detect tool paths for env_info fallback
    tools = detect_tool_paths(project_path)
    env_info = build_env_info(tools, to_posix(project_path))

    # Build the QA prompt
    qa_prompt = build_qa_prompt(
        task_title=task.title,
        task_description=task.description,
        test_file=to_posix(task.filename) if task.filename else "",
        project_path=to_posix(project_path),
        acceptance_criteria=task.acceptance_criteria,
        test_command=test_command,
        env_info=env_info,
    )

    # Create backend — LocalShellBackend for file reading + test execution
    backend = LocalShellBackend(
        root_dir=str(project_path), virtual_mode=True, inherit_env=True,
    )

    # Build subagents for delegation (M10)
    from eunomia_v2.agents.subagents import (
        make_test_writer_subagent,
        make_test_debugger_subagent,
    )
    subagents = [make_test_writer_subagent(), make_test_debugger_subagent()]

    # Create the Deep Agent with subagent delegation
    agent = create_deep_agent(
        model=model,
        backend=backend,
        system_prompt=QA_SYSTEM_PROMPT,
        name=f"qa-task-{task.id}",
        checkpointer=MemorySaver(),
        subagents=subagents,
        debug=False,
    )

    logger.info("QA Deep Agent created, starting test execution...")

    # Run the agent
    config = {"configurable": {"thread_id": f"qa-task-{task.id}"}}
    input_msg = {"messages": [HumanMessage(content=qa_prompt)]}

    all_messages: list[str] = []

    try:
        async for event in agent.astream(input_msg, config=config, stream_mode="updates"):
            for node, update in event.items():
                if update is None:
                    continue
                if not isinstance(update, dict):
                    continue
                if "messages" in update:
                    msgs = update["messages"]
                    if hasattr(msgs, "value"):
                        msgs = msgs.value
                    if not isinstance(msgs, list):
                        msgs = [msgs]
                    for msg in msgs:
                        content = getattr(msg, "content", "")
                        if content and isinstance(content, str):
                            all_messages.append(content)
                            logger.info(f"[{node}] {content[:200]}")

    except Exception as e:
        logger.error(f"QA agent execution failed: {e}")
        return QAResult(
            passed=False,
            output="\n".join(all_messages),
            errors=[str(e)],
        )

    # Parse the test results from agent output
    combined_output = "\n".join(all_messages)
    result = parse_test_output(combined_output)

    logger.info(
        f"QA [{task.id}] complete: passed={result.passed}, "
        f"total={result.total_tests}, failed={result.tests_failed}, "
        f"framework={result.framework_detected}"
    )

    return result


def parse_test_output(output: str) -> QAResult:
    """Parse test execution output into a structured QAResult.

    Supports multiple test framework output formats:
    - pytest: "=== 3 passed, 1 failed in 0.5s ==="
    - vitest: "Tests  3 passed | 1 failed (4)"
    - jest: "Tests:  2 passed, 1 failed, 3 total"
    - Generic: looks for passed/failed/total patterns

    Also parses our structured "TEST RESULTS:" format from the agent.

    Args:
        output: Raw combined output from the QA agent.

    Returns:
        Populated QAResult.
    """
    passed_count = 0
    failed_count = 0
    skipped_count = 0
    total_count = 0
    framework = ""
    errors: list[str] = []

    # Try structured format first (our agent's output format)
    structured = _parse_structured_results(output)
    if structured:
        return QAResult(
            passed=structured["failed"] == 0 and structured["total"] > 0,
            total_tests=structured["total"],
            tests_passed=structured["passed"],
            tests_failed=structured["failed"],
            tests_skipped=structured["skipped"],
            framework_detected=structured["framework"],
            output=output[-2000:] if len(output) > 2000 else output,
            errors=errors,
        )

    # Try pytest format: "=== 3 passed, 1 failed in 0.5s ==="
    pytest_match = re.search(
        r"={2,}\s+(.*?)\s+in\s+[\d.]+s\s*={2,}", output
    )
    if pytest_match:
        framework = "pytest"
        summary = pytest_match.group(1)
        p = re.search(r"(\d+)\s+passed", summary)
        f = re.search(r"(\d+)\s+failed", summary)
        s = re.search(r"(\d+)\s+skipped", summary)
        if p:
            passed_count = int(p.group(1))
        if f:
            failed_count = int(f.group(1))
        if s:
            skipped_count = int(s.group(1))
        total_count = passed_count + failed_count + skipped_count

    # Try vitest format: "Tests  3 passed | 1 failed (4)"
    if not framework:
        vitest_match = re.search(
            r"Tests\s+(?:(\d+)\s+passed)?(?:\s*\|\s*)?(?:(\d+)\s+failed)?(?:\s*\|\s*)?(?:(\d+)\s+skipped)?\s*\((\d+)\)",
            output,
        )
        if vitest_match:
            framework = "vitest"
            passed_count = int(vitest_match.group(1) or 0)
            failed_count = int(vitest_match.group(2) or 0)
            skipped_count = int(vitest_match.group(3) or 0)
            total_count = int(vitest_match.group(4))

    # Try jest format: "Tests:  2 passed, 1 failed, 3 total"
    if not framework:
        jest_match = re.search(r"Tests:\s+(.*total)", output)
        if jest_match:
            framework = "jest"
            line = jest_match.group(1)
            p = re.search(r"(\d+)\s+passed", line)
            f = re.search(r"(\d+)\s+failed", line)
            t = re.search(r"(\d+)\s+total", line)
            if p:
                passed_count = int(p.group(1))
            if f:
                failed_count = int(f.group(1))
            if t:
                total_count = int(t.group(1))
            skipped_count = total_count - passed_count - failed_count

    # Generic fallback: look for any passed/failed patterns
    if not framework:
        framework = "unknown"
        # Take the last match for each (most recent output)
        for p_match in re.finditer(r"(\d+)\s+(?:tests?\s+)?passed", output, re.IGNORECASE):
            passed_count = int(p_match.group(1))
        for f_match in re.finditer(r"(\d+)\s+(?:tests?\s+)?failed", output, re.IGNORECASE):
            failed_count = int(f_match.group(1))
        for s_match in re.finditer(r"(\d+)\s+(?:tests?\s+)?skipped", output, re.IGNORECASE):
            skipped_count = int(s_match.group(1))
        total_count = passed_count + failed_count + skipped_count

        # Final fallback: check for success signals
        if total_count == 0:
            lower = output.lower()
            if "all tests passed" in lower or "test suites: 1 passed" in lower:
                passed_count = 1
                total_count = 1
                framework = "generic"

    # Extract error messages for failed tests
    if failed_count > 0:
        # Look for common error patterns
        for err_match in re.finditer(
            r"(?:FAILED|FAIL|ERROR|AssertionError|TypeError|ValueError).*",
            output,
        ):
            errors.append(err_match.group(0)[:500])
        # Limit to 10 errors
        errors = errors[:10]

    success = failed_count == 0 and total_count > 0

    return QAResult(
        passed=success,
        total_tests=total_count,
        tests_passed=passed_count,
        tests_failed=failed_count,
        tests_skipped=skipped_count,
        framework_detected=framework,
        output=output[-2000:] if len(output) > 2000 else output,
        errors=errors,
    )


def _parse_structured_results(output: str) -> dict[str, Any] | None:
    """Parse our structured TEST RESULTS format from the agent output.

    Looks for:
        TEST RESULTS:
        - Framework: pytest
        - Total: 5
        - Passed: 4
        - Failed: 1
        - Skipped: 0
        - Status: FAILED

    Returns dict with keys: framework, total, passed, failed, skipped.
    Returns None if structured format not found.
    """
    # Look for the TEST RESULTS block
    match = re.search(r"TEST RESULTS:", output)
    if not match:
        return None

    block = output[match.start():]

    framework_match = re.search(r"Framework:\s*(\S+)", block)
    total_match = re.search(r"Total:\s*(\d+)", block)
    passed_match = re.search(r"Passed:\s*(\d+)", block)
    failed_match = re.search(r"Failed:\s*(\d+)", block)
    skipped_match = re.search(r"Skipped:\s*(\d+)", block)

    if not total_match:
        return None

    return {
        "framework": framework_match.group(1) if framework_match else "unknown",
        "total": int(total_match.group(1)),
        "passed": int(passed_match.group(1)) if passed_match else 0,
        "failed": int(failed_match.group(1)) if failed_match else 0,
        "skipped": int(skipped_match.group(1)) if skipped_match else 0,
    }


def _detect_test_command(project_path: Path, test_file: str = "") -> str:
    """Auto-detect the test command based on project structure.

    Checks for:
    1. Python venv with pytest in the project or parent directories
    2. System python with pytest
    3. Node.js projects with package.json test scripts
    4. Common executable locations on Windows

    Args:
        project_path: The project root directory.
        test_file: Optional specific test file to run.

    Returns:
        Pre-built test command string, or empty string if not detected.
    """
    import shutil
    import sys

    test_file_arg = f" {test_file}" if test_file else ""

    # Check for Python test files
    has_python_tests = any(
        project_path.glob("test_*.py")
    ) or any(
        project_path.glob("*_test.py")
    ) or (test_file and test_file.endswith(".py"))

    if has_python_tests:
        # Try 1: Look for .venv in project or parent directories
        for search_dir in [project_path, project_path.parent, project_path.parent.parent]:
            venv_python = search_dir / ".venv" / "Scripts" / "python.exe"
            if not venv_python.exists():
                venv_python = search_dir / ".venv" / "bin" / "python"
            if venv_python.exists():
                python_path = to_posix(venv_python)
                logger.info(f"Found venv python: {python_path}")
                return f'"{venv_python}" -m pytest{test_file_arg} -v'

        # Try 2: Use the current Python interpreter
        current_python = Path(sys.executable)
        if current_python.exists():
            return f'"{current_python}" -m pytest{test_file_arg} -v'

        # Try 3: System python
        python_path = shutil.which("python") or shutil.which("python3")
        if python_path:
            return f'"{python_path}" -m pytest{test_file_arg} -v'

    # Check for Node.js test files
    has_js_tests = any(
        project_path.glob("*.test.ts")
    ) or any(
        project_path.glob("*.spec.ts")
    ) or any(
        project_path.glob("*.test.js")
    )

    if has_js_tests:
        pkg_json = project_path / "package.json"
        if pkg_json.exists():
            import json
            try:
                pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
                scripts = pkg.get("scripts", {})
                if "test" in scripts:
                    return f"npx {scripts['test']}"
            except (json.JSONDecodeError, OSError):
                pass

        # Default to vitest or jest
        if (project_path / "vitest.config.ts").exists():
            return f"npx vitest run{test_file_arg}"
        if (project_path / "jest.config.js").exists():
            return f"npx jest{test_file_arg}"

    return ""
