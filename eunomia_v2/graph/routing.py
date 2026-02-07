"""Conditional routing functions for the LangGraph orchestrator."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def route_task_to_agent(state: dict[str, Any]) -> str:
    """Route the current task to the appropriate agent based on task_type.

    Returns node name: developer, qa, devops, architect, commit, or done.
    """
    tasks = state.get("tasks", [])
    current_index = state.get("current_task_index", 0)

    if current_index >= len(tasks):
        return "done"

    task = tasks[current_index]
    task_type = task.task_type if hasattr(task, "task_type") else task.get("task_type", "")
    # TaskType(str, Enum).value gives "infrastructure", str() gives "TaskType.INFRASTRUCTURE"
    task_type_str = task_type.value if hasattr(task_type, "value") else str(task_type)

    routing_map = {
        "infrastructure": "devops",
        "feature": "developer",
        "test_writing": "developer",
        "qa_validation": "qa",
        "commit": "commit",
        "architecture": "architect",
        "cicd_setup": "devops",
    }

    target = routing_map.get(task_type_str, "developer")
    logger.info(f"Routing task [{task.id}] type={task_type} → {target}")
    return target


def route_after_developer(state: dict[str, Any]) -> str:
    """Route after developer agent completes.

    - If in a retry loop (QA previously failed) → go to QA for re-testing
    - Otherwise (normal flow) → commit the work
    """
    qa_result = state.get("qa_result")
    if qa_result and not qa_result.passed:
        logger.info("Developer fix complete → re-running QA")
        return "qa"
    logger.info("Developer task complete → committing")
    return "commit"


def route_after_qa(state: dict[str, Any]) -> str:
    """Route after QA agent runs tests.

    - If tests passed → commit
    - If tests failed and retries remain → back to developer (feedback loop)
    - If retries exhausted → escalate to human
    """
    qa_result = state.get("qa_result")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if qa_result and qa_result.passed:
        logger.info("QA passed → committing")
        return "commit"
    elif retry_count < max_retries:
        logger.info(f"QA failed, retry {retry_count + 1}/{max_retries} → developer fix")
        return "developer"
    else:
        logger.warning(f"QA failed, retries exhausted ({retry_count}/{max_retries}) → escalate")
        return "escalate"


def route_after_escalate(state: dict[str, Any]) -> str:
    """Route after escalation node.

    If user requested a retry (retry_count was reset to 0 and task not
    in failed_tasks), route back to developer. Otherwise, end the pipeline.
    """
    tasks = state.get("tasks", [])
    index = state.get("current_task_index", 0)
    failed = set(state.get("failed_tasks", []))

    if index < len(tasks):
        task_id = str(tasks[index].id) if hasattr(tasks[index], "id") else str(tasks[index].get("id", ""))
        if task_id not in failed:
            logger.info("Escalation: user requested retry -> developer")
            return "developer"

    logger.info("Escalation: task failed -> END")
    return "end"
