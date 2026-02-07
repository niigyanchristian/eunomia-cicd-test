"""LangGraph orchestrator — the core execution graph for Eunomia V2.

Wires Planner -> Developer -> QA -> Commit agents into a LangGraph StateGraph
with conditional routing and a QA feedback loop (developer <- QA retry).

Graph structure:
    planner -> task_router -> [developer | qa | devops | architect | commit | done]
    developer -> [qa (retry) | commit (normal)]
    qa -> [commit (pass) | developer (retry) | escalate (max retries)]
    devops -> commit
    architect -> commit
    commit -> task_router
    escalate -> [developer (user retry) | END (user skip)]
"""

import logging
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from eunomia_v2.graph.routing import (
    route_after_developer,
    route_after_escalate,
    route_after_qa,
    route_task_to_agent,
)
from eunomia_v2.graph.state import EunomiaState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build the Eunomia V2 orchestration graph."""
    graph = StateGraph(EunomiaState)

    # Nodes
    graph.add_node("planner", _planner_node)
    graph.add_node("task_router", _task_router_node)
    graph.add_node("developer", _developer_node)
    graph.add_node("qa", _qa_node)
    graph.add_node("devops", _devops_node)
    graph.add_node("architect", _architect_node)
    graph.add_node("commit", _commit_node)
    graph.add_node("escalate", _escalate_node)

    # Entry point
    graph.set_entry_point("planner")

    # Planner -> Task Router
    graph.add_edge("planner", "task_router")

    # Task Router -> Agent (conditional on task_type)
    graph.add_conditional_edges(
        "task_router",
        route_task_to_agent,
        {
            "developer": "developer",
            "qa": "qa",
            "devops": "devops",
            "architect": "architect",
            "commit": "commit",
            "done": END,
        },
    )

    # Developer -> conditional (retry loop or commit)
    graph.add_conditional_edges(
        "developer",
        route_after_developer,
        {
            "qa": "qa",
            "commit": "commit",
        },
    )

    # QA -> conditional (pass/fail/escalate)
    graph.add_conditional_edges(
        "qa",
        route_after_qa,
        {
            "commit": "commit",
            "developer": "developer",
            "escalate": "escalate",
        },
    )

    # DevOps / Architect -> Commit (mark task done, then next)
    graph.add_edge("devops", "commit")
    graph.add_edge("architect", "commit")

    # Commit -> Task Router (pick next task)
    graph.add_edge("commit", "task_router")

    # Escalate -> conditional (user retry or END)
    graph.add_conditional_edges(
        "escalate",
        route_after_escalate,
        {
            "developer": "developer",
            "end": END,
        },
    )

    return graph


def compile_graph(checkpointer: Any = None) -> Any:
    """Compile the graph with optional checkpointing.

    Args:
        checkpointer: LangGraph checkpointer (MemorySaver, SqliteSaver, etc.)
                      If None, uses in-memory checkpointer.

    Returns:
        Compiled LangGraph graph ready for execution.
    """
    graph = build_graph()
    if checkpointer is None:
        checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

async def _planner_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run the Planner agent: PRD -> task graph.

    In approval/interactive HITL modes, interrupts after generating the
    plan so the user can approve or provide feedback. Uses cached_plan
    to avoid re-calling the LLM when the node restarts on resume.
    """
    from eunomia_v2.agents.planner import generate_task_graph

    prd_content = state.get("prd_content", "")
    model = state.get("model", "anthropic:claude-sonnet-4-5-20250929")
    hitl_level = state.get("hitl_level", "autonomous")

    if not prd_content:
        logger.error("No PRD content in state -- cannot plan")
        return {"tasks": []}

    # Idempotency guard: on resume, cached_plan is populated — skip the LLM call
    cached = state.get("cached_plan", [])
    if cached:
        tasks = cached
        logger.info(f"PLANNER: Using cached plan ({len(tasks)} tasks)")
    else:
        logger.info("=" * 60)
        logger.info("PLANNER: Generating task graph from PRD...")
        logger.info("=" * 60)

        task_graph = await generate_task_graph(prd_content, model=model)
        tasks = task_graph.tasks

        logger.info(f"PLANNER: Generated {len(tasks)} tasks")
        for t in tasks:
            logger.info(f"  [{t.id}] {t.task_type.value:18s} | {t.title}")

    # INTERRUPT: Plan approval (approval + interactive modes)
    if hitl_level in ("approval", "interactive"):
        task_summary = "\n".join(
            f"  [{t.id}] {t.task_type.value:18s} | {t.title}"
            for t in tasks
        )
        user_response = interrupt({
            "type": "plan_approval",
            "message": (
                f"Generated {len(tasks)} tasks:\n{task_summary}\n\n"
                f"Type 'approve' to proceed, or provide feedback to revise."
            ),
            "tasks": [
                {"id": t.id, "title": t.title, "task_type": t.task_type.value}
                for t in tasks
            ],
        })

        if isinstance(user_response, str) and user_response.strip().lower() != "approve":
            logger.info(f"PLANNER: User feedback: {user_response}")
            return {
                "tasks": tasks,
                "cached_plan": tasks,
                "user_feedback": user_response,
            }

    return {"tasks": tasks, "cached_plan": tasks}


async def _task_router_node(state: dict[str, Any]) -> dict[str, Any]:
    """Find the next runnable task (dependencies met, not completed/failed).

    In interactive HITL mode, interrupts before each task for user approval.
    Marks dependency-blocked tasks as failed so the graph doesn't stall.
    """
    tasks = state.get("tasks", [])
    completed = set(state.get("completed_tasks", []))
    failed = set(state.get("failed_tasks", []))
    hitl_level = state.get("hitl_level", "autonomous")

    for i, task in enumerate(tasks):
        task_id = str(task.id)

        # Skip already finished tasks
        if task_id in completed or task_id in failed:
            continue

        # If any dependency failed, this task is blocked — mark failed
        deps_failed = any(str(d) in failed for d in task.dependencies)
        if deps_failed:
            logger.warning(
                f"Task [{task.id}] {task.title} blocked by failed dependency -- skipping"
            )
            failed.add(task_id)
            continue

        # Check all dependencies are completed
        deps_met = all(str(d) in completed for d in task.dependencies)
        if deps_met:
            # INTERRUPT: Per-task approval (interactive mode only)
            if hitl_level == "interactive":
                tt = task.task_type.value if hasattr(task.task_type, "value") else str(task.task_type)
                user_response = interrupt({
                    "type": "task_approval",
                    "message": (
                        f"Next task: [{task.id}] '{task.title}' ({tt})\n"
                        f"Type 'approve' to proceed, or 'skip' to skip this task."
                    ),
                    "task_id": task.id,
                    "task_title": task.title,
                    "task_type": tt,
                })
                if isinstance(user_response, str) and user_response.strip().lower() == "skip":
                    logger.info(f"TASK ROUTER: User skipped [{task.id}]")
                    failed.add(task_id)
                    continue

            logger.info("-" * 60)
            logger.info(f"TASK ROUTER: Next -> [{task.id}] {task.title} ({task.task_type.value})")
            logger.info("-" * 60)
            return {
                "current_task_index": i,
                "retry_count": 0,
                "qa_result": None,
                "developer_output": None,
                "failed_tasks": list(failed),
            }

    # No more runnable tasks
    logger.info("=" * 60)
    logger.info(
        f"TASK ROUTER: All tasks processed. "
        f"Completed={len(completed)}, Failed={len(failed)}"
    )
    logger.info("=" * 60)
    return {
        "current_task_index": len(tasks),
        "failed_tasks": list(failed),
    }


async def _developer_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run the Developer agent on the current task.

    If a previous QA run failed (retry loop), builds a "fix" prompt
    with the test errors instead of the original task description.
    """
    from eunomia_v2.agents.developer import execute_task
    from eunomia_v2.models.task import Task, TaskType

    tasks = state.get("tasks", [])
    index = state.get("current_task_index", 0)
    project_path = state.get("project_path", "")
    model = state.get("model", "anthropic:claude-sonnet-4-5-20250929")

    task = tasks[index]
    qa_result = state.get("qa_result")
    retry_count = state.get("retry_count", 0)

    # Check if this is a retry (QA failed, developer needs to fix)
    if qa_result and not qa_result.passed:
        retry_count += 1
        logger.info(f"DEVELOPER (retry {retry_count}): Fixing code for [{task.id}] {task.title}")

        # Build a fix prompt with QA error details
        error_details = "\n".join(qa_result.errors[:5]) if qa_result.errors else "(no specific errors)"
        fix_task = Task(
            id=task.id,
            title=f"Fix: {task.title}",
            description=(
                f"Tests are failing. Read the test output carefully and fix the SOURCE CODE "
                f"(NOT the test files) to make all tests pass.\n\n"
                f"Test output (last 1500 chars):\n"
                f"```\n{qa_result.output[-1500:]}\n```\n\n"
                f"Errors:\n{error_details}"
            ),
            task_type=TaskType.FEATURE,
            filename="",  # Don't constrain — let agent find the right file
        )
        result = await execute_task(fix_task, project_path, model=model)
    else:
        logger.info(f"DEVELOPER: Executing [{task.id}] {task.title}")
        result = await execute_task(task, project_path, model=model)

    logger.info(
        f"DEVELOPER: Done. success={result.success}, "
        f"files={len(result.files_created)}, cmds={len(result.commands_run)}"
    )

    return {
        "developer_output": result,
        "retry_count": retry_count,
    }


async def _qa_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run the QA agent: execute tests and report results."""
    from eunomia_v2.agents.qa import run_tests

    tasks = state.get("tasks", [])
    index = state.get("current_task_index", 0)
    project_path = state.get("project_path", "")
    model = state.get("model", "anthropic:claude-sonnet-4-5-20250929")

    task = tasks[index]

    logger.info(f"QA: Running tests for [{task.id}] {task.title}")

    result = await run_tests(task, project_path, model=model)

    logger.info(
        f"QA: passed={result.passed}, total={result.total_tests}, "
        f"failed={result.tests_failed}, framework={result.framework_detected}"
    )

    return {"qa_result": result}


async def _devops_node(state: dict[str, Any]) -> dict[str, Any]:
    """Handle infrastructure setup using the dedicated DevOps agent.

    Infrastructure tasks (project init, dependency install, CI/CD config)
    use a DevOps-specific agent with infrastructure-focused prompts.
    """
    from eunomia_v2.agents.devops import setup_infrastructure

    tasks = state.get("tasks", [])
    index = state.get("current_task_index", 0)
    project_path = state.get("project_path", "")
    model = state.get("model", "anthropic:claude-sonnet-4-5-20250929")

    task = tasks[index]

    logger.info(f"DEVOPS: Infrastructure setup [{task.id}] {task.title}")

    result = await setup_infrastructure(task, project_path, model=model)

    logger.info(
        f"DEVOPS: Done. success={result.success}, "
        f"files={len(result.files_created)}, cmds={len(result.commands_run)}"
    )

    return {"developer_output": result}


async def _architect_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run the Architect agent to review code and make design decisions.

    The architect is READ-ONLY — it analyzes the project and produces
    ArchitectDecision objects stored in state.architect_decisions.
    """
    from eunomia_v2.agents.architect import review_architecture

    tasks = state.get("tasks", [])
    index = state.get("current_task_index", 0)
    project_path = state.get("project_path", "")
    model = state.get("model", "anthropic:claude-sonnet-4-5-20250929")

    task = tasks[index]

    # Gather existing decisions as context strings
    existing = state.get("architect_decisions", [])
    existing_strs = [
        f"{d.topic}: {d.decision}" for d in existing
    ] if existing else None

    logger.info(f"ARCHITECT: Reviewing [{task.id}] {task.title}")

    decision = await review_architecture(
        task, project_path, model=model, existing_decisions=existing_strs,
    )

    # Append to architect_decisions list
    decisions = list(existing) if existing else []
    decisions.append(decision)

    logger.info(
        f"ARCHITECT: Done. topic={decision.topic}, "
        f"alternatives={len(decision.alternatives_considered)}"
    )

    return {"architect_decisions": decisions}


async def _commit_node(state: dict[str, Any]) -> dict[str, Any]:
    """Mark the current task as completed and create a git commit.

    In interactive HITL mode, interrupts before committing to let the
    user approve or skip the commit. Git failures are non-fatal.
    """
    from eunomia_v2.utils.git import ensure_gitignore, git_add_and_commit, git_init

    tasks = state.get("tasks", [])
    index = state.get("current_task_index", 0)
    project_path = state.get("project_path", "")
    hitl_level = state.get("hitl_level", "autonomous")

    if index >= len(tasks):
        return {}

    task = tasks[index]
    completed = list(state.get("completed_tasks", []))
    completed.append(str(task.id))

    # INTERRUPT: Commit confirmation (interactive mode only)
    skip_commit = False
    if hitl_level == "interactive" and project_path:
        task_type = task.task_type.value if hasattr(task.task_type, "value") else str(task.task_type)
        user_response = interrupt({
            "type": "commit_approval",
            "message": (
                f"About to commit task [{task.id}] '{task.title}' ({task_type}).\n"
                f"Type 'approve' to commit, or 'skip' to skip the commit."
            ),
            "task_id": task.id,
        })
        if isinstance(user_response, str) and user_response.strip().lower() == "skip":
            logger.info(f"COMMIT: User skipped commit for [{task.id}]")
            skip_commit = True

    # Git commit (non-fatal — pipeline continues even if git fails)
    if project_path and not skip_commit:
        task_type = task.task_type.value if hasattr(task.task_type, "value") else str(task.task_type)
        commit_prefix = {
            "infrastructure": "chore",
            "feature": "feat",
            "test_writing": "test",
            "qa_validation": "test",
            "architecture": "docs",
        }.get(task_type, "feat")

        commit_msg = f"{commit_prefix}: {task.title}\n\nTask [{task.id}] ({task_type})"

        ensure_gitignore(project_path)
        git_init(project_path)
        committed = git_add_and_commit(project_path, commit_msg)

        if committed:
            logger.info(f"COMMIT: [{task.id}] {task.title} -- git commit created")
        else:
            logger.info(f"COMMIT: [{task.id}] {task.title} -- marked completed (no git commit)")
    else:
        logger.info(f"COMMIT: [{task.id}] {task.title} -- marked completed")

    return {"completed_tasks": completed}


async def _escalate_node(state: dict[str, Any]) -> dict[str, Any]:
    """Escalate a task that failed after max retries.

    In approval/interactive HITL modes, interrupts to let the user
    choose: skip (mark failed), retry (send back to developer), or
    provide new instructions. In autonomous mode, marks failed immediately.
    """
    tasks = state.get("tasks", [])
    index = state.get("current_task_index", 0)
    hitl_level = state.get("hitl_level", "autonomous")

    if index >= len(tasks):
        return {}

    task = tasks[index]
    qa_result = state.get("qa_result")
    retry_count = state.get("retry_count", 0)

    error_summary = ""
    if qa_result:
        errors = qa_result.errors if hasattr(qa_result, "errors") else []
        output = qa_result.output if hasattr(qa_result, "output") else ""
        error_summary = "\n".join(errors[:5]) if errors else output[-500:]

    # INTERRUPT: Escalation (approval + interactive modes)
    if hitl_level in ("approval", "interactive"):
        user_response = interrupt({
            "type": "escalation",
            "message": (
                f"Task [{task.id}] '{task.title}' FAILED after {retry_count} retries.\n\n"
                f"Errors:\n{error_summary}\n\n"
                f"Options: 'skip' to mark failed, 'retry' to try again, "
                f"or provide instructions for the developer."
            ),
            "task_id": task.id,
            "task_title": task.title,
            "retry_count": retry_count,
        })

        if isinstance(user_response, str):
            response_lower = user_response.strip().lower()
            if response_lower == "retry":
                logger.info(f"ESCALATE: User requested retry for [{task.id}]")
                return {"retry_count": 0, "user_feedback": user_response}
            elif response_lower != "skip":
                # User provided custom instructions — reset retry, store feedback
                logger.info(f"ESCALATE: User instructions for [{task.id}]: {user_response}")
                return {"retry_count": 0, "user_feedback": user_response}

    # Default: mark as failed
    failed = list(state.get("failed_tasks", []))
    failed.append(str(task.id))

    logger.error(
        f"ESCALATE: Task [{task.id}] {task.title} FAILED after {retry_count} retries. "
        f"QA errors: {qa_result.errors[:3] if qa_result and hasattr(qa_result, 'errors') else '(none)'}"
    )

    return {"failed_tasks": failed}
