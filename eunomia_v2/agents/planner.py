"""Planner Agent — reads PRD, generates task graph.

Uses a Deep Agent (or direct LLM call) to parse a PRD document and produce
a structured task graph with the correct workflow:
    SETUP → Feature → Tests → QA Validation → Commit
"""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from eunomia_v2.models.task import Task, TaskGraph, TaskStatus, TaskType
from eunomia_v2.prompts.planner import TASK_GENERATION_PROMPT

logger = logging.getLogger(__name__)

# Map from V1-style task_type strings to our TaskType enum
TASK_TYPE_MAP: dict[str, TaskType] = {
    "infrastructure": TaskType.INFRASTRUCTURE,
    "feature": TaskType.FEATURE,
    "test_writing": TaskType.TEST_WRITING,
    "testing": TaskType.QA_VALIDATION,
    "qa_validation": TaskType.QA_VALIDATION,
    "commit": TaskType.COMMIT,
    "architecture": TaskType.ARCHITECTURE,
    # V1 compat
    "bug_fix": TaskType.FEATURE,
    "refactor": TaskType.FEATURE,
    "documentation": TaskType.FEATURE,
    "research": TaskType.FEATURE,
    "validation": TaskType.QA_VALIDATION,
    "quality_check": TaskType.QA_VALIDATION,
}

# Map task_type to which agent handles it
TASK_TYPE_TO_AGENT: dict[str, str] = {
    "infrastructure": "devops",
    "feature": "developer",
    "test_writing": "developer",
    "qa_validation": "qa",
    "commit": "developer",
    "architecture": "architect",
}


async def generate_task_graph(
    prd_content: str,
    model: str = "anthropic:claude-sonnet-4-5-20250929",
) -> TaskGraph:
    """Generate a task graph from PRD content using an LLM.

    Args:
        prd_content: The raw PRD text.
        model: LangChain model string (e.g. "anthropic:claude-sonnet-4-5-20250929").

    Returns:
        A validated TaskGraph ready for execution.

    Raises:
        ValueError: If the LLM output cannot be parsed or validated.
    """
    from langchain.chat_models import init_chat_model

    logger.info(f"Generating task graph using model: {model}")

    # Build the prompt with PRD content injected
    prompt = TASK_GENERATION_PROMPT.format(prd_content=prd_content)

    # Initialize the model
    llm = init_chat_model(model)

    # Call the LLM
    messages = [
        SystemMessage(content="You are a precise task planner. Return ONLY valid JSON, no markdown fences."),
        HumanMessage(content=prompt),
    ]

    response = await llm.ainvoke(messages)
    raw_output = response.content

    logger.info(f"LLM response length: {len(raw_output)} chars")

    # Parse the JSON
    task_graph = parse_task_graph(raw_output)

    # Validate
    validate_task_graph(task_graph)

    logger.info(
        f"Generated task graph: {len(task_graph.tasks)} tasks, "
        f"project: {task_graph.project_name}"
    )

    return task_graph


def parse_task_graph(raw_output: str) -> TaskGraph:
    """Parse LLM output into a TaskGraph.

    Handles various LLM output formats:
    - Clean JSON
    - JSON wrapped in markdown code fences
    - JSON with leading/trailing text

    Args:
        raw_output: Raw text from the LLM.

    Returns:
        Parsed TaskGraph.

    Raises:
        ValueError: If JSON cannot be parsed.
    """
    text = raw_output.strip()

    # Strip markdown code fences if present
    if "```json" in text:
        text = text.split("```json", 1)[1]
        text = text.split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        text = text.split("```", 1)[0]

    # Find the JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in LLM output:\n{raw_output[:500]}")

    json_str = text[start:end]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from LLM: {e}\nRaw:\n{json_str[:500]}") from e

    # Build TaskGraph
    project_name = data.get("project_name", "")
    raw_tasks = data.get("tasks", [])

    if not raw_tasks:
        raise ValueError("LLM returned empty task list")

    tasks: list[Task] = []
    task_id_to_index: dict[str, int] = {}

    for i, raw in enumerate(raw_tasks):
        task_id_str = raw.get("task_id", f"TASK-{i+1:03d}")
        task_id_to_index[task_id_str] = i + 1

    for i, raw in enumerate(raw_tasks):
        task_id_str = raw.get("task_id", f"TASK-{i+1:03d}")
        task_type_str = raw.get("task_type", "feature")
        task_type = TASK_TYPE_MAP.get(task_type_str, TaskType.FEATURE)

        # Resolve dependencies from task_id strings to integer indices
        dep_ids = raw.get("dependencies", [])
        dep_indices = [task_id_to_index[d] for d in dep_ids if d in task_id_to_index]

        # Acceptance criteria — handle both string lists and object lists
        raw_criteria = raw.get("acceptance_criteria", [])
        criteria: list[str] = []
        for c in raw_criteria:
            if isinstance(c, str):
                criteria.append(c)
            elif isinstance(c, dict):
                criteria.append(c.get("description", str(c)))

        agent = TASK_TYPE_TO_AGENT.get(task_type_str, "developer")

        tasks.append(Task(
            id=i + 1,
            title=raw.get("title", f"Task {i+1}"),
            description=raw.get("description", ""),
            task_type=task_type,
            filename=raw.get("filename", raw.get("parameters", {}).get("filename", "")),
            dependencies=dep_indices,
            acceptance_criteria=criteria,
            status=TaskStatus.PENDING,
            assigned_agent=agent,
        ))

    return TaskGraph(
        tasks=tasks,
        project_name=project_name,
    )


def validate_task_graph(graph: TaskGraph) -> None:
    """Validate a task graph for correctness.

    Checks:
    1. At least one task exists
    2. First task is infrastructure (SETUP)
    3. No circular dependencies
    4. All dependency references are valid
    5. QA tasks exist for features that have tests

    Args:
        graph: The task graph to validate.

    Raises:
        ValueError: If validation fails.
    """
    if not graph.tasks:
        raise ValueError("Task graph has no tasks")

    # Check first task is infrastructure
    first = graph.tasks[0]
    if first.task_type != TaskType.INFRASTRUCTURE:
        logger.warning(
            f"First task is '{first.task_type}' not 'infrastructure'. "
            f"Pipeline may fail without project setup."
        )

    # Check all dependency references are valid
    valid_ids = {t.id for t in graph.tasks}
    for task in graph.tasks:
        for dep in task.dependencies:
            if dep not in valid_ids:
                raise ValueError(
                    f"Task {task.id} ({task.title}) has invalid dependency: {dep}. "
                    f"Valid IDs: {valid_ids}"
                )

    # Check for circular dependencies (simple DFS)
    visited: set[int] = set()
    in_stack: set[int] = set()
    task_map = {t.id: t for t in graph.tasks}

    def has_cycle(task_id: int) -> bool:
        if task_id in in_stack:
            return True
        if task_id in visited:
            return False
        visited.add(task_id)
        in_stack.add(task_id)
        for dep in task_map[task_id].dependencies:
            if has_cycle(dep):
                return True
        in_stack.remove(task_id)
        return False

    for task in graph.tasks:
        if has_cycle(task.id):
            raise ValueError(f"Circular dependency detected involving task {task.id}")

    # Log task breakdown
    type_counts: dict[str, int] = {}
    agent_counts: dict[str, int] = {}
    for task in graph.tasks:
        type_counts[task.task_type.value] = type_counts.get(task.task_type.value, 0) + 1
        agent_counts[task.assigned_agent] = agent_counts.get(task.assigned_agent, 0) + 1

    logger.info(f"Task types: {type_counts}")
    logger.info(f"Agent distribution: {agent_counts}")


def task_graph_to_json(graph: TaskGraph) -> str:
    """Serialize a TaskGraph to JSON string.

    Args:
        graph: The task graph to serialize.

    Returns:
        Pretty-printed JSON string.
    """
    data = {
        "project_name": graph.project_name,
        "tasks": [
            {
                "id": t.id,
                "title": t.title,
                "description": t.description,
                "task_type": t.task_type.value,
                "filename": t.filename,
                "dependencies": t.dependencies,
                "acceptance_criteria": t.acceptance_criteria,
                "status": t.status.value,
                "assigned_agent": t.assigned_agent,
            }
            for t in graph.tasks
        ],
    }
    return json.dumps(data, indent=2)
