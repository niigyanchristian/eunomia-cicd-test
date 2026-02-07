"""Architect Agent — reviews code structure and makes design decisions.

Uses Deep Agents with filesystem + shell tools (read-only analysis).
Scope-constrained to only READ and ANALYZE code, never modify it.

Architecture:
    1. Creates a Deep Agent with LocalShellBackend (read + shell)
    2. Sends review prompt with focus areas
    3. Agent explores project autonomously (reads files, runs analysis commands)
    4. Parses structured ARCHITECTURE REVIEW output into ArchitectDecision
"""

import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from eunomia_v2.models.results import ArchitectDecision
from eunomia_v2.models.task import Task
from eunomia_v2.prompts.architect import ARCHITECT_SYSTEM_PROMPT, build_architect_prompt
from eunomia_v2.utils.paths import to_posix

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "anthropic:claude-sonnet-4-5-20250929"


async def review_architecture(
    task: Task,
    project_path: str | Path,
    model: str = DEFAULT_MODEL,
    existing_decisions: list[str] | None = None,
) -> ArchitectDecision:
    """Run an architecture review using a Deep Agent.

    The agent operates in read-only mode — it explores the project
    structure, reads code, runs analysis commands (line counts, grep, etc.),
    and produces a structured architectural review.

    Args:
        task: The architecture review task.
        project_path: Absolute path to the project directory.
        model: LangChain model string.
        existing_decisions: Previous architectural decisions for context.

    Returns:
        ArchitectDecision with topic, decision, rationale, and alternatives.
    """
    from deepagents import create_deep_agent
    from deepagents.backends import LocalShellBackend
    from langgraph.checkpoint.memory import MemorySaver

    project_path = Path(project_path).resolve()

    logger.info(f"Architecture review [{task.id}] {task.title} in {project_path}")

    # Build the review prompt
    review_prompt = build_architect_prompt(
        task_title=task.title,
        task_description=task.description,
        project_path=to_posix(project_path),
        focus_areas=task.acceptance_criteria,
        existing_decisions=existing_decisions,
    )

    # Create backend — LocalShellBackend for file reading + analysis commands
    backend = LocalShellBackend(
        root_dir=str(project_path), virtual_mode=True, inherit_env=True,
    )

    # Create the Deep Agent
    agent = create_deep_agent(
        model=model,
        backend=backend,
        system_prompt=ARCHITECT_SYSTEM_PROMPT,
        name=f"architect-task-{task.id}",
        checkpointer=MemorySaver(),
        debug=False,
    )

    logger.info("Architect Deep Agent created, starting review...")

    # Run the agent
    config = {"configurable": {"thread_id": f"architect-task-{task.id}"}}
    input_msg = {"messages": [HumanMessage(content=review_prompt)]}

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

    except Exception as e:
        logger.error(f"Architect agent execution failed: {e}")
        return ArchitectDecision(
            topic=task.title,
            decision=f"Review failed: {e}",
            rationale="Agent execution error",
        )

    # Parse the structured output
    combined_output = "\n".join(all_messages)
    result = parse_architect_output(combined_output, default_topic=task.title)

    logger.info(
        f"Architect [{task.id}] complete: topic={result.topic}, "
        f"alternatives={len(result.alternatives_considered)}"
    )

    return result


def parse_architect_output(
    output: str, default_topic: str = "Architecture Review"
) -> ArchitectDecision:
    """Parse ARCHITECTURE REVIEW format into an ArchitectDecision.

    Looks for the structured block:
        ARCHITECTURE REVIEW:
        - Topic: <what was reviewed>
        - Decision: <recommendation>
        - Rationale: <why>
        - Alternatives: <other approaches>
        - Issues Found: <list or "None">
        - Quality Score: <1-10>

    Args:
        output: Raw combined output from the architect agent.
        default_topic: Fallback topic if not found in output.

    Returns:
        Populated ArchitectDecision.
    """
    topic = default_topic
    decision = ""
    rationale = ""
    alternatives: list[str] = []

    # Find the ARCHITECTURE REVIEW block
    match = re.search(r"ARCHITECTURE REVIEW:", output)
    if match:
        block = output[match.start():]

        topic_match = re.search(r"Topic:\s*(.+?)(?:\n|$)", block)
        decision_match = re.search(r"Decision:\s*(.+?)(?:\n|$)", block)
        rationale_match = re.search(r"Rationale:\s*(.+?)(?:\n|$)", block)
        alternatives_match = re.search(r"Alternatives:\s*(.+?)(?:\n|$)", block)
        issues_match = re.search(r"Issues Found:\s*(.+?)(?:\n|$)", block)

        if topic_match:
            topic = topic_match.group(1).strip()
        if decision_match:
            decision = decision_match.group(1).strip()
        if rationale_match:
            rationale = rationale_match.group(1).strip()
        if alternatives_match:
            alt_text = alternatives_match.group(1).strip()
            if alt_text.lower() != "none":
                alternatives = [a.strip() for a in alt_text.split(",") if a.strip()]
        if issues_match:
            issues_text = issues_match.group(1).strip()
            if issues_text.lower() != "none":
                # Append issues to rationale for visibility
                rationale += f"\n\nIssues found: {issues_text}"
    else:
        # No structured format — use the last substantive message as the decision
        decision = output[-1500:] if len(output) > 1500 else output

    return ArchitectDecision(
        topic=topic,
        decision=decision,
        rationale=rationale,
        alternatives_considered=alternatives,
    )
