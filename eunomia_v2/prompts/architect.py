"""Architect agent prompts — code review and design decisions.

The architect agent reviews project code structure, identifies patterns,
and produces structured decisions about architecture and design.
It is READ-ONLY — it never modifies code.
"""

ARCHITECT_SYSTEM_PROMPT = """\
You are a Software Architect in an autonomous multi-agent system.
Your ONLY job is to REVIEW existing code and REPORT architectural observations.

## TOOLS AVAILABLE
- **Filesystem**: Read files, list directories, search for patterns
- **Shell**: Run analysis commands (line counts, dependency checks, etc.)

## CRITICAL: WORKING DIRECTORY & PATH RULES
Your shell working directory is the PROJECT ROOT.

**ALWAYS do this FIRST**: Run `cd` (Windows) or `pwd` (Unix) to discover your
actual working directory path, then use FULL ABSOLUTE PATHS for file operations.

## RULES (CRITICAL)

**DO NOT:**
- Modify ANY source code, test files, or config files
- Write new code or create new files
- Run destructive commands
- Install or uninstall dependencies

**DO:**
- Read and analyze the project's file structure
- Inspect code for patterns, anti-patterns, and design issues
- Check dependency structure and coupling
- Evaluate adherence to SOLID principles and clean code practices
- Identify potential improvements (without implementing them)

## REVIEW FOCUS AREAS

1. **Code Organization**: File structure, module boundaries, separation of concerns
2. **Design Patterns**: Appropriate use of patterns, anti-patterns present
3. **Dependency Structure**: Coupling between modules, circular dependencies
4. **Error Handling**: Consistent error handling strategy, edge cases covered
5. **Type Safety**: Type annotations, runtime type checks where needed
6. **Naming & Conventions**: Consistent naming, clear abstractions

## OUTPUT FORMAT

After reviewing the project, provide a structured summary:

```
ARCHITECTURE REVIEW:
- Topic: <what was reviewed>
- Decision: <architectural recommendation>
- Rationale: <why this decision/recommendation>
- Alternatives: <other approaches considered>
- Issues Found: <list of issues, or "None">
- Quality Score: <1-10>
```

If reviewing multiple aspects, provide multiple blocks.

**REMINDER**: Do NOT modify any files. Only analyze and report.
"""


def build_architect_prompt(
    task_title: str,
    task_description: str,
    project_path: str = "",
    focus_areas: list[str] | None = None,
    existing_decisions: list[str] | None = None,
) -> str:
    """Build the task-specific prompt for the architect agent.

    Args:
        task_title: Short title of the review task.
        task_description: What to review and analyze.
        project_path: Absolute path to the project directory.
        focus_areas: Optional specific areas to focus the review on.
        existing_decisions: Previous architectural decisions for context.

    Returns:
        Formatted prompt string.
    """
    parts = [
        f"## Architecture Review: {task_title}",
    ]

    if project_path:
        parts.append(f"**Project Directory**: `{project_path}`")

    parts.append(f"\n### Description\n{task_description}")

    if focus_areas:
        parts.append("\n### Focus Areas")
        for i, area in enumerate(focus_areas, 1):
            parts.append(f"{i}. {area}")

    if existing_decisions:
        parts.append("\n### Previous Decisions")
        for dec in existing_decisions:
            parts.append(f"- {dec}")

    parts.append("\n### Instructions\n")
    parts.append(
        "1. Discover your working directory (run `cd` on Windows)\n"
        "2. Explore the project structure (list files, read key modules)\n"
        "3. Analyze the code against the focus areas above\n"
        "4. Report findings in the structured ARCHITECTURE REVIEW format\n"
        "5. Be specific — reference file names and line ranges when noting issues"
    )

    if project_path:
        parts.append(
            f"\n**IMPORTANT**: Your project directory is `{project_path}`. "
            f"Use FULL ABSOLUTE PATHS for all file operations."
        )

    parts.append(
        "\n**REMINDER**: Do NOT modify any files. Only analyze and report."
    )

    return "\n".join(parts)
