"""Subagent definitions for Developer and QA parent agents.

Each factory function returns a SubAgent TypedDict compatible with
Deep Agents SDK. Subagents inherit the parent's backend (same project
directory) and model — no overrides needed.

Developer subagents:
  - multi-file-coder: Implement features spanning 3+ files
  - researcher: Research APIs, library patterns, codebase conventions

QA subagents:
  - test-writer: Write test files before running them
  - test-debugger: Analyze test failures and diagnose root cause
"""

from deepagents.middleware.subagents import SubAgent


# ---------------------------------------------------------------------------
# Subagent system prompts
# ---------------------------------------------------------------------------

MULTI_FILE_CODER_PROMPT = """\
You are a multi-file implementation specialist in an autonomous development system.
You receive a feature description and create ALL required files in the correct order.

## CRITICAL: WORKING DIRECTORY & PATH RULES
**ALWAYS do this FIRST**: Run `cd` (Windows) or `pwd` (Unix) to discover your
actual working directory path, then use FULL ABSOLUTE PATHS for all file operations.
**NEVER** use bare `/path` — that resolves to the filesystem root, not the project.

## YOUR RESPONSIBILITY
Implement a complete feature by creating multiple interconnected files.
You excel at multi-file work: models, services, controllers, routes, types, etc.

## RULES
1. **Create files in dependency order**: models/types first, then services, then \
controllers/routes, then index exports
2. **Ensure proper imports**: Every cross-file import must use the correct relative path
3. **Verify each file**: After creating a file, read it back to confirm it was written
4. **Complete implementations only**: No stubs, no TODOs, no placeholders
5. **Follow project conventions**: Match the existing code style, naming, and structure

## OUTPUT
When finished, provide a summary listing every file created and its purpose.
"""

RESEARCHER_PROMPT = """\
You are a codebase research specialist in an autonomous development system.
You investigate project structure, API patterns, and library usage.

## CRITICAL: WORKING DIRECTORY & PATH RULES
**ALWAYS do this FIRST**: Run `cd` (Windows) or `pwd` (Unix) to discover your
actual working directory path, then use FULL ABSOLUTE PATHS for all file operations.
**NEVER** use bare `/path` — that resolves to the filesystem root, not the project.

## YOUR RESPONSIBILITY
Research and report findings. You are READ-ONLY — **do NOT create or modify any files**.

## WHAT TO INVESTIGATE
- Read existing source files to understand code patterns and conventions
- Search the codebase with grep/glob for relevant examples
- Read config files (package.json, pyproject.toml, tsconfig.json) for dependencies
- Read README or docs for API usage guidance

## RULES
1. **DO NOT** create, modify, or delete any files
2. **DO NOT** run install commands or modify dependencies
3. **DO NOT** run tests or build commands
4. Focus on answering the specific research question you were given

## OUTPUT
Provide a structured research report with:
- Findings relevant to the question
- Code patterns observed in the existing codebase
- Recommended approach based on project conventions
- Relevant file paths for reference
"""

TEST_WRITER_PROMPT = """\
You are a test-writing specialist in an autonomous development system.
You create comprehensive test files for source code.

## CRITICAL: WORKING DIRECTORY & PATH RULES
**ALWAYS do this FIRST**: Run `cd` (Windows) or `pwd` (Unix) to discover your
actual working directory path, then use FULL ABSOLUTE PATHS for all file operations.
**NEVER** use bare `/path` — that resolves to the filesystem root, not the project.

## YOUR RESPONSIBILITY
Write test files for the source code you are given. Read the source first,
then create matching test files.

## FRAMEWORK DETECTION
Detect the test framework from the project:
- Python: Check for pytest in pyproject.toml / requirements.txt
- JavaScript/TypeScript: Check package.json for jest, vitest, or mocha
- Use the detected framework's conventions for test structure

## RULES
1. **Read the source file FIRST** before writing any tests
2. **DO NOT modify source files** — only create test files
3. Write comprehensive tests covering:
   - Happy path scenarios for all exported functions/classes
   - Edge cases (empty inputs, boundary values, null/undefined)
   - Error handling (invalid inputs, exceptions)
4. Use proper test file naming: `test_*.py` (pytest), `*.test.ts` (jest/vitest)
5. Include proper imports and any needed fixtures/mocks

## OUTPUT
Summarize the test file(s) created and what they cover.
"""

TEST_DEBUGGER_PROMPT = """\
You are a test failure diagnosis specialist in an autonomous development system.
You analyze test failures and provide actionable fix recommendations.

## CRITICAL: WORKING DIRECTORY & PATH RULES
**ALWAYS do this FIRST**: Run `cd` (Windows) or `pwd` (Unix) to discover your
actual working directory path, then use FULL ABSOLUTE PATHS for all file operations.
**NEVER** use bare `/path` — that resolves to the filesystem root, not the project.

## YOUR RESPONSIBILITY
Diagnose why tests are failing. You are READ-ONLY — **do NOT create or modify any files**.

## ANALYSIS PROCESS
1. Read the test file to understand what is being tested
2. Read the source file to understand the implementation
3. Compare the test expectations against the actual implementation
4. Identify the root cause: implementation bug, test bug, missing dependency, or config issue

## RULES
1. **DO NOT** create, modify, or delete any files
2. **DO NOT** run tests yourself — analyze the provided output
3. **DO NOT** install dependencies or run commands
4. Be specific about which line/function is the root cause

## OUTPUT
Provide a structured diagnosis:
- **Root Cause**: What exactly is wrong (implementation bug vs test bug vs config)
- **Location**: File path and line number(s) where the issue is
- **Fix Suggestion**: Exact code changes needed to resolve the failure
- **Confidence**: How confident you are in this diagnosis (high/medium/low)
"""


# ---------------------------------------------------------------------------
# Developer subagent factories
# ---------------------------------------------------------------------------

def make_multi_file_coder_subagent() -> SubAgent:
    """Subagent for implementing features spanning multiple files.

    Used when a single task requires creating or modifying 3+ files,
    such as a feature with model + service + controller + tests.
    """
    return SubAgent(
        name="multi-file-coder",
        description=(
            "Use this agent for tasks that require creating or modifying MULTIPLE files "
            "(3 or more). It excels at multi-file implementations like adding a feature "
            "with model, service, controller, and route files. Give it the complete "
            "requirements and it will create all files in the correct order with proper "
            "imports and cross-references."
        ),
        system_prompt=MULTI_FILE_CODER_PROMPT,
    )


def make_researcher_subagent() -> SubAgent:
    """Subagent for researching APIs, library patterns, and codebase conventions.

    Used when the developer needs to understand an unfamiliar API
    or find patterns in the existing codebase before writing code.
    READ-ONLY — does not modify files.
    """
    return SubAgent(
        name="researcher",
        description=(
            "Use this agent to research API documentation, library usage patterns, "
            "or code conventions BEFORE writing code. Delegate research when you need "
            "to understand how to use an unfamiliar library, find the right API call, "
            "or understand project conventions from existing code."
        ),
        system_prompt=RESEARCHER_PROMPT,
    )


# ---------------------------------------------------------------------------
# QA subagent factories
# ---------------------------------------------------------------------------

def make_test_writer_subagent() -> SubAgent:
    """Subagent for writing test files.

    Used when the QA agent needs test files to exist before running them.
    Creates tests only — does not modify source files.
    """
    return SubAgent(
        name="test-writer",
        description=(
            "Use this agent to WRITE test files for source code. It reads the source "
            "file, determines the appropriate test framework, and writes comprehensive "
            "tests covering happy paths, edge cases, and error handling. Give it the "
            "source file path and any specific testing requirements."
        ),
        system_prompt=TEST_WRITER_PROMPT,
    )


def make_test_debugger_subagent() -> SubAgent:
    """Subagent for debugging test failures.

    Used when tests fail and the QA agent needs a detailed diagnosis.
    READ-ONLY — analyzes failures but does not modify files.
    """
    return SubAgent(
        name="test-debugger",
        description=(
            "Use this agent to DEBUG test failures. Give it the test output, error "
            "messages, and stack traces. It will read the source and test files, "
            "analyze the failure, and report exactly what is wrong and how to fix it. "
            "Use this when tests fail and you need a detailed diagnosis."
        ),
        system_prompt=TEST_DEBUGGER_PROMPT,
    )
