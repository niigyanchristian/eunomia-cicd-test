"""Result models for agent outputs."""

from pydantic import BaseModel, Field


class DeveloperOutput(BaseModel):
    """Output from the Developer agent."""

    success: bool = False
    files_created: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    commands_run: list[str] = Field(default_factory=list)
    output: str = ""
    error: str = ""


class QAResult(BaseModel):
    """Output from the QA agent."""

    passed: bool = False
    total_tests: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    framework_detected: str = ""
    output: str = ""
    errors: list[str] = Field(default_factory=list)


class ArchitectDecision(BaseModel):
    """A decision made by the Architect agent."""

    topic: str = ""
    decision: str = ""
    rationale: str = ""
    alternatives_considered: list[str] = Field(default_factory=list)
