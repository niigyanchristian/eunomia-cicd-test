"""Task models for Eunomia V2 — ported and refined from V1."""

from enum import Enum

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Task type taxonomy — determines which agent handles the task."""

    INFRASTRUCTURE = "infrastructure"
    FEATURE = "feature"
    TEST_WRITING = "test_writing"
    QA_VALIDATION = "qa_validation"
    COMMIT = "commit"
    ARCHITECTURE = "architecture"
    CICD_SETUP = "cicd_setup"


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Task(BaseModel):
    """A single task in the execution pipeline."""

    id: int
    title: str
    description: str
    task_type: TaskType
    filename: str = ""
    dependencies: list[int] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: str = ""
    retry_count: int = 0
    output: str = ""
    error: str = ""


class TaskGraph(BaseModel):
    """The full task graph generated from a PRD."""

    tasks: list[Task] = Field(default_factory=list)
    project_name: str = ""
    prd_summary: str = ""

    def get_next_pending(self) -> Task | None:
        """Get the next task whose dependencies are all completed."""
        completed_ids = {t.id for t in self.tasks if t.status == TaskStatus.COMPLETED}
        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                if all(dep in completed_ids for dep in task.dependencies):
                    return task
        return None

    def mark_completed(self, task_id: int, output: str = "") -> None:
        """Mark a task as completed."""
        for task in self.tasks:
            if task.id == task_id:
                task.status = TaskStatus.COMPLETED
                task.output = output
                return

    def mark_failed(self, task_id: int, error: str = "") -> None:
        """Mark a task as failed."""
        for task in self.tasks:
            if task.id == task_id:
                task.status = TaskStatus.FAILED
                task.error = error
                return
