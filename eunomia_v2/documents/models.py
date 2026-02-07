"""Pydantic models for document templates, sections, and generated documents."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocType(str, Enum):
    """Supported document types."""

    PRD = "prd"
    TDD = "tdd"
    ADR = "adr"
    README = "readme"
    TEST_PLAN = "test_plan"
    RUNBOOK = "runbook"


class RequiredContext(BaseModel):
    """A context variable required by a template."""

    name: str
    description: str = ""
    required: bool = True
    context_type: str = "string"  # string | list | dict
    default_value: str | None = None


class SectionDef(BaseModel):
    """A section definition within a document template."""

    id: str
    title: str
    level: int = 2
    description: str = ""
    required: bool = True
    min_words: int = 0
    max_words: int = 0
    context_keys: list[str] = Field(default_factory=list)
    prompt: str = ""
    subsections: list["SectionDef"] = Field(default_factory=list)
    validation_rules: list[str] = Field(default_factory=list)


class DocumentTemplate(BaseModel):
    """A complete document template loaded from YAML."""

    id: str
    name: str
    doc_type: str
    description: str = ""
    version: str = "1.0.0"
    required_context: list[RequiredContext] = Field(default_factory=list)
    output_formats: list[str] = Field(default_factory=lambda: ["markdown"])
    default_format: str = "markdown"
    quality_threshold: float = 0.85
    max_iterations: int = 3
    style_guide: str = ""
    sections: list[SectionDef] = Field(default_factory=list)
    metadata_defaults: dict[str, Any] = Field(default_factory=dict)

    def get_required_context_names(self) -> list[str]:
        """Return names of required context variables."""
        return [c.name for c in self.required_context if c.required]

    def get_section_count(self) -> int:
        """Total sections including subsections."""
        count = 0
        for section in self.sections:
            count += 1 + self._count_subsections(section)
        return count

    def _count_subsections(self, section: SectionDef) -> int:
        """Recursively count subsections."""
        count = len(section.subsections)
        for sub in section.subsections:
            count += self._count_subsections(sub)
        return count


class GeneratedSection(BaseModel):
    """A single generated section of a document."""

    section_id: str
    title: str
    level: int = 2
    content: str = ""
    word_count: int = 0


class GeneratedDocument(BaseModel):
    """A fully generated document."""

    doc_type: str
    template_id: str
    title: str
    sections: list[GeneratedSection] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_markdown(self) -> str:
        """Render all sections to a single markdown string."""
        lines: list[str] = []
        for section in self.sections:
            heading = "#" * section.level
            lines.append(f"{heading} {section.title}")
            lines.append("")
            if section.content:
                lines.append(section.content)
                lines.append("")
        return "\n".join(lines)
