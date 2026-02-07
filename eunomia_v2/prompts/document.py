"""Prompts for document section generation and review."""

import re
from typing import Any

from eunomia_v2.documents.models import (
    DocumentTemplate,
    GeneratedSection,
    SectionDef,
)

SECTION_DRAFT_SYSTEM = """You are a technical writer generating documentation sections.
Write clear, professional content following the provided style guide.
Output ONLY the section content — no headings, no meta-commentary, no markdown headers.
The heading will be added automatically."""

SECTION_REVIEW_SYSTEM = """You are a documentation reviewer.
Review the section for clarity, completeness, and adherence to the style guide.
If the section is good, output it unchanged.
If it needs improvement, output the improved version directly.
Output ONLY the improved content — no commentary, no headings."""


def build_section_prompt(
    template: DocumentTemplate,
    section: SectionDef,
    context: dict[str, Any],
    prior_sections: list[GeneratedSection] | None = None,
) -> str:
    """Build the user prompt for generating one document section.

    Interpolates {{variable}} placeholders in the section prompt with
    context values. Includes style guide and word count constraints.
    """
    parts: list[str] = []

    # Document context
    parts.append(f"Document type: {template.doc_type}")
    parts.append(f"Document name: {template.name}")
    parts.append("")

    # Style guide
    if template.style_guide:
        parts.append(f"Style guide: {template.style_guide.strip()}")
        parts.append("")

    # Section details
    parts.append(f"Section: {section.title}")
    if section.description:
        parts.append(f"Description: {section.description}")

    # Word count constraints
    if section.min_words > 0 or section.max_words > 0:
        constraints = []
        if section.min_words > 0:
            constraints.append(f"minimum {section.min_words} words")
        if section.max_words > 0:
            constraints.append(f"maximum {section.max_words} words")
        parts.append(f"Word count: {', '.join(constraints)}")
    parts.append("")

    # Interpolate prompt template with context
    prompt_text = section.prompt.strip() if section.prompt else ""
    if prompt_text:
        prompt_text = _interpolate(prompt_text, context)
        parts.append(f"Instructions:\n{prompt_text}")
        parts.append("")

    # Prior sections for continuity
    if prior_sections:
        prior_summary = "\n".join(
            f"- {s.title}: {s.content[:200]}..." if len(s.content) > 200 else f"- {s.title}: {s.content}"
            for s in prior_sections[-3:]  # last 3 for context
        )
        parts.append(f"Prior sections (for continuity):\n{prior_summary}")
        parts.append("")

    # Context data
    relevant_context = {}
    for key in section.context_keys:
        if key in context:
            relevant_context[key] = context[key]
    if relevant_context:
        ctx_lines = [f"  {k}: {v}" for k, v in relevant_context.items()]
        parts.append(f"Context data:\n" + "\n".join(ctx_lines))

    return "\n".join(parts)


def build_review_prompt(
    section: GeneratedSection,
    section_def: SectionDef,
    style_guide: str,
) -> str:
    """Build the user prompt for reviewing a generated section."""
    parts: list[str] = []

    parts.append(f"Review this section: {section.title}")
    parts.append("")

    if style_guide:
        parts.append(f"Style guide: {style_guide.strip()}")
        parts.append("")

    if section_def.min_words > 0:
        parts.append(f"Minimum words: {section_def.min_words}")
    if section_def.max_words > 0:
        parts.append(f"Maximum words: {section_def.max_words}")
    parts.append("")

    parts.append("Current content:")
    parts.append(section.content)
    parts.append("")

    parts.append(
        "If the content is good and meets all requirements, output it unchanged. "
        "Otherwise, output an improved version."
    )

    return "\n".join(parts)


def _interpolate(text: str, context: dict[str, Any]) -> str:
    """Replace {{variable}} placeholders with context values."""

    def _replacer(match: re.Match) -> str:
        key = match.group(1).strip()
        value = context.get(key, f"{{{{{key}}}}}")
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return str(value)

    return re.sub(r"\{\{(\w+)\}\}", _replacer, text)
