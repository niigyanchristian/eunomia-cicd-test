"""Document generator — LLM-powered section-by-section document creation."""

import logging
from typing import Any

from eunomia_v2.documents.loader import apply_defaults, load_template, validate_context
from eunomia_v2.documents.models import (
    DocumentTemplate,
    GeneratedDocument,
    GeneratedSection,
    SectionDef,
)
from eunomia_v2.prompts.document import (
    SECTION_DRAFT_SYSTEM,
    SECTION_REVIEW_SYSTEM,
    build_review_prompt,
    build_section_prompt,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "anthropic:claude-sonnet-4-5-20250929"


async def generate_document(
    doc_type: str,
    context: dict[str, Any],
    model: str = DEFAULT_MODEL,
    review: bool = True,
) -> GeneratedDocument:
    """Generate a complete document from a template.

    Pipeline: For each section -> Draft (LLM call) -> optional Review (LLM call).
    Uses init_chat_model() from langchain — same pattern as all V2 agents.

    Args:
        doc_type: Template type (prd, tdd, adr, readme, test_plan, runbook).
        context: Context variables for template interpolation.
        model: LangChain model string (provider:model_name).
        review: Whether to run a review pass on each section.

    Returns:
        GeneratedDocument with all sections populated.

    Raises:
        FileNotFoundError: If template doesn't exist.
        ValueError: If required context is missing.
    """
    template = load_template(doc_type)

    # Validate and apply defaults
    missing = validate_context(template, context)
    if missing:
        raise ValueError(
            f"Missing required context for '{doc_type}': {', '.join(missing)}"
        )
    full_context = apply_defaults(template, context)

    # Generate each section
    sections: list[GeneratedSection] = []
    for section_def in template.sections:
        section = await _generate_section_recursive(
            template, section_def, full_context, sections, model, review,
        )
        sections.append(section)

        # Generate subsections
        for sub_def in section_def.subsections:
            sub_section = await _generate_section_recursive(
                template, sub_def, full_context, sections, model, review,
            )
            sections.append(sub_section)

    doc = GeneratedDocument(
        doc_type=doc_type,
        template_id=template.id,
        title=template.name,
        sections=sections,
        metadata=dict(template.metadata_defaults),
    )

    logger.info(
        "Generated %s: %d sections, %d total words",
        doc_type,
        len(sections),
        sum(s.word_count for s in sections),
    )

    return doc


async def _generate_section_recursive(
    template: DocumentTemplate,
    section_def: SectionDef,
    context: dict[str, Any],
    prior_sections: list[GeneratedSection],
    model: str,
    review: bool,
) -> GeneratedSection:
    """Generate a single section, optionally with review."""
    from langchain.chat_models import init_chat_model

    # Draft
    user_prompt = build_section_prompt(template, section_def, context, prior_sections)
    llm = init_chat_model(model)

    logger.debug("Drafting section: %s", section_def.title)
    response = await llm.ainvoke([
        {"role": "system", "content": SECTION_DRAFT_SYSTEM},
        {"role": "user", "content": user_prompt},
    ])

    content = response.content if hasattr(response, "content") else str(response)
    section = parse_section_output(content, section_def)

    # Optional review pass
    if review:
        review_prompt = build_review_prompt(section, section_def, template.style_guide)
        logger.debug("Reviewing section: %s", section_def.title)
        review_response = await llm.ainvoke([
            {"role": "system", "content": SECTION_REVIEW_SYSTEM},
            {"role": "user", "content": review_prompt},
        ])
        reviewed_content = (
            review_response.content
            if hasattr(review_response, "content")
            else str(review_response)
        )
        section = parse_section_output(reviewed_content, section_def)

    return section


def parse_section_output(raw: str, section_def: SectionDef) -> GeneratedSection:
    """Parse raw LLM output into a GeneratedSection.

    Strips leading headings (the heading is added by to_markdown),
    counts words, and enforces max_words truncation.
    """
    content = raw.strip()

    # Strip leading markdown heading if the LLM included one
    lines = content.split("\n")
    if lines and lines[0].startswith("#"):
        content = "\n".join(lines[1:]).strip()

    # Word count
    word_count = len(content.split())

    # Truncate if over max_words
    if section_def.max_words > 0 and word_count > section_def.max_words:
        words = content.split()
        content = " ".join(words[: section_def.max_words])
        word_count = section_def.max_words

    return GeneratedSection(
        section_id=section_def.id,
        title=section_def.title,
        level=section_def.level,
        content=content,
        word_count=word_count,
    )
