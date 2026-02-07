"""YAML template loader and registry for document generation."""

import logging
from pathlib import Path

import yaml

from eunomia_v2.documents.models import DocumentTemplate

logger = logging.getLogger(__name__)

BUILTIN_DIR = Path(__file__).parent / "templates"

# Module-level cache
_template_cache: dict[str, DocumentTemplate] = {}


def load_template(doc_type: str) -> DocumentTemplate:
    """Load a YAML template by doc_type (prd, tdd, adr, readme, test_plan, runbook).

    Raises:
        FileNotFoundError: If the template YAML file doesn't exist.
        ValueError: If the YAML is invalid or doesn't match the schema.
    """
    if doc_type in _template_cache:
        return _template_cache[doc_type]

    yaml_path = BUILTIN_DIR / f"{doc_type}.yaml"
    if not yaml_path.exists():
        available = list_templates()
        raise FileNotFoundError(
            f"Template '{doc_type}' not found. Available: {', '.join(available)}"
        )

    raw = yaml_path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)

    if not isinstance(data, dict):
        raise ValueError(f"Template '{doc_type}' is not a valid YAML mapping")

    template = DocumentTemplate(**data)
    _template_cache[doc_type] = template
    logger.debug("Loaded template: %s (%s)", template.name, template.id)
    return template


def list_templates() -> list[str]:
    """Return available template doc_types (filenames without .yaml)."""
    if not BUILTIN_DIR.exists():
        return []
    return sorted(p.stem for p in BUILTIN_DIR.glob("*.yaml"))


def validate_context(template: DocumentTemplate, context: dict) -> list[str]:
    """Check required context variables are present.

    Returns:
        List of missing required field names. Empty list = all good.
    """
    missing: list[str] = []
    for req in template.required_context:
        if req.required and req.name not in context:
            # Check if there's a default
            if req.default_value is None:
                missing.append(req.name)
    return missing


def apply_defaults(template: DocumentTemplate, context: dict) -> dict:
    """Return a new context dict with defaults applied for missing optional fields."""
    result = dict(context)
    for req in template.required_context:
        if req.name not in result and req.default_value is not None:
            result[req.name] = req.default_value
    return result


def clear_cache() -> None:
    """Clear the template cache (useful for testing)."""
    _template_cache.clear()
