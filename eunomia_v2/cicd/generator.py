"""CI/CD config generator — detect platform, render templates, write files."""

import logging
import re
from pathlib import Path

from eunomia_v2.cicd.models import CICDConfig, CICDPlatform

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"

# Template file names keyed by platform
_TEMPLATE_FILES: dict[CICDPlatform, str] = {
    CICDPlatform.GITHUB_ACTIONS: "github_actions.yml",
    CICDPlatform.GITLAB_CI: "gitlab_ci.yml",
    CICDPlatform.JENKINS: "jenkinsfile",
}

# Language-specific defaults used to fill template variables
_LANGUAGE_DEFAULTS: dict[str, dict[str, str]] = {
    "python": {
        "language_display": "Python",
        "language_var": "PYTHON_VERSION",
        "setup_action": "actions/setup-python@v5",
        "docker_image": "python:3.12-slim",
        "artifacts_path": ".venv/",
        "default_install": "pip install -r requirements.txt",
        "default_test": "pytest --tb=short -q",
        "default_lint": "ruff check . || true",
    },
    "node": {
        "language_display": "Node.js",
        "language_var": "NODE_VERSION",
        "setup_action": "actions/setup-node@v4",
        "docker_image": "node:20-slim",
        "artifacts_path": "node_modules/",
        "default_install": "npm ci",
        "default_test": "npm test",
        "default_lint": "npm run lint || true",
    },
    "go": {
        "language_display": "Go",
        "language_var": "GO_VERSION",
        "setup_action": "actions/setup-go@v5",
        "docker_image": "golang:1.22-alpine",
        "artifacts_path": "bin/",
        "default_install": "go mod download",
        "default_test": "go test ./...",
        "default_lint": "golangci-lint run || true",
    },
    "rust": {
        "language_display": "Rust",
        "language_var": "RUST_VERSION",
        "setup_action": "dtolnay/rust-toolchain@stable",
        "docker_image": "rust:1.77-slim",
        "artifacts_path": "target/",
        "default_install": "cargo build",
        "default_test": "cargo test",
        "default_lint": "cargo clippy || true",
    },
}


def detect_cicd_platform(project_path: str | Path) -> CICDPlatform | None:
    """Detect which CI/CD platform a project already uses.

    Checks for the presence of platform-specific config files.
    Returns None if no CI/CD config is found.
    """
    project = Path(project_path)

    if (project / ".github" / "workflows").is_dir():
        return CICDPlatform.GITHUB_ACTIONS
    if (project / ".gitlab-ci.yml").exists():
        return CICDPlatform.GITLAB_CI
    if (project / "Jenkinsfile").exists():
        return CICDPlatform.JENKINS

    return None


def detect_language(project_path: str | Path) -> str:
    """Detect the primary language of a project from config files.

    Returns one of: python, node, go, rust, or python (default).
    """
    project = Path(project_path)

    if (project / "pyproject.toml").exists() or (project / "requirements.txt").exists():
        return "python"
    if (project / "package.json").exists():
        return "node"
    if (project / "go.mod").exists():
        return "go"
    if (project / "Cargo.toml").exists():
        return "rust"

    return "python"


def build_config(
    platform: CICDPlatform,
    language: str = "python",
    language_version: str = "",
    test_command: str = "",
    lint_command: str = "",
    install_command: str = "",
    branch_triggers: list[str] | None = None,
    deploy_target: str = "",
) -> CICDConfig:
    """Build a CICDConfig with sensible defaults for the given language.

    Args:
        platform: Target CI/CD platform.
        language: Project language (python, node, go, rust).
        language_version: Override version (e.g., "3.12", "20").
        test_command: Override test command.
        lint_command: Override lint command.
        install_command: Override install command.
        branch_triggers: Branches that trigger the pipeline.
        deploy_target: Deploy target name (empty = no deploy).
    """
    defaults = _LANGUAGE_DEFAULTS.get(language, _LANGUAGE_DEFAULTS["python"])

    version_defaults = {
        "python": "3.12",
        "node": "20",
        "go": "1.22",
        "rust": "stable",
    }

    return CICDConfig(
        platform=platform,
        language=language,
        language_version=language_version or version_defaults.get(language, "3.12"),
        test_command=test_command or defaults["default_test"],
        lint_command=lint_command or defaults["default_lint"],
        install_command=install_command or defaults["default_install"],
        branch_triggers=branch_triggers or ["main", "develop"],
        deploy_enabled=bool(deploy_target),
        deploy_target=deploy_target or "production",
    )


def load_template(platform: CICDPlatform) -> str:
    """Load the raw template string for a platform."""
    template_file = TEMPLATES_DIR / _TEMPLATE_FILES[platform]
    return template_file.read_text(encoding="utf-8")


def render_template(config: CICDConfig) -> str:
    """Render a CI/CD template with the given config.

    Replaces {{variable}} placeholders with values from the config
    and language defaults.
    """
    template = load_template(config.platform)
    defaults = _LANGUAGE_DEFAULTS.get(config.language, _LANGUAGE_DEFAULTS["python"])

    # Build the substitution map
    variables: dict[str, str] = {
        "language": config.language,
        "language_display": defaults["language_display"],
        "language_version": config.language_version,
        "language_var": defaults["language_var"],
        "setup_action": defaults["setup_action"],
        "docker_image": defaults["docker_image"],
        "artifacts_path": defaults["artifacts_path"],
        "install_command": config.install_command,
        "test_command": config.test_command,
        "lint_command": config.lint_command,
        "branch_triggers": ", ".join(config.branch_triggers),
        "deploy_target": config.deploy_target,
    }

    # Also include any extra_vars from the config
    variables.update({k: str(v) for k, v in config.extra_vars.items()})

    # Replace {{key}} placeholders
    def _replace(match: re.Match) -> str:
        key = match.group(1)
        return variables.get(key, match.group(0))

    rendered = re.sub(r"\{\{(\w+)\}\}", _replace, template)
    return rendered


def generate_cicd_config(
    platform: CICDPlatform,
    tech_stack: str = "",
    test_command: str = "",
    project_path: str | Path = "",
) -> str:
    """High-level: generate a rendered CI/CD config string.

    Auto-detects language from project_path if tech_stack not provided.
    """
    language = "python"
    if tech_stack:
        tech_lower = tech_stack.lower()
        if "node" in tech_lower or "javascript" in tech_lower or "typescript" in tech_lower:
            language = "node"
        elif "go" in tech_lower or "golang" in tech_lower:
            language = "go"
        elif "rust" in tech_lower:
            language = "rust"
    elif project_path:
        language = detect_language(project_path)

    config = build_config(
        platform=platform,
        language=language,
        test_command=test_command,
    )
    return render_template(config)


def write_cicd_config(
    project_path: str | Path,
    platform: CICDPlatform,
    config_content: str,
) -> Path:
    """Write a rendered CI/CD config to the correct location in the project.

    Creates parent directories as needed. Returns the path written to.
    """
    project = Path(project_path)
    from eunomia_v2.cicd.models import PLATFORM_FILE_PATHS

    relative_path = PLATFORM_FILE_PATHS[platform]
    output_path = project / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(config_content, encoding="utf-8")

    logger.info(f"Wrote CI/CD config to {output_path}")
    return output_path
