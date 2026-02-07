"""CI/CD models — platform enum and configuration."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CICDPlatform(str, Enum):
    """Supported CI/CD platforms."""

    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    JENKINS = "jenkins"


# Map platform to the config file location within a project
PLATFORM_FILE_PATHS: dict[CICDPlatform, str] = {
    CICDPlatform.GITHUB_ACTIONS: ".github/workflows/ci.yml",
    CICDPlatform.GITLAB_CI: ".gitlab-ci.yml",
    CICDPlatform.JENKINS: "Jenkinsfile",
}


class CICDStage(BaseModel):
    """A single stage in a CI/CD pipeline."""

    name: str
    commands: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class CICDConfig(BaseModel):
    """Configuration for generating a CI/CD pipeline."""

    platform: CICDPlatform
    language: str = "python"
    language_version: str = "3.12"
    test_command: str = "pytest"
    lint_command: str = ""
    build_command: str = ""
    install_command: str = "pip install -r requirements.txt"
    branch_triggers: list[str] = Field(default_factory=lambda: ["main", "develop"])
    deploy_enabled: bool = False
    deploy_target: str = ""
    stages: list[CICDStage] = Field(default_factory=list)
    extra_vars: dict[str, Any] = Field(default_factory=dict)

    @property
    def file_path(self) -> str:
        """Return the output file path for this platform."""
        return PLATFORM_FILE_PATHS[self.platform]
