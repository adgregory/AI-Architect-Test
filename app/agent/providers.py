"""Strands model providers, selected by configuration.

Locally the agent uses Gemini on Vertex AI when Application Default Credentials are mounted
(docker-compose.gcp.yml), otherwise no LLM (extractive answers); in AWS it runs on AgentCore
Runtime with a Bedrock model authorised by the runtime's IAM role. Only the provider
changes — prompts, retrieval and streaming are identical.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from app.core.config import Settings
from app.core.factories import _require
from app.core.logging import get_logger

log = get_logger(__name__)


class ModelProviderFactory:
    @staticmethod
    def _gemini(s: Settings) -> Any:
        _require("strands", "agent", "gemini")
        from google import genai
        from strands.models.gemini import GeminiModel

        # Pre-built Vertex client: credentials come from ADC (google.auth.default), never a key.
        client = genai.Client(vertexai=True, project=s.gemini_project, location=s.gemini_location)
        return GeminiModel(
            client=client,
            model_id=s.gemini_model,
            params={"temperature": s.llm_temperature, "max_output_tokens": s.llm_max_output_tokens},
        )

    @staticmethod
    def _bedrock(s: Settings) -> Any:
        _require("strands", "agent", "bedrock")
        from strands.models import BedrockModel

        # Credentials from the AgentCore runtime execution role (IAM), scoped to this model ARN.
        return BedrockModel(
            model_id=s.bedrock_model_id,
            region_name=s.aws_region,
            temperature=s.llm_temperature,
            max_tokens=s.llm_max_output_tokens,
        )

    builders: ClassVar[dict[str, Callable[[Settings], Any]]] = {"gemini": _gemini, "bedrock": _bedrock}

    @staticmethod
    def _gcp_credentials_available() -> bool:
        try:
            import google.auth

            google.auth.default()
            return True
        except Exception:  # noqa: BLE001 - not installed, no ADC, unreadable file: all mean "no"
            return False

    @classmethod
    def resolve(cls, settings: Settings) -> str:
        """`auto`: Gemini when a GCP project and credentials are present, otherwise no LLM
        (extractive answers), so the stack runs on a machine without cloud credentials."""
        if settings.llm_provider != "auto":
            return settings.llm_provider
        if settings.gemini_project and cls._gcp_credentials_available():
            return "gemini"
        log.warning(
            "agent.no_llm",
            reason="no GOOGLE_CLOUD_PROJECT" if not settings.gemini_project else "no GCP credentials",
            fallback="extractive answers",
        )
        return "none"

    @classmethod
    def create(cls, settings: Settings, provider: str | None = None) -> Any:
        """The Strands model for the provider, or None when answering without an LLM."""
        provider = provider or cls.resolve(settings)
        return None if provider == "none" else cls.builders[provider](settings)
