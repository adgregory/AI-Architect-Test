"""Strands model providers, selected by configuration.

Locally the agent uses Gemini on Vertex AI with Application Default Credentials; in AWS it
runs on AgentCore Runtime with a Bedrock model authorised by the runtime's IAM role. Only
the provider changes — prompts, retrieval and streaming are identical.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from app.core.config import Settings
from app.core.factories import _require


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

    @classmethod
    def create(cls, settings: Settings) -> Any:
        return cls.builders[settings.llm_provider](settings)
