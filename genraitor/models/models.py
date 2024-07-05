"""LLM Model Builder."""

import os
from enum import Enum

from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
import os


class LLMModel(str, Enum):
    """Response modes of the response builder (and synthesizer)."""

    LLAMA3_1 = "llama3_1"
    PHI3 = "phi_3"
    OPENAI = "openai"

    @staticmethod
    def build(model):
        """Get LLM model from alias."""
        from ..conf import env
        match model:
            case "ollama" | LLMModel.LLAMA3_1:
                llm = Ollama(model="llama3.1", request_timeout=120)
            case "microsoft/Phi-3-mini-4k-instruct" | LLMModel.PHI3:
                llm = Ollama(
                    model="microsoft/Phi-3-mini-4k-instruct", request_timeout=120,
                )
            case "openai" | LLMModel.OPENAI:
                from ..conf import env
                os.environ["OPENAI_API_KEY"] = env.model.api_key
                os.environ["OPENAI_API_BASE"] = env.model.api_url
                llm = OpenAI(model="gpt-4o")
            case _:
                raise ValueError(f"model not recognized: {model}")
        return llm
