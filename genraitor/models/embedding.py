from enum import Enum

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding


class EmbedModel(str, Enum):
    """Response modes of the response builder (and synthesizer)."""

    LLAMA3_1 = "llama3_1"
    BIONLP_BERT = "bionlp_pubmed"
    CAMBRIDGE_PUBMED = "cambridge_pubmed"
    MICROSOFT_PUBMED = "microsoft_pubmed"
    OPENAI = "openai"

    @staticmethod
    def build(model):
        match model:
            case "ollama" | EmbedModel.LLAMA3_1:
                embed_model = OllamaEmbedding(
                    model_name="llama3.1",
                    base_url="http://localhost:11434",
                    ollama_additional_kwargs={"mirostat": 0},
                )
            case EmbedModel.BIONLP_BERT:
                embed_model = HuggingFaceEmbedding(
                    model_name="bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
                )
            case EmbedModel.CAMBRIDGE_PUBMED:
                embed_model = HuggingFaceEmbedding(
                    model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
                )
            case EmbedModel.MICROSOFT_PUBMED:
                embed_model = HuggingFaceEmbedding(
                    model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
                )
            case EmbedModel.OPENAI:
                embed_model = OpenAIEmbedding()
            case _:
                raise ValueError(f"model not recognized: {model}")
        return embed_model
