from typing import Optional

from enum import Enum

import structlog
from pydantic import BaseModel, Extra, Field
from pydantic_settings import BaseSettings


class VendorConfig(BaseModel):
    hf_token: Optional[str] = Field(None, alias="HF_TOKEN")
    ncbi_api_key: Optional[str] = Field(None, alias="NCBI_API_KEY")
    path: Optional[str] = Field(None, alias="PATH")
    ld_library_path: Optional[str] = Field(None, alias="LD_LIBRARY_PATH")
    pytorch_cuda_alloc_conf: Optional[str] = Field(
        None, alias="PYTORCH_CUDA_ALLOC_CONF"
    )

    class Config:
        extra = Extra.allow


class PathConfig(BaseModel):
    app: str = "/path/to/repo"
    data: str = "/path/to/data"
    rag_data: str = "/path/to/ragdata"
    rag_cache: str = "/path/to/ragdata.db"

class EmbedModelNames(str, Enum):
    LLAMA3_1 = "llama3_1"
    BIONLP_BERT = "bionlp_pubmed"
    CAMBRIDGE_PUBMED = "cambridge_pubmed"
    MICROSOFT_PUBMED = "microsoft_pubmed"
    OPENAI = "openai"

class ModelConfig(BaseModel):
    name: str = "meta-llama/Meta-Llama-3-8B"
    output_name: str = "genraitor"
    embed_name: str = "ollama"
    api_key: str = ""
    quantization_type: str = "fp4"  # or "nf4"
    device_map: str = "auto"  # or "cuda"
    embed_model_names: Enum = EmbedModelNames
    
class TrainingConfig(BaseModel):
    attention: str = "flash_attention_2"  # or "eager"
    batch_size: int = 20
    max_seq_len: int = 8000


class EvalConfig(BaseModel):
    max_tokens: int = 4000


class Settings(BaseSettings, VendorConfig):
    project_name: str = "genraitor"
    paths: PathConfig = PathConfig()
    training: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig()
    eval: EvalConfig = EvalConfig()

    class Config:
        env_prefix = "GENRAITOR__"
        env_file = ".env"
        env_nested_delimiter = "__"
        env_file_encoding = "utf-8"


log = structlog.get_logger()
env = Settings()
