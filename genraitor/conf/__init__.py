from typing import Optional

import structlog
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class VendorConfig(BaseModel):
    hf_token: str = Field(alias="HF_TOKEN")
    ncbi_api_key: str = Field(alias="NCBI_API_KEY")

class PathConfig(BaseModel):
    app: str = "/path/to/repo"
    data: str = "/path/to/data"
    rag_data: str = "/path/to/ragdata"
    rag_cache: str = "/path/to/ragdata.db"


class ModelConfig(BaseModel):
    name: str = "microsoft/Phi-3-mini-4k-instruct"
    output_name: str = "genraitor"
    embed_name: str = "ollama"
    api_key: str = ""


class TrainingConfig(BaseModel):
    attention: str = "flash_attention_2"  # or "eager"


class Settings(BaseSettings, VendorConfig):
    project_name: str = "genraitor"
    paths: PathConfig = PathConfig()
    training: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig()

    class Config:
        env_prefix = "GENRAITOR__"
        env_file = ".env"
        env_nested_delimiter = "__"
        env_file_encoding = "utf-8"


log = structlog.get_logger()
env = Settings()
