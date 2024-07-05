from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from pathlib import Path
from typing import Optional
from ..conf import env, log

from llama_index.core.llms.custom import CustomLLM

from llama_index.core.base.embeddings.base import BaseEmbedding


from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    ServiceContext,
)
from ..vector_store import DuckDBVectorStore


def get_document_index(
    data_path: Optional[Path],
    cache_path: Path,
    llm_model: CustomLLM,
    embed_model: BaseEmbedding,
    reindex=False,
):
    if reindex:
        cache_path.unlink(missing_ok=True)
        log.warning("force reindex")
    try:
        vector_store = DuckDBVectorStore.from_local(database_path=str(cache_path))
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=embed_model
        )
        log.info(f"vector store cache found: {cache_path}")
    except ValueError as _:
        assert (
            data_path and data_path.exists()
        ), f"documents path does not exist: {data_path}"
        log.info(f"creating vector store for document search: {cache_path}")
        vector_store = DuckDBVectorStore(
            cache_path.name, persist_dir=str(cache_path.parent)
        )
        service_context = ServiceContext.from_defaults(
            llm=llm_model, embed_model=embed_model
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        documents = SimpleDirectoryReader(str(data_path)).load_data()
        index = VectorStoreIndex.from_documents(
            documents, service_context=service_context, storage_context=storage_context
        )
        # index.storage_context.persist(persist_dir=str(cache_path))
    return index


def get_index(llm, embed_model, cache_path: Path = Path(env.paths.rag_cache)):
    assert cache_path.exists(), f"no embeddings database found: {cache_path}"
    index = get_document_index(
        data_path=None, cache_path=cache_path, llm_model=llm, embed_model=embed_model
    )
    return index
