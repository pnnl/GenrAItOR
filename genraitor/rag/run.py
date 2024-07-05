"""
Largely taken from https://docs.llamaindex.ai/en/stable/getting_started/starter_example/.
"""

from pathlib import Path
from typing import Optional
from ..conf import env, log
from .embedding import get_document_index
from ..models import EmbedModel

from llama_index.core.llms.custom import CustomLLM
from llama_index.llms.ollama import Ollama
from llama_index.core.response_synthesizers.type import ResponseMode

from llama_index.core.base.embeddings.base import BaseEmbedding


from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    ServiceContext,
    get_response_synthesizer,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.core.base.base_query_engine import BaseQueryEngine
from ..vector_store import DuckDBVectorStore


def print_query_and_response(engine: BaseQueryEngine, query: str) -> None:
    log.info("\n")
    log.info("-" * 80)
    log.info(f"Query:{query}")
    log.info("-" * 80)

    log.info("Context:")
    nodes = engine.retrieve(query)
    for node in nodes:
        log.info(
            f"{node.metadata['file_name']}: \"{node.text[:80].replace('\n', '')}...\""
        )

    log.info("-" * 80)
    log.info("Response:")
    response = str(engine.query(query))
    log.info(f"{response}")
    log.info("-" * 80)


def main(
    data_path: Path = Path(env.paths.rag_data),
    cache_path: Path = Path(env.paths.rag_cache),
    response_mode=ResponseMode.COMPACT,
    top_k=5,
    reindex=False,
    embedding_model: EmbedModel = EmbedModel.LLAMA3_1,
):
    llm = Ollama(model="llama3.1")
    embed_model = EmbedModel.build(embedding_model)
    index = get_document_index(
        data_path, cache_path, llm_model=llm, embed_model=embed_model, reindex=reindex
    )

    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode=response_mode,
    )

    # assemble query engine
    engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    # engine = index.as_query_engine(llm=llm, response_mode=response_mode)

    # Query for information not learned during training
    print_query_and_response(
        engine, "Make a comprehensive summary of the effects of floogling."
    )

    # Query for something unspecified
    print_query_and_response(
        engine, "What pubmed ids are related to tsp4_human protein?"
    )

    print_query_and_response(engine, "How is tsp4_human related to pch2_human??")
