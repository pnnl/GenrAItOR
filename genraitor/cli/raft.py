"""."""

from pathlib import Path

import click
import os
import pickle

from ..conf import env, log

# default instructions for the biology domain
CUSTOM_INSTRUCTIONS = "You are a synthetic question-answer pair generator for the biology domain. Given a chunk of context from biological literature and databases, generate %s example questions a user could ask and would be answered using information from the chunk. For example, if the given context was PubMed abstracts and database entries with information about proteins A, B, and C, example questions could be 'What biological functions do A, B, and C perform?' or 'What, if any, is the nature of the interaction between A, B, and C?'. The questions should be able to be answered in a few sentences or less."

@click.group()
def cli():
    """Retrieval of context for creating a dataset for Retrieval-Augmented Fine-Tuning"""
    pass


@cli.command("raft:merge")
@click.option(
    "-a",
    "--adapter_path",
    "adapter_path",
    required=True,
    type=click.Path(file_okay=False, path_type=Path, exists=True),
)
@click.option(
    "-b",
    "--base_model",
    "--base",
    "base_model",
    default=env.model.name,
)
@click.option("-o", "--output_path", "--save_path", "save_path", default="genraitor")
def merge(adapter_path, base_model, save_path):
    """Merge the trained model with the base model."""
    from ..raft import train

    model = train.load(
        adapter_path=adapter_path,
        base_model=base_model,
    )
    log.info("saving merged")
    model.save_pretrained(save_path)
    log.info(f"saved: {save_path}")

@cli.command("raft:context")
def context()


@cli.command("raft:data")
# translate the argparses to click options:
@click.option(
    "--embed",
    type=click.Choice(["cloud", "local"]),
    default="cloud",
    help="One of either 'cloud' or 'local'.  If 'cloud' then uses the llama-index interface to the OpenAI.  If 'local', then using the HuggingFaceLLM llama-index wrapper for huggingface models.",
)
@click.option(
    "--hf_embed_model",
    type=str,
    default="Alibaba-NLP/gte-large-en-v1.5",
    help="A huggingface embedding model",
)
@click.option(
    "--embed_model_cache",
    type=str,
    default="/qfs/projects/genraitor/models/cache",
    help="Cache directory for huggingface models from llama-index",
)
@click.option(
    "--chat_model_name",
    type=str,
    default="gpt-4o",
    help="The name of the language model to use for completions.  Defaults to 'gpt-4o' and probably won't work with other models.",
)
@click.option(
    "--context_path",
    type=str,
    default="context.txt",
    help="The path to a text file containing the context to be chunked and used to generate QA pairs.",
)
@click.option(
    "--output_path",
    type=str,
    help="The path to save the huggingface dataset produced by the raft output",
)
@click.option(
    "--save_chunks_path",
    type=str,
    help="A path to a checkpoint to load chunks from if it exists, or save to if not.",
)
@click.option(
    "--oai_key",
    type=str,
    default=".secret",
    help="The path to a file with a single line containing the openAI api key.",
)
@click.option(
    "--hf_token",
    type=str,
    default=".hf_token",
    help="The path to a file with a single line containing a huggingface access token.  Alternatively set HF_TOKEN environment variable.",
)
@click.option(
    "--api_base",
    type=str,
    default="https://ai-incubator-api.pnnl.gov",
    help="The base URL for the OpenAI API.",
)
def make_raft_dataset(
    embed,
    hf_embed_model,
    embed_model_cache,
    chat_model_name,
    context_path,
    output_path,
    save_chunks_path,
    oai_key,
    hf_token,
    api_base
):
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    import datetime

    from genraitor.data.raft_dataset import RAFTDatasetPack

    if not os.environ.get("OPENAI_API_KEY"):
        with open(oai_key) as f:
            API_KEY=f.readline().strip("\n")
    else:
        API_KEY = os.environ.get("OPENAI_API_KEY")

    # Used to generate questions and answers about the text content
    llm = OpenAI(
        model=chat_model_name,
        api_key=API_KEY,
        api_base=api_base
    )

    # used to semantically segment the document into chunks that will be used as 'documents' to reason over.
    if embed == "cloud":
        embed_model = OpenAIEmbedding(
            api_key=API_KEY, 
            api_base=api_base
        )
    elif embed == "local":
        if hf_token:
            with open(hf_token) as f:
                os.environ["HF_TOKEN"] = f.readline().strip("\n")

        # Defaulting to some embedding model from HF for this.
        embed_model = HuggingFaceEmbedding(
            model_name = hf_embed_model, 
            cache_folder = embed_model_cache,
            trust_remote_code = True,
            # model_kwargs = {"device_map":"auto"}
        )

    raft_dataset = RAFTDatasetPack(
        instruction_template = CUSTOM_INSTRUCTIONS,
        file_path = context_path, 
        llm = llm, 
        embed_model=embed_model
    )

    if not output_path:
        thetime = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
        output_path = f'raft-dataset-{os.path.basename(os.path.splitext(os.path.basename(context_path))[0])}-{thetime}.hf'
    else:
        output_path = output_path

    log.info(f"Beginning raft dataset construction, writing to: {output_path}")

    chunks = None

    if save_chunks_path:
        if os.path.exists(save_chunks_path):
            chunks = pickle.load(open(save_chunks_path, 'rb'))

    # a raft dataset in huggingface format
    dataset = raft_dataset.run(
        checkpoint_path = output_path,
        chunks = chunks,
        save_chunks_path=save_chunks_path    
    )

    dataset.save_to_disk(output_path)
