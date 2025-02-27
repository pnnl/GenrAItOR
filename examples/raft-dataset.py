"""
Example of using the RAFTDatasetPack class to generate a dataset from a text file containing context (e.g. from pubmed abstracts and uniprot entries). The dataset is then used to generate a dataset in huggingface format.

By default it will use gpt-4o using the llanaand the Alibaba-NLP/gte-large-en-v1.5 embeddings via the llama-index and huygg interface to OpenAI, using our endpoint.  To get around rate limit issues using the local version, specify to use a local embedding model like so:

```
python raft-dataset.py --embed local
```

"""
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from genraitor.data.raft_dataset import RAFTDatasetPack

import os
import argparse
import logging
import sys
import datetime
import pickle

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

CUSTOM_INSTRUCTIONS = "You are a synthetic question-answer pair generator for the biology domain. Given a chunk of context from biological literature and databases, generate %s example questions a user could ask and would be answered using information from the chunk. For example, if the given context was PubMed abstracts and database entries with information about proteins A, B, and C, example questions could be 'What biological functions do A, B, and C perform?' or 'What, if any, is the nature of the interaction between A, B, and C?'. The questions should be able to be answered in a few sentences or less."

def main(args):
    if not os.environ.get("OPENAI_API_KEY"):
        with open(args.oai_key) as f:
            API_KEY=f.readline().strip("\n")
    else:
        API_KEY = os.environ.get("OPENAI_API_KEY")

    # Used to generate questions and answers about the text content
    llm = OpenAI(
        model='gpt-4o',
        api_key=API_KEY,
        api_base="https://ai-incubator-api.pnnl.gov"
    )

    # used to semantically segment the document into chunks that will be used as 'documents' to reason over.
    if args.embed == "cloud":
        embed_model = OpenAIEmbedding(
            api_key=API_KEY, 
            api_base="https://ai-incubator-api.pnnl.gov"
        )
    elif args.embed == "local":
        if args.hf_token:
            with open(args.hf_token) as f:
                os.environ["HF_TOKEN"] = f.readline().strip("\n")

        # Defaulting to some embedding model from HF for this.
        embed_model = HuggingFaceEmbedding(
            model_name = args.hf_embed_model, 
            cache_folder = args.model_cache,
            trust_remote_code = True,
            # model_kwargs = {"device_map":"auto"}
        )

    raft_dataset = RAFTDatasetPack(
        instruction_template = CUSTOM_INSTRUCTIONS,
        file_path = args.context_path, 
        llm = llm, 
        embed_model=embed_model
    )

    if not args.output_path:
        thetime = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
        output_path = f'raft-dataset-{os.path.basename(os.path.splitext(os.path.basename(args.context_path))[0])}-{thetime}.hf'
    else:
        output_path = args.output_path

    logging.info(f"Beginning raft dataset construction, writing to: {output_path}")

    chunks = None

    if args.save_chunks_path:
        if os.path.exists(args.save_chunks_path):
            chunks = pickle.load(open(args.save_chunks_path, 'rb'))

    # a raft dataset in huggingface format
    dataset = raft_dataset.run(
        checkpoint_path = output_path,
        chunks = chunks,
        save_chunks_path=args.save_chunks_path    
    )

    dataset.save_to_disk(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed", choices = ['cloud', 'local'], default = 'cloud', type = str, help = "One of either 'cloud' or 'local'.  If 'cloud' then uses the llama-index interface to the OpenAI.  If 'local', then using the HuggingFaceLLM llama-index wrapper for huggingface models.")
    parser.add_argument("--hf_embed_model", type = str, default='Alibaba-NLP/gte-large-en-v1.5', help = "A huggingface embedding model")
    parser.add_argument("--model_cache", type = str, default= "/qfs/projects/genraitor/models/cache", help = "Cache directory for huggingface models from llama-index")
    parser.add_argument("--model_name", default = "gpt-4o", type = str, help = "The name of the language model to use for completions.  Defaults to 'gpt-4o' and probably won't work with other models.")
    parser.add_argument("--context_path", default = "context.txt", type = str, help = "The path to a text file containing the context to be chunked and used to generate QA pairs.")
    parser.add_argument("--output_path", type = str, help = "The path to save the huggingface dataset produced by the raft output")
    parser.add_argument("--save_chunks_path", type = str, help = "A path to a checkpoint to load chunks from if it exists, or save to if not.")
    parser.add_argument("--oai_key", default = ".secret", type = str, help = "The path to a file with a single line containing the openAI api key.")
    parser.add_argument("--hf_token", default = ".hf_token", type = str, help = "The path to a file with a single line containing a huggingface access token.  Alternatively set HF_TOKEN environment variable.")

    args = parser.parse_args()

    main(args)
