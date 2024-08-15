"""
Example of using the RAFTDatasetPack class to generate a dataset from a text file containing context (e.g. from pubmed abstracts and uniprot entries). The dataset is then used to generate a dataset in huggingface format.

By default it will use gpt-4o and the ada-002 embeddings via the llama-index interface to OpenAI, using our endpoint.  To get around rate limit issues using the local version, specify to use a local embedding model like so:

```
python build-raft.py --embed local
```

"""
from llama_index.llms.openai import OpenAI
from llama_index.packs.raft_dataset import RAFTDatasetPack
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.packs.raft_dataset import RAFTDatasetPack
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import ChatMessage

import warnings
import os
import argparse
import logging
import sys
import datetime

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

DEFAULT_INSTRUCTGEN_TEMPLATE = "You are a synthetic question-answer pair generator. Given a chunk of context about some topic(s), generate %s example questions a user could ask and would be answered using information from the chunk. For example, if the given context was a Wikipedia paragraph about the United States, an example question could be 'How many states are in the United States?'. The questions should be able to be answered in a few words or less."

CUSTOM_INSTRUCTIONS = "You are a synthetic question-answer pair generator for the biology domain. Given a chunk of context from biological literature and databases, generate %s example questions a user could ask and would be answered using information from the chunk. For example, if the given context was PubMed abstracts and database entries with information about proteins A, B, and C, example questions could be 'What biological functions do A, B, and C perform?' or 'What, if any, is the nature of the interaction between A, B, and C?'. The questions should be able to be answered in a few sentences or less."

class BioRAFTDatasetPack(RAFTDatasetPack):
    # init with a new keyword argument
    def __init__(self, instruction_template=DEFAULT_INSTRUCTGEN_TEMPLATE, **kwargs):
        super().__init__(**kwargs)
        self.instruction_template = instruction_template

    def generate_instructions_gen(self, chunk, x = 5):
        messages = [
            ChatMessage(
                role="system",
                content=self.instruction_template % x
            ),
            ChatMessage(role="user", content=str(chunk)),
        ]

        queries = str(self.llm.chat(messages)).split("\n")
        questions = [self.strip_str(q) for q in queries]
        questions = [q for q in questions if any(c.isalpha() for c in q)][:x]

        num_questions_generated = len(questions)
        if num_questions_generated < x:
            warnings.warn(
                f"Fewer questions generated ({num_questions_generated}) "
                f"than requested ({x})."
            )

        return questions

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
        embed_model = HuggingFaceEmbedding(model_name = 'Alibaba-NLP/gte-large-en-v1.5', trust_remote_code = True)

    raft_dataset = BioRAFTDatasetPack(
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

    # a raft dataset in huggingface format
    dataset = raft_dataset.run()

    dataset.save_to_disk(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed", choices = ['cloud', 'local'], default = 'cloud', type = str, help = "One of either 'cloud' or 'local'.  If 'cloud' then uses the llama-index interface to the OpenAI.  If 'local', then using the HuggingFaceLLM llama-index wrapper for huggingface models.")
    parser.add_argument("--model_name", default = "gpt-4o", type = str, help = "The name of the language model to use for completions.  Defaults to 'gpt-4o' and probably won't work with other models.")
    parser.add_argument("--context_path", default = "context.txt", type = str, help = "The path to a text file containing the context to be chunked and used to generate QA pairs.")
    parser.add_argument("--output_path", type = str, help = "The path to save the huggingface dataset produced by the raft output")
    parser.add_argument("--oai_key", default = ".secret", type = str, help = "The path to a file with a single line containing the openAI api key.")
    parser.add_argument("--hf_token", default = ".hf_token", type = str, help = "The path to a file with a single line containing a huggingface access token.  Alternatively set HF_TOKEN environment variable.")

    args = parser.parse_args()

    main(args)
