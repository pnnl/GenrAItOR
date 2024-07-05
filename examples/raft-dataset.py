# Simple example of using the RAFTDatasetPack class to generate a dataset from a text file containing abstracts. The dataset is then used to generate a dataset in huggingface format.
from llama_index.llms.openai import OpenAI
from llama_index.packs.raft_dataset import RAFTDatasetPack
from llama_index.embeddings.openai import OpenAIEmbedding

with open(".secret") as f:
    API_KEY=f.readline()

# Used to generate questions and answers about the text content
llm = OpenAI(
    model='gpt-4o',
    api_key=API_KEY,
    api_base="https://ai-incubator-api.pnnl.gov"
)

# used to semantically segment the document into chunks that will be used as 'documents' to reason over.
embed_model = OpenAIEmbedding(
    api_key=API_KEY, 
    api_base="https://ai-incubator-api.pnnl.gov"
)

fpath = 'example_abstract.txt'

raft_dataset = RAFTDatasetPack(fpath, llm = llm, embed_model=embed_model)

# a raft dataset in huggingface format
dataset = raft_dataset.run()
