# Simple example of using the RAFTDatasetPack class to generate a dataset from a text file containing abstracts. The dataset is then used to generate a dataset in huggingface format.
from llama_index.llms.openai import OpenAI
from llama_index.packs.raft_dataset import RAFTDatasetPack
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.packs.raft_dataset import RAFTDatasetPack
import os
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

# with open("../.secret") as f:
#     API_KEY=f.readline()

os.environ["OPENAI_API_KEY"] = API_KEY
os.environ[
    "AZURE_OPENAI_ENDPOINT"
] = "https://ai-incubator-api.pnnl.gov"
os.environ["OPENAI_API_VERSION"] = "2024-05-13"

llm = AzureOpenAI(
    model='gpt-4o',
    deployment_name="gpt-4o"
)

# You need to deploy your own embedding model as well as your own chat completion model
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="text-embedding-ada-002",
)


# Used to generate questions and answers about the text content
# llm = OpenAI(
#     model='gpt-4o',
#     api_key=API_KEY,
#     api_base="https://ai-incubator-api.pnnl.gov"
# )

# # used to semantically segment the document into chunks that will be used as 'documents' to reason over.
# embed_model = OpenAIEmbedding(
#     api_key=API_KEY, 
#     api_base="https://ai-incubator-api.pnnl.gov"
# )

fpath = 'example_abstract.txt'
fpath = '/people/clab683/git_repos/llama-3-raft/larger_context_P05067_Q9NQA5.txt'

raft_dataset = RAFTDatasetPack(fpath, llm = llm, embed_model=embed_model)

# a raft dataset in huggingface format
dataset = raft_dataset.run()

#### Local LLM 

from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
import torch

model_path = "/rcfs/projects/ops/Meta-Llama-3.1-405B-Instruct"
config_path = "/people/clab683/git_repos/llama-3-raft/hf_configs/Meta-Llama-3-70B-Instruct.json"
# quantization_config = BitsAndBytesConfig(
#       load_in_4bit=True,
#       bnb_4bit_quant_type="nf4",
#       bnb_4bit_use_double_quant=True,
#       bnb_4bit_compute_dtype=torch.bfloat16,
#   )

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
cfg = AutoConfig.from_pretrained(config_path)

model = AutoModelForCausalLM.from_config(
    cfg
)

model = AutoModelForCausalLM.from_pretrained(
    config_path, device_map="auto", 
    torch_dtype=torch.bfloat16, 
    quantization_config=quantization_config)

model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", 
    torch_dtype=torch.bfloat16, 
    quantization_config=quantization_config)

tokenizer = AutoTokenizer.from_pretrained(model_path)
