# Generating Understanding and Interpretation of Multi-Omics Data with an Automated and Generalizable Pipeline

# Objective

We aim to develop a prototype pipeline that rapidly predicts mechanisms driving disease pathogenesis using multi-omics data, demonstrating feasibility with generative AI for uncovering biological mechanisms based on key features from data harmonization.

# Team

## PNNL

- **Samantha Erwin** - AI Research
- **Lisa Bramer** - Domain Expert
- **Daniel Claborne** - AI Engineer
- **Javier Flores** - AI Research
- **Matt Jensen** - AI Engineer

## Domain & Relevant Sectors - Predictive Phenomics and 'Omics Discovery

- **Karl Mueller** - Basic Energy Sciences: Chem, Geo, and Biosciences
- **David Wunschel** - National Security: Chem/Bio

## Reviewer

- Lauren Charles

# Background

With advances in instrumentation, we now collect vast multi-omics datasets that provide insights into biological systems and disease mechanisms.
Current methods are often manual and time-consuming.
Our team has developed a scalable deep learning model for multi-omics data harmonization.
We aim to automate interpretation of predictive features using generative AI, thereby speeding up the discovery of biological mechanisms from existing datasets.

# Approach

We will use our harmonization model to identify key features from multi-omics datasets. Subsequently, we will apply Llama 3, fine-tuned with Retrieval Augmented Fine-Tuning (RAFT) on 'omics-related literature, to interpret these features and elucidate mechanisms driving disease pathogenesis.


# Install

Use your favorite python manager to initial a virtual environment, `venv` for example:

```bash

# install a virtual env in .venv
python3 -m venv .venv

# scope python packages to this path
source .venv/bin/activate

```

Then install the dependencies and the repo as a python package:

```bash

# install project as standalone python package
pip install -e .

```

# Running The Code

The code is presented as a python package as well as a CLI using the `click` package.  The CLI is invoked by running:

```bash
python -m genraitor <cli-entrypoint>
```

Help documentation for each entrypoint can be accessed by running:

```
python -m genraitor <cli-entrypoint> --help
```

Some of the main steps in carrying out our fine-tuning procedure and their associated endpoints are described below.

## Generating Data

Our synthetic data processing pipeline starts with a set of uniprot identifiers you are interested in.  You can collect these beforehand using variable selections methods such as LASSO, or Shapley values.

### Option 1:  From a file of UniProt ID's (Recommended)

Start with a file containing uniprot [Accession numbers](https://www.uniprot.org/help/accession_numbers), one per line, as below:

```
# data/examples/uniprots.txt
Q9BRJ2
P09758
P84085
P08708
P46013
P02768
P05026
P14618
```

Then provide this file to the `raft:context` cli endpoint.  This file will also default to some example uniprot ids when no file is provided.

```bash
python -m genraitor raft:context \
--uniprot_ids=./data/examples/uniprots.txt \
--output_dir=./data
```

This will produce two files in `./data`, one (`uniprot_context_results...`) with the raw results of querying uniprot for pathway information and abstracts, and the other (`uniprot_context_postprocessed...`) with context derived from those results and usable by the `RAFTDatasetPack` class from `llama-index`.

### Option 2:  From a file with scores:

To generate the top uniprot ids (the directory `data/deepimv` should exist and contain a .csv file starting with 'shap', and containing 'AH1' and 'pro'):

```bash

# to save as a parquet file:
python3 -m genraitor data:uniprot --save_path data/training/uniprot.parquet

# to just print the values to stdout:
python3 -m genraitor data:uniprot
```

To parse the uniprot data for their associated pubmed ids:

```bash

# to save as a parquet file:
python3 -m genraitor data:uniprot-to-pubmed --uniprot_path data/training/uniprot.parquet --save_path data/training/uniprot_pubmed_ids.parquet

# to just print the values to stdout:
python3 -m genraitor data:uniprot-to-pubmed --uniprot_path data/training/uniprot.parquet
```

To generate documents for usage in a RAG model:

```bash

# to save as json files:
python3 -m genraitor data:rag --uniprot_path data/training/uniprot.parquet --save_path data/training/rag/documents

```

## RAFT Dataset
Once you have used the above to create a text file of context, you can use our modified `RAFTDatasetPack` class to create synthetic question-answer pairs about chunks of that context.

You will need an OpenAI API key as well as a huggingface API key.  The entrypoint for the cli is `raft:data`, or there is an example script at `examples/raft-dataset.py`.

To run from the cli do:

```
# set keys
export HF_TOKEN=<your-hf-token>
export OPENAI_API_KEY=<your-oai-key>

python -m genraitor raft:data \
--embed local \
--context_path /path/to/context.txt \
--output_path /path/to/save_data_folder
```

See `python -m raft:data --help` for more options.  The resulting huggingface dataset is a folder of files and can be loaded as below:

```python
from datasets import load_from_disk

dataset = load_from_disk('/path/to/save_data_folder')
```

## RAG Model Inference

To run the rag model:

```bash

python3 -m genraitor rag:run

```

To inspect the documents nearest to a prompt:

```bash

python3 -m genraitor rag:index "How is tsp4_human related to pch2_human?"

```

# Data Sources

- **Multi-Omics Data**: From PNNL's study on host responses to lethal human viruses.
    - [Publication](https://www.nature.com/articles/s41597-024-03124-3)
    - [PNNL DataHub for Multi-onmics Publication](https://data.pnnl.gov/group/nodes/publication/33863)
- **Literature**: ‘Omics-related publications from PubMed, Medline, and Wikipathways.

# Success Metrics

- **30-day Goal**: Fine-tune Llama 3 with RAFT using relevant literature.
- **60-day Goal**: Demonstrate the model's ability to identify known biological mechanisms.

# Contact

- **Samantha Erwin**: [samantha.erwin@pnnl.gov](mailto:samantha.erwin@pnnl.gov)

# References

1. Erwin, S et al. 2024; doi:10.1101/2023.09.06.556597
2. Lee, C & van der Schaar, M. 2021; doi:10.48550/2102.03014
3. Slenter DN et al. 2018; doi:10.1093/nar/gkx1064
4. Zhang, T et al. 2024; doi:10.48550/2403.10131

# Project High Level

## Model Pipeline

**Step 1 (existing work):**

- In: Omics from infected and control
- Model: DeepIMV
- Out: Infection detection, macromolecules used as features in order of importance to the prediction

**Step 2 (genraitor):**

- In: Macromolecules
- Model: Raft Llama3
- Out: Relevant metabolic pathways, citations to wikipathways or PubMed and chain of reasoning


# Relevant Links

## Project

- [Project Repo](https://tanuki.pnnl.gov/GENRAItOR/llama3-raft)
- [PNNL GenAI Repo](https://tanuki.pnnl.gov/GenAI/genai-infrastructure)
- [Teams](https://teams.microsoft.com/l/team/19%3A6iZ7HVCgPFC6VX97MF3kDGyc3vntKMHcXLSjkXeCKcg1%40thread.tacv2/conversations?groupId=e4eacec8-4a12-49c2-a4f3-04f6c9401a73&tenantId=d6faa5f9-0ae2-4033-8c01-30048a38deeb)

## RAFT

- [Original RAFT paper](https://arxiv.org/abs/2403.10131)
- [RAFT press release (Microsoft)](https://techcommunity.microsoft.com/t5/ai-ai-platform-blog/raft-a-new-way-to-teach-llms-to-be-better-at-rag/ba-p/4084674?WT_mc_id=aiml-122224-cedricvidal)
- [RAFT press release (Meta)](https://ai.meta.com/blog/raft-llama-retrieval-augmented-generation-supervised-fine-tuning-microsoft/)
- [RAFT press release (Berkeley)](https://gorilla.cs.berkeley.edu/blogs/9_raft.html)
- [Blog post w/ example MVP RAFT implementation](https://www.automateyournetwork.ca/uncategorized/local-raft-fine-tuning-llama3-with-domain-specific-knowledge-locally-and-privately/)
- [Repo with MVP RAFT training](https://github.com/automateyournetwork/fine_tune_example)
- [Webinar on RAFT (LlamaIndex)](https://www.youtube.com/watch?v=pira_p6aRVA)
- [dataset generation (LlamaIndex)](https://docs.llamaindex.ai/en/stable/api_reference/packs/raft_dataset/)
- [RAFT LlamaIndex Pack](https://llamahub.ai/l/llama-packs/llama-index-packs-raft-dataset):

## Llama3

- [Llama3 Repo](https://github.com/meta-llama/llama3)
- [Llama3 Huggingface page](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)
- [Getting to know Llama models](https://github.com/meta-llama/llama-recipes/blob/main/recipes/quickstart/Getting_to_know_Llama.ipynb)
- [Llama3 fine-tuning guide](https://llama.meta.com/docs/how-to-guides/fine-tuning/)

## LlamaIndex

- [LlamaIndex llamafile llm](https://pypi.org/project/llama-index-llms-llamafile/)
- [LlamaIndex llamafile embeddings](https://pypi.org/project/llama-index-embeddings-llamafile/)
- [LlamaIndex RAFT example](https://github.com/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-raft-dataset/examples/raft_dataset.ipynb)

## Fine-Tuning

- [HuggingFace: TRL - Transformer Reinforcement Learning](https://huggingface.co/docs/trl/en/index)
- [HuggingFace: Odds Ratio Preference Optimization (ORPO) Trainer](https://huggingface.co/docs/trl/en/orpo_trainer)
- [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)

## Omics

- [Data Harmonization Dataset](https://pnnl.sharepoint.com/:f:/r/teams/DataHarmonization/Shared%20Documents/Data%20Analysis/deep_learning/multiview_mlp/mlflow_runs/2631ad87eab14b2a9804841e47e842b2/artifacts?csf=1&web=1&e=bjob7o)
- [PNNL Publication on Disease Prediction](https://www.nature.com/articles/s41597-024-03124-3)
- [PNNL DataHub for Multi-onmics Publication](https://data.pnnl.gov/group/nodes/publication/33863)
- [Uniprot Protein Database REST API](https://www.uniprot.org/help/api_queries)
- [Wikipathways Python Package](https://pywikipathways.readthedocs.io/en/latest/)
- [RefMat](https://www.metabolomicsworkbench.org/databases/refmet/index.php)
- [LipidMaps](https://www.lipidmaps.org/)


# Data Generation

## Notes

Data generation task is going to be the more important, labor intensive and hardest to define in this process.
To train the model, we need a set questions-answer pairs, and a rotating corpus of documents the model can use to answer the question.
The documents used for training can be helpful (called 'oracle' documents) or unhelpful (called 'destractor' documents).
During training the model learns which information is relevant/irrelevant and memorizes domain knowledge.
There is a package called LlamaIndex (credit to Matt Gaddis) that automates this process, but we'll have to heavily modify it to fit with our use case. 

LlamaIndex can provide a blueprint for the data structure RAFT requires - which is helpful because there's some nuance (like providing 'distractor' documents, good system prompts, etc). 
The drawback: LlamaIndex generates question-answer pairs using an LLM - it takes a set of documents and uses ChatGPT or some other LLM to generate random question-answer pairs.
For our purposes, this dataset should focus on a specific set of questions. 
For example, we might want the LLM to respond to the question 'given {macromolecules xyz}, what metobolic pathway do they share in common?' with the answer 'the {macromolecules xyz} are related through {abc pathway} according to {doc_id} and {doc_id}`
LlamaIndex might generate question pairs like 'Who was the first author of the xyz paper?' instead.
So we'll need to manually create a finetuning dataset.

There are a couple of different fine tuning methods.
General techniques for tuning with smaller resource requirements than the original training regime (often called [Parameter Efficient Fine-Tuning (PEFT)](https://huggingface.co/docs/peft/main/en/index)) include [LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) and [QLoRA](https://arxiv.org/abs/2305.14314).
Another class of training techniques use Reinforcement Learning (often called [Reinforcement Learning Through Human Feedback (RLHF)](https://huggingface.co/blog/rlhf)), including the [TRL](https://huggingface.co/docs/trl/en/index) package from HuggingFace.

According to a recent blog post, the RLHF technique finetuning with [odds ratio preference optimization algorithm (ORPO)](https://arxiv.org/abs/2403.07691) works with as little as 50 unique prompts.
There is a [TRL HuggingFace package](https://huggingface.co/docs/trl/en/orpo_trainer) that implements this technique.

# Fine Tuning

```python
orpo_dataset_dict = {
    "prompt": [
        "hello",
        "how are you",
        "What is your name?",
        "What is your name?",
        "Which is the best programming language?",
        "Which is the best programming language?",
        "Which is the best programming language?",
    ],
    "chosen": [
        "hi nice to meet you",
        "I am fine",
        "My name is Mary",
        "My name is Mary",
        "Python",
        "Python",
        "Java",
    ],
    "rejected": [
        "leave me alone",
        "I am not fine",
        "Whats it to you?",
        "I dont have a name",
        "Javascript",
        "C++",
        "C++",
    ],
}
```

```
{
    "prompt": "How is protein A and protein B related {uniprot.A.function} {uniprot.A.interactions} {uniprot.A.names} ...",
    "answer" : "They are related through XYZ according to DOI ZYX"
}

```

# Uniprot Ids of Important Proteins

| uniprot_id  |
|-------------|
| TSP4_HUMAN  |
| TM131_HUMAN |
| COMP_HUMAN  |
| BGH3_HUMAN  |
| S10A6_HUMAN |
| AN32C_HUMAN |
| CALU_HUMAN  |
| SH3L3_HUMAN |
| PCH2_HUMAN  |
| FA83H_HUMAN |
| LACRT_HUMAN |
| MGT4B_HUMAN |
| BOLA2_HUMAN |
| CNBP_HUMAN  |
| LAMB3_HUMAN |
| ATOX1_HUMAN |
| NGAL_HUMAN  |
| AR6P1_HUMAN |
| PEPD_HUMAN  |
| KI16B_HUMAN |


# Throwing it at the wall

## Multi Hop Training

- Question: Is glucose related to protein a?
    - Context:
        - doc 1: sugar is related to protein a
        - doc 2: sugar is a synonym to glucose
    - Answer:
        - yes, glucose is related to protien a.

## How well can we do with simple queries

- Question: Is glucose related to protein a?
    - Context:
        - synonyms:
        - doc 1:
        - doc 2:
    - Answer:
        - 
