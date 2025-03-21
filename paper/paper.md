---
title: 'GenrAItOR: Generative AI for ‘Omics Research'
tags:
  - Python
  - multiomics
  - LLM
  - proteins
authors:
  - name: Daniel Claborne
    orcid: 0000-0001-5293-3628
    affiliation: 1
  - name: Matthew Jensen
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Javier E. Flores
    orcid: 0000-0002-1550-1655
    affiliation: 1
  - name: Lisa Bramer
    orcid: 0000-0002-8384-1926
    affiliation: 1
  - name: Samantha Erwin
    orcid: 0000-0002-5162-4193
    affiliation: 1
    corresponding: true
affiliations:
 - name: Pacific Northwest National Laboratory
   index: 1
date: 24 December 2024
bibliography: refs.bib

---

# Summary

We present software to employ Retrieval Augmented Fine-Tuning (RAFT) to fine-tune Llama 3, a large language model (LLM), using the textual corpora of proteomics-related literature harvested from publicly available databases and abstracts.  The software is organized as a command-line interface (CLI) to abstract away tedious details in the fine-tuning process.  The resulting Llama 3-RAFT model accepts queries about biomolecules (proteins) and returns relevant biological information (e.g., reaction pathways, function) based on the user-provided context and learned patterns from the RAFT fine tuning.

Recent advances in LLMs provide an opportunity to improve the efficiency of the human-dependent aspects of model interrogation, e.g. domain experts making sense of sets of biomolecules determined to be 'important' to model prediction.  Specifically, domain-level experts may query an LLM for biological information on the indicated biomolecules and draw domain-level inference that is contextualized by the knowledge contained within and relayed by the LLM.

Many existing LLMs are general purpose, having been trained on the vast corpora of data available from social media and other public sources. Since these foundational LLMs were trained without the domain-specific language required by ‘omic-based queries, the aim of this work is to use RAFT [@zhang_raft_2024] to update an open-source, foundational LLM so that it may serve as an AI-assistant to the domain expert in their contextualization of important modeled features.  The RAFT approach is a special type of fine-tuning (FT) that includes context in a question-answering task where some or all of the context may be irrelevant to answering the question. By including irrelevant information within the supplied context, the RAFT approach is more robust to the presence of irrelevant context relative to a traditional retrieval-augmented generation (RAG) system [@zhang_raft_2024].  We present a software package/CLI that performs the required steps in RAFT from data collection to training to enable other interested researchers to develop their own AI-assistants for biological research.  

![Genraitor Process Overview. Synthetic training data are generated using ChatGPT-4o. These question-answer-context triplicates are then used to fine-tune Llama 3 in a RAFT context. The output RAFT model is then implemented/evaluated on a hold-out set of generated triplicates.](images/workflow.png){#fig:workflow}

# Statement of Need

The RAFT process requires several steps and many software dependencies, as well as modification of some of the code in those dependencies.  This presents a large barrier to experimentation with the RAFT technique on top of already restrictive hardware requirements.  The software we present provides an easy-to-use CLI for researchers to implement RAFT in the proteomics domain, as well as a starting template to adapt the code to other `omics-related research.  It also conveniently organizes the many dependencies required to perform RAFT.

An overview of our development of a RAFT model is provided by [Figure 1](#fig:workflow).  The general steps are:

1. PubMed abstracts and UniProt data such as interactions were harvested using their publicly available APIs.

2. These data were provided as chunks of context to GPT-4o with instructions to generate question-answer pairs to use as synthetic data.  Context chunks are determined by grouping semantically similar text via a text embedding approach.  An example question-answer pair used for RAFT training is given in [Figure 2](#fig:qapair).

3. Context is prepended to the Q-A pair by sampling random 'distractor' documents and also including the context used to generate the Q-A pair with some probability.  This creates contexts with varied levels of relevancy.

4. GPT-4o synthetic data was split into training and evaluation subsets, with the training subset used to implement RAFT on Llama 3. 

5. The evaluation subset was used to compare the performance of the RAFT-Llama 3 to the RAG-Llama 3 via the Align Score [@zha_alignscore_2023].

![Example of a QA-pair generated using sampled context. The \[... CONTEXT ...\] chunk is a collection of documents that may or may not contain the text used to generate the QA-pair](images/qapair.png){#fig:qapair}

# Example Implementation

The entire process is encapsulated into a python package and run using the built in command line interface.  After installing the package and its dependencies as described in the README, we can proceed to retrieve some context and perform RAFT.  In `data/examples/uniprots.txt` you will find several protein identifiers known as Accession numbers.  These will be used to search the uniprot database and retrieve article abstracts as well as information about protein-protein interactions and associated pathways.  

Our first step is to retrieve this context.  We can do so by invoking the cli as follows:

```bash
python -m genraitor data:context \
--uniprot_ids=./data/examples/uniprots.txt \
--output_dir=./data
```

This will produce two files in `./data`, one (`uniprot_context_results...`) with the raw results of querying uniprot for pathway information and abstracts, and the other (`uniprot_context_postprocessed...`) with context derived from those results and usable by the `RAFTDatasetPack` class from `llama-index`.

Once you have the text file of context, you can use our modified `RAFTDatasetPack` class to create synthetic question-answer pairs about chunks of that context.  You will need an OpenAI API key as well as a huggingface API key if you use a local embedding model.  The entrypoint for the cli is `raft:data`, and can be run as below using a huggingface embedding model and GPT-4o through the OpenAI api.

```bash
# set keys
export HF_TOKEN=<your-hf-token>
export OPENAI_API_KEY=<your-oai-key>

python -m genraitor raft:data \
--embed local \
--context_path /path/to/context.txt \
--output_path ./data/training/hf_dataset
```

This will produce the folder `./data/training/hf_dataset` which can be loaded via:

```python
from datasets import load_from_disk

dataset = load_from_disk('/path/to/save_data_folder')
```

To perform RAFT using this dataset, we simply point the cli target `train:raft` for training to the dataset on disk.  The cli target also takes a model name to be passed to the huggingface `AutoModelForCausalLM.from_pretrained` method as well as an output path.  For certain models, such as the Llama series, you will again need a huggingface api key and have accepted the terms of service on their model page.

```bash
python -m genraitor train:raft \
-t /path/to/raft_data \
-m meta-llama/Meta-Llama-3.1-8B \
-n data/finetuned
```

The fine-tuned model will be saved in data/finetuned and loadable via the huggingface interface:

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

tokenizer = AutoTokenizer.from_pretrained('./data/finetuned', padding_side="left")
model = AutoModelForCausalLM.from_pretrained("./data/finetuned")
```

The commands are configurable via flags or environment variables. The default options are set by reading from either the environment or a .env file. For help with available options, pass the --help flag to any of the commands:

```python
python3 -m genraitor train:raft --help
```

In addition to wrapping data processing, model training and result evaluation into a single python package, this effort extended the python packages `llama-index` [@liu_llamaindex_2022] in the following ways:

- We modified the `llama-index` plugin `llama-index-packs-raft-dataset` to allow configurable system prompts when generating questions for each chunk of the RAG documents. This allowed us to experiment with prompt engineering to improve the relevance of the generated questions to each document.
- We modified the source code for the function `get_chunks` to respect the parameter `chunk_size` when parsing the retrieved documents. This modification allowed us to optimize the length of text each document used for training.

## Results

We evalute our methods output through the comparison of align scores. In Figure 2 we compare the distribution of the align scores for the RAFT-Llama 3 model (in blue) and the base Llama 3 model. While both distributions appear largely similar, it should be noted that:

- Mean align score is slightly improved in the RAFT model relative to the base Llama 3
- In line with the improved mean, the RAFT distribution of align scores is more left skewed

Jointly, these results indicate that the RAFT model more typically generates responses that are marginally better aligned with the truth. Furthermore the software package provided herein provides a streamlines method to implemnt the required steps in RAFT from data collection to training to enable other  researchers to develop a RAFT-Llama 3 model. 

![Distribution of Align Scores for the RAFT-Llama 3 (“Finetuned”) and the RAG-Llama3 model.](images/results.png){#fig:results}

# Discussion/Limitations

Though we provide an easy-to-use interface to the RAFT process, some challenges and limitations remain:

1. **Model Training/Computation Challenges.** Large context chunks were sometimes difficult to load into GPU memory, forcing us to limit the generated context size. Long training/generation times for modestly sized datasets required careful checkpointing/monitoring.  Libraries and techniques that reduce memory footprint exacerbate dependency issues.

2.  **Domain Specificity to Proteomics.**  Our package is currently focused on obtaining context related to lists of proteins.  Custom functions would have to be written into the framework to harvest context for other domains.  Once this is done however it is straightforward to change the system prompt for creation of the RAFT-ready dataset.  Finally, the training process is agnostic to the domain once the dataset has been created.

3. **Dependency Mismatches.**  There are many dependencies in any LLM fine-tuning project, even with good tracking of python dependencies, there will be issues such as CUDA version mismatches, certain package versions not being available on particular operating systems, etc.  For example, to use the public implementation of AlignScore, the package needed to be updated to work with newer versions of PyTorch.

Our work found comparable performance between RAFT and RAG implementations of Llama 3 on ‘omics-based queries, with the RAFT implementation showing marginal improvement.

The align scores of each approach indicate a similar ability to summarize the original input documents that contain the information of interest, thereby demonstrating some promise in either approach for service as an AI-assistant to the inquiring ‘omics expert.

Importantly, these results were achieved based on synthetically generated data. These synthetic data were not verified for their fidelity to source truth, and thus it is entirely possible that better results may be obtained through a more involved curation of training data led by biological experts.

Our software package allows users to create context for performing RAFT in the proteomics domain given a list of proteins of interest.  We hope this provides an easy-to-use base for researchers to explore the relationships between proteins identified in their experiments and expand the use of RAFT to different domain areas.

<!--- Can we soften this discussion point? Also, what can we discuss that was an improvement?--->

# Acknowledgements

We acknowledge contributions from

<!---Include GenAI investments--->

# References
---
bibliography: sample.bib
nocite: "@*"
---
