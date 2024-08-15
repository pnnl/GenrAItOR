# Getting Context From Uniprot and Making RAFT Dataset

In this folder are a couple scripts that help with the creation of a synthetic dataset for retrieval-augmented fine-tuning.  Requirements to run everything should be in `environment.yml` at the top level of the repo.

First, see `create_context.ipynb` to see how the context was created.  Essentially abstracts, interactions, and pathway information are fetched from PubMed/Uniprot.  These are collected into a single text file that is chunked up and used as context to create synthetic QA pairs.

Second, we run the `raft-dataset.py` script which constructs the RAFT dataset.  We specify the location of the context file with `--context_path`.  You will also probably need to specify your api_key for the on-prem OpenAI api given to you by RC.  If you are using a local huggingface model to extract embeddings, you will also need to provide a huggingface access token.  See `python raft-dataset.py --help` for options.

The huggingface RAFT dataset will be dumped to a folder, and can be loaded as follows:

```python
from datasets import load_from_disk

dataset = load_from_disk('/path/to/hf/dataset/folder')
```
