from llama_index.packs.raft_dataset import RAFTDatasetPack
from llama_index.core.llms import ChatMessage
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser

import sys
import logging
import warnings
import pickle
from typing import Any, List, TypedDict
import numpy as np

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

DEFAULT_INSTRUCTGEN_TEMPLATE = "You are a synthetic question-answer pair generator. Given a chunk of context about some topic(s), generate %s example questions a user could ask and would be answered using information from the chunk. For example, if the given context was a Wikipedia paragraph about the United States, an example question could be 'How many states are in the United States?'. The questions should be able to be answered in a few words or less."

class SentenceCombination(TypedDict):
    sentence: str
    index: int
    combined_sentence: str
    combined_sentence_embedding: List[float]

class TruncatedNodeParser(SemanticSplitterNodeParser):
    max_chunk_len: Any
    min_chunk_len: Any

    def __init__(self, *args, max_chunk_len: int = 1024, min_chunk_len: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_chunk_len = max_chunk_len
        self.min_chunk_len = min_chunk_len

    def _assemble_truncated_chunks(self, group: List[SentenceCombination], chunks: List = []) -> List[str]:
        combined_text = ""
        for g in group:
            # assume approximately 4 characters per token
            sent_len = len(g['sentence']) / 4
            cur_len = len(combined_text) / 4 + sent_len

            if sent_len > self.max_chunk_len:
                continue
            elif cur_len > self.max_chunk_len:
                if len(combined_text) / 4 > self.min_chunk_len:
                    chunks.append(combined_text)
                combined_text = g['sentence']
            else:
                combined_text += g['sentence']
        else:
            if len(combined_text) / 4 > self.min_chunk_len:
                chunks.append(combined_text)

    def _build_node_chunks(
    self, sentences: List[SentenceCombination], distances: List[float]) -> List[str]:
        chunks = []
        if len(distances) > 0:
            breakpoint_distance_threshold = np.percentile(
                distances, self.breakpoint_percentile_threshold
            )

            indices_above_threshold = [
                i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
            ]

            # Chunk sentences into semantic groups based on percentile breakpoints
            start_index = 0

            for index in indices_above_threshold:
                group = sentences[start_index : index + 1]

                self._assemble_truncated_chunks(group, chunks)

                start_index = index + 1

            if start_index < len(sentences):
                group = sentences[start_index:]
                self._assemble_truncated_chunks(group, chunks)
        else:
            # If, for some reason we didn't get any distances (i.e. very, very small documents) just
            # treat the whole document as a single node
            chunks = [" ".join([s["sentence"] for s in sentences])]

        return chunks

class BioRAFTDatasetPack(RAFTDatasetPack):
    # init with a new keyword argument
    def __init__(self, instruction_template=DEFAULT_INSTRUCTGEN_TEMPLATE, min_chunk_size = 100, **kwargs):
        super().__init__(**kwargs)
        self.instruction_template = instruction_template
        self.min_chunk_size = min_chunk_size

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
    
    def get_chunks(self, file_path: str, max_chunk_size: int, min_chunk_size: int) -> List[str]:
        """
        Takes in a `file_path`, retrieves the document, breaks it down into chunks of size
        `chunk_size`, and returns the chunks.
        """
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        splitter = TruncatedNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=self.default_breakpoint_percentile_threshold,
            embed_model=self.embed_model,
            max_chunk_len=max_chunk_size,
            min_chunk_len=min_chunk_size
        )
        nodes = splitter.get_nodes_from_documents(documents)

        return [node.get_content() for node in nodes]

    def run(self, checkpoint_path = None, chunks = None, save_chunks_path = None) -> Any:
        """Run the pipeline."""
        if chunks is None:
            chunks = self.get_chunks(self.file_path, self.chunk_size, self.min_chunk_size)

            if save_chunks_path is not None:
                pickle.dump(chunks, open(save_chunks_path, 'wb'))

        logging.info(f"Number of chunks created: {len(chunks)}")

        self.num_distract_docs = (
            min(self.num_distract_docs, len(chunks)) - 1
        )  # should be less than number of chunks/ nodes created

        for index, chunk in enumerate(chunks):
            logging.info(f"Processing chunk: {index}")
            self.add_chunk_to_dataset(
                chunks, chunk, self.num_questions_per_chunk, self.num_distract_docs
            )

            if index % 100 == 0 and index > 0 and checkpoint_path is not None:
                logging.info(f"Completed {index} chunks, saving checkpoint")
                self.ds.save_to_disk(checkpoint_path)

        return self.ds
    
