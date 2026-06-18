import os
import numpy as np
import torch # For checking CUDA availability
from typing import List, Dict, Any
from dataclasses import dataclass
from rag_kag.types import Chunk, Example, Sentence
from rag_kag.chunkers.base import Chunker

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer

class SemanticChunker(Chunker):
    """
    A chunker that uses semantic similarity to split text into chunks.
    Sentences are embedded, and chunks are formed by grouping sentences
    until the cosine similarity between adjacent sentences drops below a threshold.
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", min_sentences: int = 2, max_sentences: int = 12, breakpoint_threshold: float = 0.75):
        """
        Initializes the SemanticChunker.

        Args:
            config (Dict[str, Any]): A dictionary containing configuration parameters,
                                     expected to have 'embedder.model_name',
                                     'chunker.chunk_size', and 'chunker.overlap'
                                     (though chunk_size and overlap are not strictly used in this semantic splitting logic,
                                     they might be used in a hybrid approach or for metadata).
        """
        self.model_name = model_name
        # chunk_size and overlap are kept for consistency with other chunkers, but not directly
        # used in this semantic splitting logic. They could be used for further refinement.
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences

        # Initialize the tokenizer and sentence transformer model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.sentence_transformer = SentenceTransformer(self.model_name)
        self.breakpoint_threshold = breakpoint_threshold # Default threshold, can be made configurable in the future

    def chunk(self, example: Example) -> list[Chunk]:
        """
        Chunks the given text into semantically coherent segments.

        Args:
            example (Example): Object of type Example.

        Returns:
            list[Chunk]: A list of Chunk objects, each representing a semantically coherent text segment.
        """
        all_final_chunks = []
        global_chunk_counter = 0
        effective_max_chunk_sentences = 12 # Hardcoded limit as per user's request

        for doc_idx, doc_sentences_list in enumerate(example.documents_sentences):
            sentences = doc_sentences_list

            if not sentences:
                continue

            # Extract string content from Sentence objects
            string_sentences = [s.text for s in sentences]

            # Embed sentences. Use CUDA if available for faster computation.
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            sentence_embeddings = self.sentence_transformer.encode(string_sentences, convert_to_tensor=True, device=device)

            # Calculate cosine similarities between adjacent sentences
            similarities = []
            for i in range(len(sentence_embeddings) - 1):
                sim = util.cos_sim(sentence_embeddings[i], sentence_embeddings[i+1])
                similarities.append(sim.item())

            # Find breakpoints where similarity drops below threshold
            breakpoints = [i for i, sim in enumerate(similarities) if sim < self.breakpoint_threshold]

            current_start_idx = 0

            # Process segments defined by semantic breakpoints
            for bp in breakpoints:
                segment_to_process = sentences[current_start_idx : bp + 1]
                for i in range(0, len(segment_to_process), effective_max_chunk_sentences):
                    sub_chunk_sentences = segment_to_process[i : i + effective_max_chunk_sentences]
                    if len(sub_chunk_sentences) >= self.min_sentences:
                        chunk_text = " ".join([s.text for s in sub_chunk_sentences])
                        current_sentence_keys = [s.key for s in sub_chunk_sentences]
                        chunk_id = f"{example.id}-{doc_idx}-{global_chunk_counter}"
                        chunk_metadata = {"example_id": example.id, "doc_index": doc_idx}
                        all_final_chunks.append(Chunk(chunk_id, chunk_text, doc_idx, current_sentence_keys, chunk_metadata))
                        global_chunk_counter += 1
                current_start_idx = bp + 1

            # Process any remaining sentences after the last breakpoint for the current document
            remaining_sentences_doc = sentences[current_start_idx:]
            if remaining_sentences_doc:
                for i in range(0, len(remaining_sentences_doc), effective_max_chunk_sentences):
                    sub_chunk_sentences = remaining_sentences_doc[i : i + effective_max_chunk_sentences]
                    if len(sub_chunk_sentences) >= self.min_sentences:
                        chunk_text = " ".join([s.text for s in sub_chunk_sentences])
                        current_sentence_keys = [s.key for s in sub_chunk_sentences]
                        chunk_id = f"{example.id}-{doc_idx}-{global_chunk_counter}"
                        chunk_metadata = {"example_id": example.id, "doc_index": doc_idx}
                        all_final_chunks.append(Chunk(chunk_id, chunk_text, doc_idx, current_sentence_keys, chunk_metadata))
                        global_chunk_counter += 1

        return all_final_chunks
