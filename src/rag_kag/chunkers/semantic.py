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

    # def _split_into_sentences(self, text: str) -> List[str]:
    #     """
    #     Splits the input text into a list of sentences.
    #     This is a basic implementation; a more robust solution would use NLTK's `sent_tokenize` or SpaCy.
    #     """
    #     sentences = []
    #     temp_sentence = ""
    #     # Add a space to ensure sentences ending at the very end of the text get processed
    #     text_with_terminator = text + " "
    #     for char in text_with_terminator:
    #         temp_sentence += char
    #         if char in ['.', '?', '!']:
    #             cleaned_sentence = temp_sentence.strip()
    #             if cleaned_sentence:
    #                 sentences.append(cleaned_sentence)
    #             temp_sentence = ""
    #         elif char == '\n': # Treat newlines as potential sentence breaks too
    #             cleaned_sentence = temp_sentence.strip()
    #             if cleaned_sentence and cleaned_sentence != '\n':
    #                 sentences.append(cleaned_sentence.replace('\n', ' ')) # Replace newline within sentence
    #             temp_sentence = ""
    #     # Final check for any remaining text that didn't end with a punctuation
    #     if temp_sentence.strip():
    #         sentences.append(temp_sentence.strip().replace('\n', ' ')) # Replace newline within sentence

    #     return [s.strip() for s in sentences if s.strip()]

    def chunk(self, example: Example) -> list[Chunk]:
        """
        Chunks the given text into semantically coherent segments.

        Args:
            example (Example): Object of type Example.

        Returns:
            list[Chunk]: A list of Chunk objects, each representing a semantically coherent text segment.
        """
        for doc_idx, doc_text in enumerate(example.documents):
            sentences = (
                example.documents_sentences[doc_idx]
                if doc_idx < len(example.documents_sentences)
                else []
            )
        print(sentences[0],'\n',sentences[1])
        if not sentences:
            return []

        # Add the condition to check sentence length
        if len(sentences) < self.min_sentences or len(sentences) > self.max_sentences:
            raise ValueError(
                f"Number of sentences ({len(sentences)}) is outside the allowed range "
                f"[{self.min_sentences}, {self.max_sentences}]")

        # Embed sentences. Use CUDA if available for faster computation.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sentence_embeddings = self.sentence_transformer.encode(sentences, convert_to_tensor=True, device=device)

        # Calculate cosine similarities between adjacent sentences
        similarities = []
        for i in range(len(sentence_embeddings) - 1):
            sim = util.cos_sim(sentence_embeddings[i], sentence_embeddings[i+1])
            similarities.append(sim.item())

        # Find breakpoints where similarity drops below threshold
        # A breakpoint index 'i' means that sentences[i] and sentences[i+1] are dissimilar,
        # so a new chunk should ideally start after sentences[i].
        breakpoints = [i for i, sim in enumerate(similarities) if sim < self.breakpoint_threshold]

        chunks = []
        current_sentence_idx = 0
        for bp in breakpoints:
            # Form a chunk from `current_sentence_idx` up to and including `bp`
            chunk_sentences = sentences[current_sentence_idx : bp + 1]
            chunk_text = " ".join(chunk_sentences)
            if chunk_text:
                chunks.append(Chunk(content=chunk_text))
            current_sentence_idx = bp + 1

        # Add the last chunk if there are any remaining sentences
        if current_sentence_idx < len(sentences):
            chunk_text = " ".join(sentences[current_sentence_idx:])
            if chunk_text:
                chunks.append(Chunk(content=chunk_text))

        return chunks
