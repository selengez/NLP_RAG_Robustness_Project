# Builds a FAISS index from documents using SentenceTransformers embeddings.

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.data_pipeline.loader import Document, QASample
from src.data_pipeline.poisoner import PoisonedDocument


@dataclass
class ChunkMetadata:
    """Metadata for one indexed chunk, stored parallel to its embedding vector."""
    doc_id:      str
    chunk_index: int
    text:        str
    source_type: str  # "clean" or "poisoned"
    poison_type: str  # strategy name or ""


def chunk_text(text, chunk_size, chunk_overlap):
    """Split text into overlapping word-based windows."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    step   = max(1, chunk_size - chunk_overlap)
    chunks = []
    start  = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks


class DocumentIndexer:
    """Chunks documents, encodes them, and stores them in a FAISS index."""

    def __init__(self, model_name="all-MiniLM-L6-v2", chunk_size=100, chunk_overlap=20):
        self.model_name    = model_name
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self._model        = SentenceTransformer(model_name)
        self.index         = None
        self.metadata      = []

    def _doc_metadata(self, doc, chunk_index, text):
        """Build chunk metadata from a document."""
        if isinstance(doc, PoisonedDocument):
            source_type = "poisoned"
            poison_type = doc.poison_type
        else:
            source_type = "clean"
            poison_type = ""
        return ChunkMetadata(
            doc_id      = doc.doc_id,
            chunk_index = chunk_index,
            text        = text,
            source_type = source_type,
            poison_type = poison_type,
        )

    def index_documents(self, documents):
        """Chunk all documents, encode them, and build the FAISS index."""
        all_texts    = []
        all_metadata = []

        # Split each document into chunks and record metadata for each chunk
        for doc in documents:
            chunks = chunk_text(doc.text, self.chunk_size, self.chunk_overlap)
            for i, chunk in enumerate(chunks):
                all_texts.append(chunk)
                all_metadata.append(self._doc_metadata(doc, i, chunk))

        n_chunks = len(all_texts)
        n_docs   = len(documents)
        print(f"Encoding {n_chunks} chunks from {n_docs} documents...")

        embeddings = self._model.encode(
            all_texts,
            batch_size        = 64,
            show_progress_bar = True,
            convert_to_numpy  = True,
        ).astype(np.float32)

        # Build the FAISS index from the embeddings
        dim        = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.metadata = all_metadata

        n_poisoned = sum(1 for m in self.metadata if m.source_type == "poisoned")
        print(
            f"Index built — {self.index.ntotal} vectors | dim={dim} | "
            f"{n_poisoned} poisoned chunks ({n_chunks - n_poisoned} clean)"
        )

    def save(self, index_dir):
        """Save the FAISS index, chunk metadata, and build config to disk."""
        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)

        # Save the FAISS index binary
        faiss.write_index(self.index, str(path / "index.faiss"))

        # Save chunk metadata as JSON (asdict converts each dataclass to a plain dict)
        metadata_list = []
        for m in self.metadata:
            metadata_list.append(asdict(m))
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata_list, f, indent=2)

        # Save build config so the index can be reloaded correctly
        n_poisoned = sum(1 for m in self.metadata if m.source_type == "poisoned")
        with open(path / "index_config.json", "w") as f:
            json.dump(
                {
                    "model_name":    self.model_name,
                    "chunk_size":    self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "n_vectors":     self.index.ntotal,
                    "embedding_dim": self.index.d,
                    "n_clean":       len(self.metadata) - n_poisoned,
                    "n_poisoned":    n_poisoned,
                },
                f,
                indent=2,
            )

        print(f"Saved → {index_dir}  ({self.index.ntotal} vectors)")

    @classmethod
    def load(cls, index_dir):
        """Load a saved index from disk."""
        path = Path(index_dir)

        with open(path / "index_config.json") as f:
            config = json.load(f)

        indexer = cls(
            model_name    = config["model_name"],
            chunk_size    = config["chunk_size"],
            chunk_overlap = config["chunk_overlap"],
        )
        indexer.index = faiss.read_index(str(path / "index.faiss"))

        # Reconstruct ChunkMetadata objects from the saved JSON
        with open(path / "metadata.json") as f:
            raw = json.load(f)
        indexer.metadata = []
        for r in raw:
            indexer.metadata.append(ChunkMetadata(**r))

        print(
            f"Loaded ← {index_dir}  "
            f"({config['n_vectors']} vectors | "
            f"{config['n_poisoned']} poisoned)"
        )
        return indexer


def docs_from_samples(samples):
    """Flatten all documents from a list of QASamples into a single list."""
    docs = []
    for s in samples:
        for doc in s.documents:
            docs.append(doc)
    return docs
