from dataclasses import dataclass, field
from typing import List


# Represents a single evidence document retrieved for a claim.
@dataclass
class Document:
    doc_id: str
    text: str
    label: str = "real"  # "real" for clean documents, "poisoned" for injected ones


# Represents one claim with its gold label and associated evidence documents.
@dataclass
class QASample:
    question_id: str
    question: str
    gold_answer: str
    documents: List[Document] = field(default_factory=list)
