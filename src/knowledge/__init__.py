"""Knowledge domain — swappable backend, function-named agent tools."""

from knowledge.backend import KnowledgeBackend, KnowledgeEntry, KnowledgeSearchHit
from knowledge.chroma import ChromaKnowledgeBackend, build_chroma_backend
from knowledge.tools import create_tools

__all__ = [
    "ChromaKnowledgeBackend",
    "KnowledgeBackend",
    "KnowledgeEntry",
    "KnowledgeSearchHit",
    "build_chroma_backend",
    "create_tools",
]
