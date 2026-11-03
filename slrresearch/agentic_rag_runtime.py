from __future__ import annotations

from .rag_models import Chunk, ProposedEvidence
from .rag_pipeline import lexical_retrieve, semantic_retrieve


class ProtocolEmbeddings:
    """Embeddings contrôlés pour isoler la logique agentique dans les tests."""

    def __init__(self, chunks: list[Chunk], relevant_ids: set[str]):
        self.chunks = chunks
        self.relevant_ids = relevant_ids

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0] if chunk.chunk_id in self.relevant_ids else [0.0, 1.0]
            for chunk in self.chunks
        ]


def build_retrieval_tools(embeddings) -> dict:
    return {
        "lexical": lexical_retrieve,
        "semantic": lambda chunks, question, k: semantic_retrieve(
            chunks, question, embeddings, k
        ),
    }


def controlled_extractor(state):
    """Générateur reproductible : cite un passage effectivement sélectionné."""

    def extract(question: str, context: str) -> ProposedEvidence:
        candidate = state.selected_chunks[0]
        return ProposedEvidence(
            answer="Les passages sélectionnés fournissent une preuve contrôlable.",
            supporting_quote=candidate["text"],
            paper_id=candidate["paper_id"],
            chunk_id=candidate["chunk_id"],
            page=candidate["page"],
        )

    return extract
