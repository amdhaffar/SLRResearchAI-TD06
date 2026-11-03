from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Callable

from .rag_models import Chunk, ProposedEvidence, VerifiedEvidence


def validate_chunking(size: int, overlap: int) -> int:
    """Valide la configuration et retourne le pas de découpage."""
    # ACQUIS TD05 - Condition et calcul du pas.
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Il faut size > 0 et 0 <= overlap < size")
    return size - overlap


def chunk_pages(paper_id: str, pages: list[str], size: int = 400, overlap: int = 80) -> list[Chunk]:
    """Découpe des pages en fragments tout en conservant leur provenance."""
    step = validate_chunking(size, overlap)
    chunks: list[Chunk] = []
    # ACQUIS TD05 - Construire un Chunk avec les 4 champs attendus.
    for page_number, page_text in enumerate(pages, 1):
        for chunk_number, start in enumerate(range(0, len(page_text), step), 1):
            text = page_text[start : start + size].strip()
            if text:
                chunks.append(
                    Chunk(
                        paper_id=paper_id,
                        chunk_id=f"{paper_id}_p{page_number}_c{chunk_number}",
                        page=page_number,
                        text=text,
                    )
                )
    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ0-9]{2,}", text.casefold())


def _cosine_sparse(left: dict[str, float], right: dict[str, float]) -> float:
    common = set(left) & set(right)
    dot = sum(left[word] * right[word] for word in common)
    norm_left = math.sqrt(sum(value * value for value in left.values()))
    norm_right = math.sqrt(sum(value * value for value in right.values()))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0


def lexical_retrieve(chunks: list[Chunk], question: str, k: int = 3) -> list[dict]:
    """Classe les chunks par TF-IDF et cosinus."""
    # Le calcul TF-IDF est fourni ; ACQUIS TD05 - Trier et garder Top-k.
    documents = [tokenize(chunk.text) for chunk in chunks]
    query_terms = tokenize(question)
    n_docs = len(documents)
    document_frequency = Counter(
        word for words in documents for word in set(words)
    )

    def vector(words: list[str]) -> dict[str, float]:
        counts = Counter(words)
        return {
            word: count * (math.log((1 + n_docs) / (1 + document_frequency.get(word, 0))) + 1)
            for word, count in counts.items()
        }

    query_vector = vector(query_terms)
    scored = [
        {**chunk.model_dump(), "retrieval_score": _cosine_sparse(query_vector, vector(words))}
        for chunk, words in zip(chunks, documents)
    ]
    return sorted(scored, key=lambda item: item["retrieval_score"], reverse=True)[:k]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    # ACQUIS TD05 - Retourner le cosinus ou 0 si une norme est nulle.
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(value * value for value in left))
    norm_right = math.sqrt(sum(value * value for value in right))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0


def semantic_retrieve(chunks: list[Chunk], question: str, embeddings, k: int = 3) -> list[dict]:
    # L'encodage est fourni ; ACQUIS TD05 - Trier et garder Top-k.
    question_vector = embeddings.embed_query(question)
    document_vectors = embeddings.embed_documents([chunk.text for chunk in chunks])
    scored = [
        {**chunk.model_dump(), "retrieval_score": cosine_similarity(question_vector, vector)}
        for chunk, vector in zip(chunks, document_vectors)
    ]
    return sorted(scored, key=lambda item: item["retrieval_score"], reverse=True)[:k]


def retrieval_metrics(returned_ids: list[str], relevant_ids: set[str]) -> dict[str, float]:
    # ACQUIS TD05 - Formules de Precision@k et Recall@k.
    if not returned_ids:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0}
    hits = len(set(returned_ids) & relevant_ids)
    return {
        "precision_at_k": hits / len(returned_ids),
        "recall_at_k": hits / len(relevant_ids) if relevant_ids else 0.0,
    }


def build_context(results: list[dict]) -> str:
    # ACQUIS TD05 - Joindre les blocs traçables.
    blocks = [
        f'[PAPER {item["paper_id"]}] [CHUNK {item["chunk_id"]}] [PAGE {item["page"]}]\n{item["text"]}'
        for item in results
    ]
    return "\n\n".join(blocks)


def build_llm_extractor(model) -> Callable[[str, str], ProposedEvidence]:
    structured_model = model.with_structured_output(ProposedEvidence)

    # ACQUIS TD05 - Appeler le modèle structuré avec le prompt fourni.
    def extract(question: str, context: str) -> ProposedEvidence:
        prompt = f"""Tu extrais une preuve scientifique à partir du SEUL contexte fourni.
Si le contexte est insuffisant, indique-le dans answer sans inventer de citation.
Question : {question}

CONTEXTE DOCUMENTAIRE :
{context}

Recopie supporting_quote mot pour mot et conserve paper_id, chunk_id et page."""
        return structured_model.invoke(prompt)

    return extract


def _assess_proposal(proposal: ProposedEvidence, chunk: dict | None) -> tuple[str, str]:
    """Politique fournie : transforme la comparaison en statut."""
    if chunk is None:
        return "rejected", "Chunk inconnu dans le contexte récupéré."
    quote = " ".join(proposal.supporting_quote.split()).casefold()
    source = " ".join(chunk["text"].split()).casefold()
    provenance_ok = proposal.page == chunk["page"] and proposal.paper_id == chunk["paper_id"]
    if quote and quote in source and provenance_ok:
        return "verified", "Citation exacte et provenance cohérente."
    similarity = SequenceMatcher(None, quote, source).ratio()
    if quote and (similarity >= 0.60 or quote in source):
        return "human_review", "Citation proche ou provenance incohérente."
    return "rejected", "Citation absente du chunk annoncé."


def verify_evidence(proposal: ProposedEvidence, selected_chunks: list[dict]) -> VerifiedEvidence:
    """Vérifie l'identifiant, la page et la présence de la citation."""
    # ACQUIS TD05 - Retrouver le chunk portant proposal.chunk_id.
    chunk = next(
        (item for item in selected_chunks if item["chunk_id"] == proposal.chunk_id),
        None,
    )
    status, reason = _assess_proposal(proposal, chunk)
    return VerifiedEvidence(
        **proposal.model_dump(), status=status, verification_reason=reason
    )


def run_rag(
    chunks: list[Chunk],
    question: str,
    retriever: Callable[[list[Chunk], str, int], list[dict]],
    extractor: Callable[[str, str], ProposedEvidence],
    k: int = 3,
) -> dict:
    # Les trois premières étapes sont fournies ; ACQUIS TD05 - Vérifier.
    selected = retriever(chunks, question, k)
    context = build_context(selected)
    proposal = extractor(question, context)
    evidence = verify_evidence(proposal, selected)
    return {"selected_chunks": selected, "context": context, "evidence": evidence}
