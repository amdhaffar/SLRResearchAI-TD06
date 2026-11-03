import pytest
import slrresearch.rag_pipeline as pipeline

from slrresearch.rag_models import Chunk, ProposedEvidence
from slrresearch.rag_pipeline import (
    build_context,
    build_llm_extractor,
    chunk_pages,
    cosine_similarity,
    lexical_retrieve,
    retrieval_metrics,
    run_rag,
    semantic_retrieve,
    validate_chunking,
    verify_evidence,
)


def sample_chunks():
    return [
        Chunk(paper_id="P1", chunk_id="P1_p1_c1", page=1, text="méthode générale"),
        Chunk(paper_id="P1", chunk_id="P1_p2_c1", page=2, text="le rappel du screening atteint 92 pour cent"),
    ]


def test_todo_1_refuse_un_overlap_invalide():
    with pytest.raises(ValueError):
        validate_chunking(size=10, overlap=10)
    assert validate_chunking(size=10, overlap=2) == 8


def test_todo_2_conserve_la_provenance(monkeypatch):
    # Isole le TODO 2 : le calcul du pas du TODO 1 est déjà considéré correct.
    monkeypatch.setattr(pipeline, "validate_chunking", lambda size, overlap: 4)
    chunks = chunk_pages("P7", ["abcdefghij"], size=6, overlap=2)
    assert chunks[0].chunk_id == "P7_p1_c1" and chunks[0].page == 1
    assert chunks[1].text == "efghij"


def test_todo_3_tfidf_classe_le_passage_lexical():
    result = lexical_retrieve(sample_chunks(), "rappel screening", k=1)
    assert result[0]["chunk_id"] == "P1_p2_c1" and result[0]["retrieval_score"] > 0


def test_todo_4_cosinus_gere_vecteurs_orthogonaux_et_nuls():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0)
    assert cosine_similarity([0, 0], [1, 1]) == 0


def test_todo_5_embeddings_classent_par_le_sens(monkeypatch):
    # Isole le TODO 5 du calcul mathématique demandé au TODO 4.
    monkeypatch.setattr(pipeline, "cosine_similarity", lambda left, right: sum(a*b for a, b in zip(left, right)))
    class FakeEmbeddings:
        def embed_query(self, text): return [1.0, 0.0]
        def embed_documents(self, texts): return [[0.0, 1.0], [0.9, 0.1]]

    result = semantic_retrieve(sample_chunks(), "question paraphrasée", FakeEmbeddings(), k=1)
    assert result[0]["chunk_id"] == "P1_p2_c1"


def test_todo_6_mesures_exigent_des_references_annotees():
    metrics = retrieval_metrics(["c1", "c2"], {"c2", "c3"})
    assert metrics == {"precision_at_k": 0.5, "recall_at_k": 0.5}


def test_todo_7_contexte_affiche_source_chunk_et_page():
    context = build_context([{**sample_chunks()[1].model_dump(), "retrieval_score": 0.8}])
    assert "[PAPER P1]" in context and "[CHUNK P1_p2_c1]" in context and "[PAGE 2]" in context


def test_todo_8_sortie_llm_est_structuree_et_contrainte():
    class Structured:
        def invoke(self, prompt):
            assert "SEUL contexte" in prompt and "CONTEXTE DOCUMENTAIRE" in prompt
            return ProposedEvidence(answer="92 %", supporting_quote="92 %", paper_id="P1", chunk_id="c1", page=1)
    class Model:
        def with_structured_output(self, schema):
            assert schema is ProposedEvidence
            return Structured()
    assert build_llm_extractor(Model())("question", "contexte").answer == "92 %"


def test_todo_9_refuse_une_citation_inventee():
    valid = ProposedEvidence(answer="rappel", supporting_quote="92 pour cent", paper_id="P1", chunk_id="P1_p2_c1", page=2)
    assert verify_evidence(valid, [sample_chunks()[1].model_dump()]).status == "verified"
    proposal = ProposedEvidence(answer="x", supporting_quote="citation inventée", paper_id="P1", chunk_id="P1_p2_c1", page=2)
    assert verify_evidence(proposal, [sample_chunks()[1].model_dump()]).status == "rejected"


def test_todo_10_pipeline_ne_transmet_que_le_top_k(monkeypatch):
    seen = {}
    def retriever(chunks, question, k): return [chunks[1].model_dump()]
    def extractor(question, context):
        seen["context"] = context
        return ProposedEvidence(answer="rappel", supporting_quote="92 pour cent", paper_id="P1", chunk_id="P1_p2_c1", page=2)
    # Isole l'orchestration du TODO 10 : les TODO 7 et 9 sont remplacés par des
    # composants contrôlés, comme dans une architecture à dépendances injectées.
    monkeypatch.setattr(pipeline, "build_context", lambda selected: selected[0]["text"])
    expected = object()
    monkeypatch.setattr(pipeline, "verify_evidence", lambda proposal, selected: expected)
    result = run_rag(sample_chunks(), "rappel", retriever, extractor, k=1)
    assert "méthode générale" not in seen["context"]
    assert result["evidence"] is expected
