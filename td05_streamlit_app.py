import json
from pathlib import Path

import streamlit as st

from slrresearch.rag_artifacts import load_rag_handoff, save_evidence_artifact
from slrresearch.config import Settings, build_model, build_embeddings
from slrresearch.rag_models import ProposedEvidence
from slrresearch.rag_pipeline import (
    build_llm_extractor,
    chunk_pages,
    lexical_retrieve,
    retrieval_metrics,
    run_rag,
    semantic_retrieve,
)

st.set_page_config(page_title="TD05 RAG classique", page_icon=":material/search:", layout="wide")
st.title("TD05 — Construire et évaluer un RAG classique")
st.caption("Même corpus, même question et même Top-k : seule la stratégie de retrieval varie.")


@st.cache_data
def load_experiment():
    artifact_dir = Path(__file__).parent / "artifacts"
    documents, lineage = load_rag_handoff(artifact_dir)
    evaluation = json.loads((artifact_dir / "rag_evaluation.json").read_text(encoding="utf-8"))
    return documents, lineage, evaluation


def corpus_chunks(documents):
    chunks = []
    for document in documents:
        chunks.extend(chunk_pages(document.paper_id, document.pages, size=1000, overlap=0))
    return chunks


class PedagogicalEmbeddings:
    """Vecteurs contrôlés : le mode réel est disponible séparément."""
    def __init__(self, relevant_id: str, chunks):
        self.relevant_id = relevant_id
        self.chunks = chunks

    def embed_query(self, text):
        return [1.0, 0.0]

    def embed_documents(self, texts):
        return [
            [1.0, 0.0] if chunk.chunk_id == self.relevant_id else [0.0, 1.0]
            for chunk in self.chunks
        ]


documents, lineage, evaluation = load_experiment()
chunks = corpus_chunks(documents)
questions = {item["id"]: item for item in evaluation["questions"]}

st.subheader("Continuité des artefacts")
with st.container(horizontal=True):
    st.metric("TD03 · trouvés", lineage["found"])
    st.metric("TD04 · inclus", lineage["included"])
    st.metric("Documents disponibles", lineage["documents_available"])
    st.metric("Corpus RAG", lineage["rag_documents"])
with st.expander("Voir la transmission TD03 → TD04 → RAG"):
    st.json(lineage)
    st.write("Le RAG ne reçoit que les articles à la fois recherchés, inclus et disponibles.")

if not chunks:
    st.warning("Complétez les TODO 1 et 2 pour transformer les documents transmis en chunks.")
    st.stop()

with st.container(border=True):
    st.subheader("Protocole expérimental commun")
    st.caption("Ces paramètres s'appliquent simultanément aux deux retrievers.")
    protocol_question, protocol_options = st.columns([2, 1], vertical_alignment="bottom")
    with protocol_question:
        question_id = st.selectbox(
            "Question annotée",
            list(questions),
            format_func=lambda key: f'{key} — {questions[key]["question"]}',
        )
    with protocol_options:
        k = st.slider("Nombre de passages — Top-k", 1, min(5, len(chunks)), 2)
        embedding_mode = st.segmented_control(
            "Calcul des embeddings",
            ["Contrôlés", "Ollama réel"],
            default="Contrôlés",
        )

question_case = questions[question_id]
question = question_case["question"]
relevant = set(question_case["relevant_chunk_ids"])
protocol_signature = (question_id, k, embedding_mode)

if st.button("Lancer la comparaison des retrievers", icon=":material/compare_arrows:"):
    try:
        embeddings = (
            build_embeddings(Settings())
            if embedding_mode == "Ollama réel"
            else PedagogicalEmbeddings(next(iter(relevant)), chunks)
        )
        lexical = lexical_retrieve(chunks, question, k)
        semantic = semantic_retrieve(chunks, question, embeddings, k)
        st.session_state["retrieval_experiment"] = {
            "signature": protocol_signature,
            "lexical": lexical,
            "semantic": semantic,
            "lexical_metrics": retrieval_metrics(
                [item["chunk_id"] for item in lexical], relevant
            ),
            "semantic_metrics": retrieval_metrics(
                [item["chunk_id"] for item in semantic], relevant
            ),
        }
        st.session_state.pop("rag_result", None)
    except Exception as exc:
        st.error(f"Retrieval impossible : {exc}")

experiment = st.session_state.get("retrieval_experiment")
if experiment and experiment["signature"] != protocol_signature:
    experiment = None

st.subheader("1. Comparer les retrievers")
if experiment is None:
    st.info(
        "Choisissez une question et les paramètres, puis lancez la comparaison. "
        "Les Top-k, Precision@k et Recall@k seront calculés après l'exécution."
    )
    st.stop()

lexical = experiment["lexical"]
semantic = experiment["semantic"]
lexical_metrics = experiment["lexical_metrics"]
semantic_metrics = experiment["semantic_metrics"]
left, right = st.columns(2)
for column, title, results, metrics in [
    (left, "Lexical — TF-IDF", lexical, lexical_metrics),
    (right, "Sémantique — embeddings", semantic, semantic_metrics),
]:
    with column:
        with st.container(border=True):
            st.markdown(f"### {title}")
            with st.container(horizontal=True):
                st.metric("Precision@k", f'{metrics["precision_at_k"]:.2f}')
                st.metric("Recall@k", f'{metrics["recall_at_k"]:.2f}')
            st.dataframe(
                results,
                hide_index=True,
                column_config={
                    "retrieval_score": st.column_config.NumberColumn("Score", format="%.3f"),
                    "text": st.column_config.TextColumn("Passage", width="large"),
                },
            )

st.info("Un score de retrieval classe un passage ; il ne valide ni une réponse ni une preuve.")

st.subheader("2. Générer puis contrôler une preuve")
retrieval_choice = st.segmented_control("Contexte envoyé au générateur", ["Lexical", "Sémantique"], default="Lexical")
generation_mode = st.segmented_control("Générateur", ["Contrôlé", "LLM réel"], default="Contrôlé")

selected_results = lexical if retrieval_choice == "Lexical" else semantic

def selected_retriever(all_chunks, user_question, top_k):
    return selected_results


def controlled_extractor(user_question, context):
    first = selected_results[0]
    quote = first["text"].split(".")[0].strip()
    return ProposedEvidence(
        answer="Le premier passage récupéré fournit la preuve candidate.",
        supporting_quote=quote,
        paper_id=first["paper_id"],
        chunk_id=first["chunk_id"],
        page=first["page"],
    )


if st.button("Exécuter le pipeline RAG", type="primary", icon=":material/play_arrow:"):
    try:
        extractor = controlled_extractor
        if generation_mode == "LLM réel":
            extractor = build_llm_extractor(build_model(Settings()))
        result = run_rag(chunks, question, selected_retriever, extractor, k)
        st.session_state["rag_result"] = result
        save_evidence_artifact(result["evidence"], Path(__file__).parent / "artifacts" / "verified_evidence.json")
    except Exception as exc:
        st.error(f"Exécution impossible : {exc}")

if result := st.session_state.get("rag_result"):
    with st.container(border=True):
        st.markdown("### Trace du pipeline")
        st.write("**Retrieval → augmentation → génération → contrôle**")
        st.code(result["context"], language=None)
        evidence = result["evidence"]
        st.json(evidence.model_dump())
        st.caption("Artefact produit : artifacts/verified_evidence.json")
        if evidence.status == "verified":
            st.success(evidence.verification_reason)
        elif evidence.status == "human_review":
            st.warning(evidence.verification_reason)
        else:
            st.error(evidence.verification_reason)
