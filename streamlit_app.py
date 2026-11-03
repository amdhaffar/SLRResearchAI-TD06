from pathlib import Path

import streamlit as st

from slrresearch.agentic_rag import (
    build_llm_decider,
    compare_baseline_and_agentic,
    controlled_decider,
    finalize_agentic_evidence,
    run_adaptive_retrieval,
)
from slrresearch.agentic_rag_artifacts import load_agentic_cases, save_comparison_artifact
from slrresearch.agentic_rag_runtime import (
    ProtocolEmbeddings,
    build_retrieval_tools,
    controlled_extractor,
)
from slrresearch.config import Settings, build_embeddings, build_model
from slrresearch.rag_artifacts import load_rag_handoff
from slrresearch.rag_pipeline import build_llm_extractor, chunk_pages


st.set_page_config(page_title="TD06 RAG agentique", page_icon=":material/route:", layout="wide")
st.title("TD06 — Du RAG fixe au retrieval adaptatif")
st.caption("Même corpus et mêmes annotations : seule la politique de retrieval change.")

ROOT = Path(__file__).parent


@st.cache_data
def load_protocol():
    documents, lineage = load_rag_handoff(ROOT / "artifacts")
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_pages(document.paper_id, document.pages, size=1000, overlap=0)
    ]
    cases = load_agentic_cases(ROOT / "artifacts" / "agentic_rag_cases.json")
    return chunks, lineage, cases


chunks, lineage, cases = load_protocol()
case_by_id = {case.id: case for case in cases}

st.subheader("Continuité du projet")
with st.container(horizontal=True):
    st.metric("TD03 · articles", lineage["found"])
    st.metric("TD04 · inclus", lineage["included"])
    st.metric("TD05 · corpus RAG", lineage["rag_documents"])
    st.metric("TD06 · scénarios", len(cases))
st.caption(
    "Le TD06 réutilise les chunks, les deux retrievers, les métriques, la génération "
    "structurée et le contrôle des preuves du TD05."
)

with st.expander("Comprendre les 8 TODO avant l'exécution", expanded=True):
    st.dataframe(
        [
            {"TODO": "1", "Composant": "Observer", "Question": "Quelles facettes sont couvertes ou manquantes ?"},
            {"TODO": "2", "Composant": "Décider", "Question": "Faut-il rechercher, terminer ou s'abstenir ?"},
            {"TODO": "3", "Composant": "Agir", "Question": "Quel outil autorisé faut-il appeler ?"},
            {"TODO": "4", "Composant": "Mémoriser", "Question": "Comment fusionner les chunks sans doublons ?"},
            {"TODO": "5", "Composant": "Mettre à jour", "Question": "Que faut-il écrire dans l'état et la trace ?"},
            {"TODO": "6", "Composant": "Boucler", "Question": "Quand recommencer et quand s'arrêter ?"},
            {"TODO": "7", "Composant": "Évaluer", "Question": "Le gain de rappel justifie-t-il les appels supplémentaires ?"},
            {"TODO": "8", "Composant": "Décideur LLM", "Question": "Comment obtenir une action structurée et contrôlée ?"},
        ],
        hide_index=True,
    )
    st.caption(
        "Les TODO 1 à 6 construisent la boucle agentique. Le TODO 7 la compare "
        "au RAG fixe. Le TODO 8 remplace seulement la politique de décision."
    )

st.subheader("Protocole expérimental commun")
case_id = st.selectbox(
    "Question du TD05 réutilisée",
    list(case_by_id),
    format_func=lambda key: f"{key} — {case_by_id[key].label}",
)
current_case = case_by_id[case_id]
with st.container(border=True):
    st.write(f"**Question :** {current_case.question}")
    st.write(f"**Ce que ce cas permet d'observer :** {current_case.pedagogical_goal}")
    st.caption(f"Comportement attendu : {current_case.expected_behavior}")
    if case_id in {"Q1", "Q2"}:
        st.info(
            "Scénario de gain attendu : avec les modes contrôlés et un budget "
            "de deux cycles, le TD06 doit retrouver une preuve manquée par le TD05."
        )
    else:
        st.info(
            "Scénario témoin : la baseline suffit ; l'agent doit le reconnaître "
            "et éviter une recherche supplémentaire."
        )

with st.form("agentic_protocol", border=True):
    st.markdown("#### Paramètres de l'expérience")
    with st.container(horizontal=True):
        planner_mode = st.segmented_control(
            "Décideur",
            ["Contrôlé", "LLM réel"],
            default="Contrôlé",
        )
        embedding_mode = st.segmented_control(
            "Embeddings",
            ["Contrôlés", "Ollama réel"],
            default="Contrôlés",
        )
        generation_mode = st.segmented_control(
            "Générateur final",
            ["Contrôlé", "LLM réel"],
            default="Contrôlé",
        )
    with st.container(horizontal=True):
        k = st.slider("Top-k par cycle", 1, 3, 1)
        max_iterations = st.slider("Budget maximal d'itérations", 1, 3, 2)
    st.caption(
        "Pour observer une abstention contrôlée sans changer de question, "
        "choisissez Q2 ou Q3 et limitez le budget à une itération."
    )
    st.info(
        "Pour isoler l'apport du décideur LLM : choisissez Q1 ou Q2, "
        "Décideur = LLM réel, Embeddings = Contrôlés, Générateur = Contrôlé, "
        "puis gardez un budget de 2. Activer tous les composants réels mélange "
        "la qualité de la décision, des embeddings et de la génération."
    )
    submitted = st.form_submit_button(
        "Comparer TD05 et TD06", type="primary", icon=":material/play_arrow:"
    )

case = current_case
signature = (case_id, planner_mode, embedding_mode, generation_mode, k, max_iterations)

if submitted:
    try:
        embeddings = (
            build_embeddings(Settings())
            if embedding_mode == "Ollama réel"
            else ProtocolEmbeddings(chunks, set(case.relevant_chunk_ids))
        )
        tools = build_retrieval_tools(embeddings)
        baseline = tools[case.baseline_tool](chunks, case.question, case.baseline_k)

        decider = controlled_decider
        if planner_mode == "LLM réel":
            llm_decider = build_llm_decider(build_model(Settings()))

            def guarded_decider(state):
                # Comparaison équitable : TD05 et TD06 commencent par le même
                # retrieval lexical. Le LLM intervient seulement après avoir
                # observé cette première recherche.
                if state.iteration == 0:
                    return controlled_decider(state)
                if not state.missing_facets:
                    return controlled_decider(state)
                try:
                    action = llm_decider(state)
                except Exception:
                    fallback = controlled_decider(state)
                    return fallback.model_copy(
                        update={
                            "reason": (
                                "Fallback contrôlé : la décision LLM ne respecte "
                                "pas le contrat RetrievalAction."
                            )
                        }
                    )
                if action.action == "finish" and state.missing_facets:
                    fallback = controlled_decider(state)
                    return fallback.model_copy(
                        update={
                            "reason": (
                                "Fallback contrôlé : le LLM voulait terminer "
                                "alors que des facettes manquent."
                            )
                        }
                    )
                if action.action == "retrieve" and action.tool != "semantic":
                    fallback = controlled_decider(state)
                    return fallback.model_copy(
                        update={
                            "reason": (
                                "Fallback contrôlé : après l'échec lexical, le "
                                "LLM devait choisir le retriever sémantique."
                            )
                        }
                    )
                return action.model_copy(
                    update={"reason": f"Décision LLM structurée : {action.reason}"}
                )

            decider = guarded_decider

        state = run_adaptive_retrieval(
            case.question,
            case.facets,
            chunks,
            tools,
            decider=decider,
            k=k,
            max_iterations=max_iterations,
        )
        report = compare_baseline_and_agentic(case, baseline, state)

        extractor = controlled_extractor(state)
        if generation_mode == "LLM réel":
            extractor = build_llm_extractor(build_model(Settings()))
        evidence = finalize_agentic_evidence(state, extractor)

        trace = [event.model_dump() for event in state.trace]
        artifact_error = None
        try:
            save_comparison_artifact(
                report,
                trace,
                ROOT / "artifacts" / "agentic_rag_result.json",
            )
        except OSError as exc:
            artifact_error = str(exc)
        st.session_state["td06_result"] = {
            "signature": signature,
            "baseline": baseline,
            "state": state,
            "report": report,
            "evidence": evidence,
            "artifact_error": artifact_error,
        }
    except Exception as exc:
        st.error(f"Exécution impossible : {exc}")

result = st.session_state.get("td06_result")
if not result or result["signature"] != signature:
    st.info(
        "Lancez le protocole pour observer la baseline fixe, les décisions de "
        "l'agent, les changements d'outil et la condition d'arrêt."
    )
    st.stop()

report = result["report"]
state = result["state"]

st.subheader("Lecture guidée : où apparaissent les TODO ?")
terminal_action = "finish" if state.status == "completed" else "abstain"
actions = " → ".join([*(event.action for event in state.trace), terminal_action])
tools_used = " → ".join(event.tool or "—" for event in state.trace) or "aucun outil"
llm_step = (
    "appelé après la première observation ; sa décision apparaît dans la trace"
    if planner_mode == "LLM réel" and state.retrieval_calls > 1
    else "non sollicité : règle contrôlée ou arrêt immédiat"
)
st.dataframe(
    [
        {"TODO": "1", "Ce que montre cette exécution": f"Couvertes : {state.covered_facets or 'aucune'} ; manquantes : {state.missing_facets or 'aucune'}"},
        {"TODO": "2", "Ce que montre cette exécution": f"Décisions observées : {actions}"},
        {"TODO": "3", "Ce que montre cette exécution": f"Outils appelés : {tools_used}"},
        {"TODO": "4", "Ce que montre cette exécution": f"Mémoire finale : {len(state.selected_chunks)} chunk(s) unique(s)"},
        {"TODO": "5", "Ce que montre cette exécution": f"État : {state.iteration} itération(s), {len(state.trace)} événement(s) tracé(s)"},
        {"TODO": "6", "Ce que montre cette exécution": f"Condition d'arrêt : {state.status}"},
        {"TODO": "7", "Ce que montre cette exécution": f"Δ Recall@k = {report.delta_recall:+.2f} ; appels supplémentaires = {report.extra_retrieval_calls}"},
        {"TODO": "8", "Ce que montre cette exécution": llm_step},
    ],
    hide_index=True,
)

st.subheader("1. Même question, deux politiques")
baseline_col, agent_col = st.columns(2)
with baseline_col:
    with st.container(border=True):
        st.markdown("### TD05 — Baseline fixe")
        st.write(f"**Outil :** {case.baseline_tool} · **Top-k :** {case.baseline_k}")
        with st.container(horizontal=True):
            st.metric("Precision@k", f'{report.baseline["precision_at_k"]:.2f}')
            st.metric("Recall@k", f'{report.baseline["recall_at_k"]:.2f}')
            st.metric("Appels retrieval", report.baseline["retrieval_calls"])
        st.write("**Facettes couvertes :**", report.baseline["covered_facets"] or "aucune")
        st.write("**Facettes manquantes :**", report.baseline["missing_facets"] or "aucune")
        st.dataframe(result["baseline"], hide_index=True)

with agent_col:
    with st.container(border=True):
        st.markdown("### TD06 — Variante agentique")
        st.caption(
            "Même premier retrieval lexical que le TD05 ; adaptation uniquement "
            "si l'observation révèle des facettes manquantes."
        )
        st.write(f"**Statut :** `{state.status}` · **Budget :** {state.max_iterations}")
        with st.container(horizontal=True):
            st.metric("Precision@k", f'{report.agentic["precision_at_k"]:.2f}')
            st.metric("Recall@k", f'{report.agentic["recall_at_k"]:.2f}')
            st.metric("Appels retrieval", report.agentic["retrieval_calls"])
        st.write("**Facettes couvertes :**", state.covered_facets or "aucune")
        st.write("**Facettes manquantes :**", state.missing_facets or "aucune")
        st.dataframe(state.selected_chunks, hide_index=True)

st.subheader("Verdict de la comparaison")
if report.delta_recall > 0:
    st.success(
        f"Gain démontré : Recall@k +{report.delta_recall:.2f}. "
        "Le RAG agentique a retrouvé une preuve que la baseline fixe avait manquée."
    )
elif report.delta_recall == 0 and report.agentic["recall_at_k"] == 1.0:
    st.info(
        "Qualité identique : la baseline couvrait déjà la question. "
        "Le bon comportement agentique consiste à s'arrêter sans action inutile."
    )
else:
    st.warning(
        f"Aucun gain observé (Δ Recall@k = {report.delta_recall:.2f}). "
        "Vérifiez le budget et les modes sélectionnés, ou analysez la trace."
    )

st.subheader("2. Trace de la boucle décisionnelle")
trace_rows = [
    {
        "cycle": event.iteration,
        "TODO mobilisés": "2 → 3 → 1 → 4 → 5 → 6",
        "outil": event.tool,
        "requête": event.query,
        "résultats": event.found,
        "couvertes": ", ".join(event.covered_facets) or "—",
        "manquantes": ", ".join(event.missing_facets) or "—",
        "raison": event.reason,
    }
    for event in state.trace
]
st.dataframe(trace_rows, hide_index=True)

if state.status == "completed":
    st.success("Arrêt : les facettes attendues sont couvertes.")
elif state.status == "budget_exhausted":
    st.warning("Arrêt : budget épuisé. Le système s'abstient au lieu d'inventer une preuve.")
else:
    st.warning(f"Arrêt : {state.status}")

st.subheader("3. Preuve finale contrôlée")
evidence = result["evidence"]
if evidence is None:
    st.info("Aucune génération : les preuves sont insuffisantes ou le budget est épuisé.")
else:
    st.json(evidence.model_dump())
    if evidence.status == "verified":
        st.success(evidence.verification_reason)
    elif evidence.status == "human_review":
        st.warning(evidence.verification_reason)
    else:
        st.error(evidence.verification_reason)

st.caption("Artefact produit : artifacts/agentic_rag_result.json")
if result["artifact_error"]:
    st.warning(f"Résultats calculés, mais artefact non sauvegardé : {result['artifact_error']}")
