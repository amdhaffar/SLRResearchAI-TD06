import json
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

from slrresearch.artifacts import load_search_artifact
from slrresearch.config import Settings, build_model
from slrresearch.llm_screening_agent import compare_decisions, screen_with_llm
from slrresearch.screening_agent import ScreeningAgent


ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "search_results.json"
INCLUDE_TERMS = {"slr", "screening", "systematic review"}
EXCLUDE_TERMS = {"medicine", "clinical trial"}


def has_real_notice_url(url: str) -> bool:
    if not url:
        return False
    host = (urlparse(url).hostname or "").lower()
    return host not in {"example.org", "www.example.org"}


@st.dialog("Résumé complet du papier")
def show_paper_details(paper):
    st.subheader(paper.title)
    st.caption(
        f"Identifiant : {paper.id} · Source : {paper.source} · "
        f"DOI : {paper.doi or 'non disponible'}"
    )
    st.markdown("**Résumé intégral utilisé par les agents**")
    if paper.abstract.strip():
        with st.container(height=320, border=True):
            st.write(paper.abstract)
    else:
        st.warning("Aucun résumé n'a été transmis par le TD03.")
    if has_real_notice_url(paper.url):
        st.link_button(
            "Ouvrir la notice scientifique",
            paper.url,
            icon=":material/open_in_new:",
        )


st.set_page_config(
    page_title="SLRResearchAI · TD04",
    page_icon=":material/fact_check:",
    layout="wide",
)
st.session_state.setdefault("td04_result", None)
st.session_state.setdefault("td04_llm_results", None)
st.session_state.setdefault("td04_comparisons", None)
st.session_state.setdefault("td04_a_human_decisions", {})
st.session_state.setdefault("td04_human_decisions", {})

st.title("TD04 · Trois stratégies de screening")
st.caption(
    "Screening initial sur titre et résumé : include signifie « à évaluer sur "
    "texte intégral ». A et B fonctionnent indépendamment ; C contrôle leurs "
    "sorties automatiques avant l’arbitrage humain."
)

try:
    artifact = load_search_artifact(ARTIFACT_PATH)
except Exception as exc:
    st.error(f"Artefact TD03 invalide : {exc}")
    st.stop()

settings = Settings()
papers_by_id = {paper.id: paper for paper in artifact.papers}

with st.container(border=True):
    st.subheader("1. Entrée réelle du TD03")
    metrics = st.columns(4)
    metrics[0].metric("Origine", artifact.origin)
    metrics[1].metric("Articles uniques", len(artifact.papers))
    metrics[2].metric(
        "Avec résumé", sum(bool(paper.abstract.strip()) for paper in artifact.papers)
    )
    metrics[3].metric(
        "Sans résumé", sum(not paper.abstract.strip() for paper in artifact.papers)
    )
    if artifact.pedagogical_fixture:
        st.info(artifact.note)
    else:
        st.success("Artefact réel issu du SearchAgent ReAct contrôlé du TD03")
        st.caption(artifact.note)
    with st.expander("Voir l’artefact complet transmis par le TD03"):
        artifact_payload = artifact.model_dump(mode="json")
        st.caption(
            "Objet exact validé par Pydantic et consommé par le TD04 : "
            "provenance, bilan de recherche, requêtes et papiers."
        )
        with st.container(height=520):
            st.json(artifact_payload, expanded=2)
        st.download_button(
            "Télécharger l’artefact JSON",
            data=json.dumps(artifact_payload, ensure_ascii=False, indent=2),
            file_name="search_results.json",
            mime="application/json",
            icon=":material/download:",
            key="download_search_artifact",
        )

st.subheader("2. Corpus transmis")
st.dataframe(
    [
        {
            "id": paper.id,
            "source": paper.source,
            "titre": paper.title,
            "résumé disponible": bool(paper.abstract.strip()),
            "doi": paper.doi or "—",
        }
        for paper in artifact.papers
    ],
    hide_index=True,
    column_config={
        "titre": st.column_config.TextColumn("Titre", width="large"),
        "résumé disponible": st.column_config.CheckboxColumn("Résumé ?"),
    },
)

with st.container(border=True):
    st.subheader("3. Trois expériences")
    strategy_columns = st.columns(3)
    with strategy_columns[0]:
        st.markdown("**Version A — déterministe**")
        st.write("Mots-clés, règles explicites, coût faible et reproductibilité.")
    with strategy_columns[1]:
        st.markdown("**Version B — LLM**")
        st.write(f"Analyse sémantique indépendante avec `{settings.provider}`.")
    with strategy_columns[2]:
        st.markdown("**Version C — coopération**")
        st.write("Compare les sorties automatiques de A et B, puis route vers AUTO ou HUMAIN.")

with st.container(horizontal=True):
    run_a = st.button(
        "Lancer A indépendamment",
        icon=":material/rule:",
        type="primary",
    )
    run_b = st.button(
        "Lancer B indépendamment",
        icon=":material/psychology:",
        type="primary",
    )
    run_c = st.button(
        "Lancer C · coopération",
        icon=":material/hub:",
        disabled=(
            st.session_state.td04_result is None
            or st.session_state.td04_llm_results is None
        ),
    )

if run_a:
    try:
        st.session_state.td04_result = ScreeningAgent().run(
            artifact.papers, INCLUDE_TERMS, EXCLUDE_TERMS
        )
        st.session_state.td04_a_human_decisions = {}
        st.session_state.td04_comparisons = None
        st.rerun()
    except NotImplementedError:
        st.error("Complétez les TODO 1 à 6 dans slrresearch/screening_agent.py.")

if run_b:
    try:
        model = build_model(settings)
        llm_results = []
        with st.spinner(f"Version B : screening avec {settings.provider}…"):
            for paper in artifact.papers:
                llm_results.append(
                    screen_with_llm(
                        model,
                        paper,
                        artifact.topic,
                        INCLUDE_TERMS,
                        EXCLUDE_TERMS,
                    )
                )
        st.session_state.td04_llm_results = llm_results
        st.session_state.td04_comparisons = None
        st.rerun()
    except NotImplementedError:
        st.error("Complétez les TODO 7 à 10 dans slrresearch/llm_screening_agent.py.")
    except Exception as exc:
        st.error(f"Version B impossible : {exc}")

if run_c:
    deterministic_by_id = {
        item.paper_id: item
        for item in st.session_state.td04_result["decisions"]
    }
    llm_by_id = {
        item.paper_id: item for item in st.session_state.td04_llm_results
    }
    st.session_state.td04_comparisons = [
        compare_decisions(deterministic_by_id[paper.id], llm_by_id[paper.id])
        for paper in artifact.papers
    ]
    st.rerun()

result_a = st.session_state.td04_result
results_b = st.session_state.td04_llm_results

if result_a or results_b:
    st.subheader("4. Résultats indépendants côte à côte")
    summary_columns = st.columns(2)
    with summary_columns[0]:
        st.markdown("### Version A · règles")
        if result_a:
            st.metric("Décisions produites", len(result_a["decisions"]))
            st.metric("Demandes humaines", len(result_a["human_queue"]))
        else:
            st.info("Version A non exécutée. La version B reste consultable.")
    with summary_columns[1]:
        st.markdown(f"### Version B · LLM ({settings.provider})")
        if results_b:
            st.metric("Décisions produites", len(results_b))
            st.metric(
                "Demandes humaines",
                sum(item.needs_human_review for item in results_b),
            )
        else:
            st.info("Version B non exécutée. La version A reste consultable.")

    deterministic_by_id = {
        item.paper_id: item for item in (result_a or {}).get("decisions", [])
    }
    llm_by_id = {item.paper_id: item for item in (results_b or [])}

    for paper in artifact.papers:
        with st.container(border=True):
            st.markdown(f"**{paper.id} — {paper.title}**")
            left, right = st.columns(2)
            with left:
                st.caption("VERSION A · DÉTERMINISTE")
                if decision_a := deterministic_by_id.get(paper.id):
                    current_a = st.session_state.td04_a_human_decisions.get(
                        paper.id
                    )
                    if decision_a.needs_human_review:
                        st.write(
                            f"Recommandation automatique : **{decision_a.decision}**"
                        )
                        if current_a:
                            st.success(
                                f"Décision effective humaine : "
                                f"**{current_a['decision']}**"
                            )
                            st.caption(
                                f"Justification humaine : "
                                f"{current_a['justification']}"
                            )
                        else:
                            st.warning("Décision effective : EN ATTENTE HUMAINE")
                    else:
                        st.write(f"Décision automatique : **{decision_a.decision}**")
                        st.success("Décision effective : validée automatiquement")
                    st.write(f"Confiance de règle : **{decision_a.confidence:.2f}**")
                    st.write("Raisons :", " ; ".join(decision_a.reasons))
                    if decision_a.needs_human_review:
                        with st.form(f"a_human_review_{paper.id}", border=True):
                            human_decision_a = st.segmented_control(
                                "Votre décision finale",
                                ["include", "exclude", "uncertain"],
                                key=f"a_human_choice_{paper.id}",
                            )
                            justification_a = st.text_area(
                                "Justification",
                                key=f"a_human_reason_{paper.id}",
                            )
                            submitted_a = st.form_submit_button(
                                "Valider la décision humaine",
                                icon=":material/how_to_reg:",
                                type="primary",
                            )
                        if submitted_a:
                            if (
                                human_decision_a is None
                                or not justification_a.strip()
                            ):
                                st.error(
                                    "Choisissez une décision et justifiez-la."
                                )
                            else:
                                st.session_state.td04_a_human_decisions[
                                    paper.id
                                ] = {
                                    "decision": human_decision_a,
                                    "justification": justification_a.strip(),
                                }
                                st.rerun()
                else:
                    st.info("Exécutez A pour afficher ce résultat.")
            with right:
                st.caption(f"VERSION B · LLM ({settings.provider})")
                if decision_b := llm_by_id.get(paper.id):
                    st.write(f"Recommandation automatique : **{decision_b.decision}**")
                    st.write(f"Confiance déclarée : **{decision_b.confidence:.2f}**")
                    st.write("Raisons :", " ; ".join(decision_b.reasons))
                    st.write(f"Preuve : “{decision_b.evidence or 'aucune'}”")
                    if decision_b.needs_human_review:
                        st.warning("Décision effective : EN ATTENTE HUMAINE")
                    else:
                        st.success("Décision effective : validée automatiquement")
                else:
                    st.info("Exécutez B pour afficher ce résultat.")
            if st.button(
                "Lire le résumé intégral",
                key=f"details_compare_{paper.id}",
                icon=":material/article:",
            ):
                show_paper_details(paper)

comparisons = st.session_state.td04_comparisons
if result_a and results_b and not comparisons:
    st.info(
        "A et B sont disponibles. Comparez-les vous-même, puis lancez C pour "
        "observer la stratégie de coopération."
    )

if comparisons:
    st.subheader("5. Version C · coopération contrôlée")
    comparison_by_id = {item.paper_id: item for item in comparisons}
    deterministic_by_id = {
        item.paper_id: item for item in result_a["decisions"]
    }
    llm_by_id = {item.paper_id: item for item in results_b}
    disagreements = sum(not item.agreement for item in comparisons)
    with st.container(horizontal=True):
        st.metric("Accords", len(comparisons) - disagreements, border=True)
        st.metric("Désaccords", disagreements, border=True)
        st.metric(
            "Arbitrages humains",
            sum(item.needs_human_review for item in comparisons),
            border=True,
        )

    st.write(
        "C n'est pas un troisième classifieur. Il compare les recommandations "
        "automatiques originales de A et B, consulte leurs drapeaux de revue et "
        "transmet les cas sensibles à l'humain. Une décision humaine déjà saisie "
        "dans A reste une décision finale locale : elle ne remplace pas la sortie "
        "automatique utilisée pour cette expérience de comparaison."
    )
    for paper in artifact.papers:
        comparison = comparison_by_id[paper.id]
        with st.container(border=True):
            st.markdown(f"**{paper.id} — {paper.title}**")
            row = st.columns(3)
            row[0].metric("A · règles", comparison.deterministic_decision)
            row[1].metric("B · LLM", comparison.llm_decision)
            row[2].metric("Coopération", "HUMAN" if comparison.needs_human_review else "AUTO")
            if comparison.agreement:
                st.success(comparison.escalation_reason)
            else:
                st.warning(comparison.escalation_reason)

    st.subheader("6. Arbitrage humain produit par C")
    review_ids = {
        item.paper_id for item in comparisons if item.needs_human_review
    }
    for paper_id in sorted(review_ids):
        paper = papers_by_id[paper_id]
        decision_a = deterministic_by_id[paper_id]
        decision_b = llm_by_id[paper_id]
        comparison = comparison_by_id[paper_id]
        with st.container(border=True):
            st.markdown(f"### {paper_id} — {paper.title}")
            st.write(
                f"A propose **{decision_a.decision}** · "
                f"B propose **{decision_b.decision}**"
            )
            st.write(f"Motif du routage : {comparison.escalation_reason}")
            st.write("Raisons A :", " ; ".join(decision_a.reasons))
            st.write("Raisons B :", " ; ".join(decision_b.reasons))
            st.write(f"Preuve B : “{decision_b.evidence or 'aucune'}”")
            with st.container(horizontal=True):
                if st.button(
                    "Lire le résumé intégral",
                    key=f"details_hybrid_{paper_id}",
                    icon=":material/article:",
                ):
                    show_paper_details(paper)
                if has_real_notice_url(paper.url):
                    st.link_button(
                        "Ouvrir la notice",
                        paper.url,
                        icon=":material/open_in_new:",
                    )

            current = st.session_state.td04_human_decisions.get(paper_id)
            if current:
                st.success(
                    f"Décision humaine : {current['decision']} — "
                    f"{current['justification']}"
                )
            with st.form(f"human_review_{paper_id}", border=False):
                human_decision = st.segmented_control(
                    "Décision humaine",
                    ["include", "exclude", "uncertain"],
                    key=f"human_choice_{paper_id}",
                )
                justification = st.text_area(
                    "Justification fondée sur le titre et le résumé",
                    key=f"human_reason_{paper_id}",
                )
                submitted = st.form_submit_button(
                    "Enregistrer la décision",
                    icon=":material/save:",
                    type="primary",
                )
            if submitted:
                if human_decision is None or not justification.strip():
                    st.error("Une décision et une justification sont obligatoires.")
                else:
                    st.session_state.td04_human_decisions[paper_id] = {
                        "decision": human_decision,
                        "justification": justification.strip(),
                    }
                    st.rerun()

    with st.expander("Voir les trois sorties structurées"):
        st.json(
            {
                "version_a": [asdict(item) for item in result_a["decisions"]],
                "version_b": [item.model_dump() for item in results_b],
                "version_c": [item.model_dump() for item in comparisons],
            }
        )
