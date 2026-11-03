from slrresearch.research_questions import build_zero_shot_messages
from slrresearch.search_plan import build_question_pipeline


def test_acquis_td01_research_questions_est_conserve():
    topic = "LLM et screening des SLR"
    messages = build_zero_shot_messages(topic)
    assert len(messages) == 2
    assert topic in str(messages[1].content)


def test_acquis_td02_search_plan_est_conserve_et_executable():
    text = (
        "RQ1. Comment les LLM améliorent-ils le rappel du screening ?\n"
        "RQ2. Quel contrôle humain doit-il être conservé pendant le screening ?\n"
        "RQ3. Quelles métriques évaluent-elles la qualité de ce screening ?"
    )
    queries = build_question_pipeline().invoke(text)
    assert set(queries) == {"scholar", "ieee", "acm"}
    assert all(query.strip() for query in queries.values())
