from slrresearch.real_tools import build_real_registry, semantic_scholar_search


def test_todo_3_registre_contient_trois_vrais_outils():
    registry = build_real_registry()
    assert set(registry.names) == {"openalex", "crossref", "semantic_scholar"}
    assert all(callable(registry.authorize(name).search) for name in registry.names)


def test_todo_3_semantic_scholar_configure_et_normalise(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {
                        "paperId": "P1",
                        "title": "Article réel",
                        "abstract": "Résumé",
                        "externalIds": {"DOI": "10.1/demo"},
                        "url": "https://example.org/P1",
                        "openAccessPdf": {"url": "https://example.org/P1.pdf"},
                    }
                ]
            }

    def fake_get(url, params, timeout):
        captured.update(url=url, params=params, timeout=timeout)
        return Response()

    monkeypatch.setattr("slrresearch.real_tools.httpx.get", fake_get)
    papers = semantic_scholar_search("LLM screening", 4)
    assert "semanticscholar.org" in captured["url"]
    assert captured["params"]["query"] == "LLM screening"
    assert captured["params"]["limit"] == 4
    assert papers[0]["doi"] == "10.1/demo"
    assert papers[0]["open_access"] is True
