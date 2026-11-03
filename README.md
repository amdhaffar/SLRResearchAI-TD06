# SLRResearchAI — TD06 RAG agentique

Ce dépôt cumulatif transforme la baseline RAG fixe du TD05 en retrieval
adaptatif sous contraintes.

```text
décider → récupérer → observer → adapter ou terminer
```

Le package `slrresearch/` conserve les composants acquis pendant les TD01 à
TD05. Les huit nouveaux TODO sont uniquement dans :

```text
slrresearch/agentic_rag.py
```

## Démarrage

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
pytest -v
```

Après avoir terminé les TODO :

```bash
python3 main.py
python3 -m streamlit run streamlit_app.py
```

## Documents

- `ENONCE_TD06.md` : déroulement détaillé des huit TODO ;
- `CONSTRUIRE_PROJET_BOUT_EN_BOUT.md` : chaîne complète TD01 → TD06 ;
- `ACQUIS_TD05.md` : rappel de l'énoncé précédent ;
- `GUIDE_RETRIEVAL_LEXICAL.md` : rappel sur le retrieval lexical.

Le mode contrôlé est reproductible et ne nécessite aucun service externe. Les
modes Ollama ou Groq utilisent `.env`, créé localement à partir de
`.env.example` et ignoré par Git.
