# Construire SLRResearchAI de bout en bout jusqu'au TD06

## 1. Un projet cumulatif

Le TD06 n'est pas un nouveau mini-projet. Il ajoute une capacité au même package
Python :

```text
slrresearch/
├── research_questions.py, search_plan.py       # TD01–TD02
├── tools.py, real_tools.py, react_agent.py      # TD03
├── artifacts.py, screening_agent.py             # TD04
├── llm_screening_agent.py                       # TD04
├── rag_models.py, rag_artifacts.py               # TD05
├── rag_pipeline.py, pdf_loader.py                # TD05
├── agentic_rag_models.py                         # TD06
├── agentic_rag_artifacts.py                      # TD06
├── agentic_rag_runtime.py                        # TD06
└── agentic_rag.py                                # TD06
```

Les modules assurent la continuité du code. Les artefacts assurent la continuité
des données entre des exécutions indépendantes.

## 2. Contrats entre les étapes

| Producteur | Artefact | Consommateur |
|---|---|---|
| TD02 | `search_plan.json` | TD03 SearchAgent |
| TD03 | `search_results.json` | TD04 ScreeningAgent |
| TD04 | `screening_decisions.json` | préparation documentaire |
| Préparation | `document_manifest.json`, `extracted_pages.json` | TD05 |
| TD05 | chunks, métriques et preuves contrôlées | TD06 |
| Protocole TD06 | `agentic_rag_cases.json` | agent adaptatif |
| TD06 | `agentic_rag_result.json` | futur module de synthèse |

`paper_id` et `chunk_id` ne doivent jamais être recréés arbitrairement entre
deux étapes.

## 3. Pourquoi les artefacts sont préremplis

Pendant le TD, les artefacts sont fournis afin d'isoler le concept étudié. Les
étudiants peuvent donc travailler sur la boucle agentique sans dépendre :

- d'une API scientifique disponible ;
- d'un téléchargement de PDF ;
- d'une décision humaine précédente ;
- d'un LLM distant stable.

Dans le projet final, les fichiers préremplis sont remplacés par les sorties
réelles des composants déjà présents dans le dépôt.

## 4. Passer du mode autonome au projet intégré

### Étape A — Produire les résultats de recherche

Le TD03 doit écrire `artifacts/search_results.json` avec des identifiants stables,
les titres, résumés, DOI, sources et URL.

### Étape B — Produire les décisions de screening

Le TD04 consomme les articles du TD03 et écrit
`artifacts/screening_decisions.json`. Les conflits doivent être arbitrés avant
l'export de la décision finale.

### Étape C — Préparer les documents

Les PDF autorisés sont extraits page par page. Le manifeste conserve le statut,
l'URL et la clé qui relie l'article aux pages extraites.

### Étape D — Construire le RAG classique

Le TD05 transforme les pages en chunks, exécute le retrieval, construit le
contexte, demande une preuve structurée et vérifie la citation.

### Étape E — Ajouter l'adaptation

Le TD06 conserve les mêmes chunks et les mêmes outils. Il ajoute :

```text
état partagé
+ décision
+ observation de la couverture
+ nouvelle action conditionnelle
+ budget
+ trace
```

Le changement essentiel est que le deuxième retrieval n'est pas exécuté par
défaut. Il est déclenché uniquement si l'observation du premier cycle montre
une couverture insuffisante.

## 5. Deux modes du TD06

### Mode contrôlé

- décisions déterministes ;
- embeddings pédagogiques ;
- génération reproductible ;
- adapté aux tests et à l'explication en classe.

### Mode réel

- décision structurée par Ollama ou Groq ;
- embeddings Ollama ;
- génération structurée réelle ;
- résultats potentiellement variables.

Les deux modes partagent les mêmes contrats, outils autorisés, budgets et
contrôles de preuves.

## 6. Vérifications avant intégration

```bash
pytest -v
python3 main.py
python3 -m streamlit run streamlit_app.py
```

Vérifiez ensuite :

1. que tous les tests des acquis passent ;
2. que Q1 ou Q2 peut déclencher un second cycle lorsque la couverture manque ;
3. que Q3 s'arrête après un appel lorsque toutes les stratégies sont couvertes ;
4. qu'un budget d'une itération produit une abstention contrôlée si nécessaire ;
5. que chaque appel apparaît dans la trace ;
6. que l'artefact final contient le rapport comparatif et la trace.

## 7. Artefact de sortie du TD06

`artifacts/agentic_rag_result.json` contient :

- les métriques de la baseline ;
- les métriques de la variante agentique ;
- le gain ou la perte de rappel ;
- le nombre d'appels supplémentaires ;
- la trace des décisions, outils, requêtes et observations.

Ce fichier permettra à un prochain composant de distinguer une réponse étayée,
une preuve à revoir et une abstention justifiée.
