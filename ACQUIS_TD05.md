# TD05 RAG — Construire, comparer et vérifier un pipeline RAG classique

## 1. Objectif

Ce TD matérialise la première partie du chapitre RAG. Vous construirez les deux
phases d'un RAG classique : préparer un corpus traçable, puis retrouver les
passages utiles avant de demander au LLM une preuve structurée.

```text
documents → chunks + provenance → retrieval lexical / sémantique
          → Top-k → contexte augmenté → LLM → contrôle de la preuve
```

Le TD ne construit pas encore un RAG agentique : l'ordre des étapes reste fixé
par le programme.

## 2. Continuité avec les TD précédents

Le TD05 est une **extension du même projet Python**, et non un projet RAG
séparé. Le package `slrresearch/` conserve donc le code développé depuis le
TD01. Vous ne devez pas réécrire ces acquis : vous ajoutez la nouvelle capacité
RAG et vous vérifiez que les anciennes fonctionnalités fonctionnent toujours.

| Acquis | Fichiers conservés dans le projet |
|---|---|
| TD01–TD02 : questions et plan de recherche | `research_questions.py`, `search_plan.py` |
| TD03 : outils et agent de recherche | `tools.py`, `real_tools.py`, `react_agent.py`, `workflow.py`, `contracts.py`, `policies.py`, `runtime.py` |
| TD04 : artefact et contrôle du screening | `artifacts.py`, `screening_agent.py`, `llm_screening_agent.py` |
| TD05 : nouvelle capacité RAG | `rag_models.py`, `rag_artifacts.py`, `rag_pipeline.py`, `pdf_loader.py` |

Les anciens points d'entrée sont également conservés sous les noms
`td04_main.py` et `td04_streamlit_app.py`. Les tests du dossier
`tests/acquis_td04/` jouent le rôle de **tests de non-régression** : ils
garantissent que l'ajout du RAG ne casse pas les TD précédents.

Le code demeure dans le même dépôt, tandis que les artefacts transmettent les
données d'une étape à l'autre. Ce TD ne relance pas obligatoirement la recherche
ni le screening : il peut consommer la chaîne suivante :

```text
TD03 · SearchAgent
artifacts/search_results.json
        ↓ jointure par paper_id
TD04 · ScreeningAgent + arbitrage humain
artifacts/screening_decisions.json
        ↓ conserver final_decision = include
Préparation documentaire
artifacts/document_manifest.json + extracted_pages.json
        ↓ conserver status = downloaded
TD05 RAG
chunks → Top-k → preuve contrôlée
        ↓
artifacts/verified_evidence.json
        ↓
prochain TD de synthèse scientifique
```

Les artefacts sont préremplis pour que la séance reste autonome. Dans le projet
intégré, les mêmes fichiers seront produits automatiquement par les modules
déjà présents dans `slrresearch/`. L'artefact n'est donc pas un remplacement du
code précédent : c'est son contrat de sortie persistant.

La fonction fournie `load_rag_handoff()` réalise une jointure contrôlée par
`paper_id`. Un article n'entre dans le RAG que s'il est simultanément :

1. présent dans les résultats du TD03 ;
2. définitivement inclus au TD04 ;
3. associé à un document disponible et extractible.

## 3. Fichiers

- `slrresearch/` : package cumulatif contenant les acquis TD01 à TD05 ;
- `slrresearch/rag_models.py` : contrats Pydantic fournis ;
- `slrresearch/pdf_loader.py` : extraction page par page fournie avec `pypdf` ;
- `slrresearch/rag_artifacts.py` : contrats et jointure entre les TD ;
- `slrresearch/rag_pipeline.py` : seul fichier contenant les TODO ;
- `artifacts/` : entrées préremplies et future sortie du RAG ;
- `tests/test_td05.py` : un test indépendant par TODO ;
- `tests/acquis_td04/` : tests de non-régression des TD antérieurs ;
- `streamlit_app.py` : comparaison visuelle des deux retrievers.

Le corpus expérimental est volontairement déjà extrait dans
`artifacts/extracted_pages.json` :
cela permet de conserver des questions et des passages pertinents annotés pour
calculer les métriques. Dans le projet complet, `extract_pdf_pages()` alimente
la même fonction `chunk_pages()` à partir des PDF autorisés. Un PDF scanné exige
un OCR, qui reste hors du périmètre de ce premier TD.

## 4. Installation et commandes

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest -v
```

Test ciblé :

```bash
pytest -v -k "todo_3_"
```

## 5. TODO 1 — Valider le découpage

**Fichier :** `slrresearch/rag_pipeline.py` — **fonction :** `validate_chunking()`

Complétez uniquement les deux `______` :

```python
if ______:
    raise ValueError(...)
return ______
```

La condition doit refuser `size <= 0`, `overlap < 0` ou `overlap >= size`.
Le pas retourné vaut `size - overlap`.

## 6. TODO 2 — Créer un chunk traçable

**Fonction :** `chunk_pages()`

Les boucles sont fournies. Remplacez seulement `chunk = None` par :

```python
Chunk(
    paper_id=paper_id,
    chunk_id=f"{paper_id}_p{page_number}_c{chunk_number}",
    page=page_number,
    text=text,
)
```

## 7. TODO 3 — Garder le Top-k lexical

**Fonction :** `lexical_retrieve()`

Le calcul TF-IDF est fourni. Remplacez uniquement le `return ______` pour :

1. trier `scored` selon `retrieval_score` ;
2. utiliser l’ordre décroissant ;
3. conserver les `k` premiers résultats.

**Indice — méthode `sorted()` :**

Pour consulter son manuel directement depuis le terminal :

```bash
python3 -c "help(sorted)"
```

```python
sorted(
    liste_a_trier,
    key=lambda element: element["nom_du_score"],
    reverse=True,
)[:nombre_a_conserver]
```

- `key` indique la valeur utilisée pour comparer deux éléments ;
- `reverse=True` place le score le plus élevé en premier ;
- `[:k]` conserve uniquement les `k` premiers éléments du classement.

Adaptez `liste_a_trier`, `nom_du_score` et `nombre_a_conserver` aux variables
déjà présentes dans la fonction. Vous ne devez pas recalculer les scores.

## 8. TODO 4 — Terminer la similarité cosinus

**Fonction :** `cosine_similarity()`

Le produit scalaire et les normes sont fournis. Complétez seulement le
`return ______` :

- si les deux normes sont non nulles, retourner
  `dot / (norm_left * norm_right)` ;
- sinon retourner `0.0`.

## 9. TODO 5 — Garder le Top-k sémantique

**Fonction :** `semantic_retrieve()`

L’encodage de la question et des chunks est fourni. Complétez uniquement le
`return ______` avec le même tri que dans le TODO 3.

## 10. TODO 6 — Calculer les deux métriques

**Fonction :** `retrieval_metrics()`

`hits` est déjà calculé. Remplacez les deux `None` :

```python
"precision_at_k": hits / len(returned_ids)
"recall_at_k": hits / len(relevant_ids) if relevant_ids else 0.0
```

## 11. TODO 7 — Assembler le contexte

**Fonction :** `build_context()`

Les blocs contenant `paper_id`, `chunk_id`, `page` et `text` sont déjà
construits. Remplacez `return ______` pour les joindre avec `"\n\n"`.

## 12. TODO 8 — Appeler le LLM structuré

**Fonction :** `build_llm_extractor()`

Le prompt et le modèle structuré sont fournis. Remplacez seulement :

```python
return ...
```

par l’appel du modèle structuré avec `prompt`.

## 13. TODO 9 — Retrouver le chunk annoncé

**Fonction :** `verify_evidence()`

La politique `verified / human_review / rejected` est fournie dans
`_assess_proposal()`. Remplacez seulement `chunk = None` par une recherche
dans `selected_chunks` :

```python
chunk = next(
    (item for item in selected_chunks
     if item["chunk_id"] == proposal.chunk_id),
    None,
)
```

## 14. TODO 10 — Terminer l’orchestration

**Fonction :** `run_rag()`

Les étapes retrieval, contexte et extraction sont fournies. Remplacez uniquement
`evidence = None` par l’appel qui contrôle `proposal` à partir de
`selected`.

## 15. Observer les résultats

```bash
python -m streamlit run streamlit_app.py
```

L'interface impose un protocole équitable : même corpus, même question et même
`k`. Elle affiche les Top-k, `Precision@k`, `Recall@k`, le contexte transmis et
la preuve contrôlée.

Après l'exécution, l'interface écrit `artifacts/verified_evidence.json`. Ce
fichier ne contient pas une réponse libre isolée : il conserve la citation, le
papier, le chunk, la page et le statut nécessaires au TD de rédaction.

Le mode contrôlé est reproductible. Le mode Ollama utilise réellement
`nomic-embed-text`. Le générateur peut ensuite être contrôlé ou utiliser le LLM
défini dans `.env`.
