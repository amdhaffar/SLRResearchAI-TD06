# TD06 — Du RAG classique au retrieval adaptatif

## 1. Objectif pédagogique

Le TD05 exécutait toujours une stratégie déterminée avant de voir les résultats :

```text
question → retriever fixé → Top-k fixé → génération → vérification
```

Dans ce TD, la recherche suivante dépend de l'observation précédente :

```text
question
→ décider
→ appeler un retriever autorisé
→ observer les facettes couvertes
→ terminer, adapter la recherche ou s'abstenir
```

Vous construirez une boucle agentique **bornée et traçable**. L'objectif n'est
pas de multiplier les appels, mais d'adapter la recherche uniquement lorsqu'un
diagnostic le justifie.

À la fin du TD, vous saurez :

- distinguer RAG fixe et RAG agentique ;
- représenter une décision comme une donnée structurée ;
- appeler un outil de retrieval autorisé ;
- conserver un état partagé entre plusieurs cycles ;
- évaluer la couverture d'une question multi-aspects ;
- imposer un budget et une condition d'arrêt ;
- tracer chaque décision et chaque observation ;
- comparer le gain de qualité au coût de l'adaptation.

## 2. Continuité avec les TD précédents

Le projet reste cumulatif. Le package `slrresearch/` contient toujours les
fichiers Python acquis lors des TD01 à TD05.

```text
TD01–TD02  questions de recherche et plan structuré
TD03       agent de recherche et outils scientifiques
TD04       screening déterministe, LLM et arbitrage humain
TD05       chunks, retrievers, métriques et preuves contrôlées
TD06       décision adaptative, état, boucle, budget et traces
```

Le TD06 réutilise directement du TD05 :

- `Chunk` et les métadonnées de provenance ;
- `lexical_retrieve()` et `semantic_retrieve()` ;
- `retrieval_metrics()` ;
- `build_context()` et `build_llm_extractor()` ;
- `verify_evidence()`.

Les tests de `tests/acquis_td04/` et `tests/acquis_td05/` sont des tests de
non-régression. Vous ne devez pas modifier les anciens modules pour réaliser les
TODO du TD06.

## 3. Chaîne d'artefacts

```text
TD03 · search_results.json
          ↓
TD04 · screening_decisions.json
          ↓
Préparation · document_manifest.json + extracted_pages.json
          ↓
TD05 · corpus RAG + protocole rag_evaluation.json
          ↓
TD06 · agentic_rag_cases.json
          ↓
       agentic_rag_result.json
```

`agentic_rag_cases.json` reprend exactement les trois questions du TD05 :

1. **Q1 — niveaux de rappel** : la question française vise une preuve numérique
   anglaise ; l'agent peut cibler les valeurs manquantes avec le sémantique ;
2. **Q2 — supervision humaine** : la question française et la preuve anglaise
   illustrent un écart de vocabulaire ; l'agent peut reformuler et utiliser le
   retrieval sémantique ;
3. **Q3 — stratégies de prompting** : plusieurs éléments sont attendus ;
   l'agent vérifie leur couverture avant de répondre.

Pour observer l'abstention sans inventer une quatrième question, limitez le
budget à une itération sur Q2 ou Q3.

Les annotations servent à l'évaluation expérimentale. Elles ne sont pas
transmises au générateur comme une réponse.

## 4. Architecture du TD06

```text
agentic_rag_models.py       contrats Pydantic fournis
agentic_rag_artifacts.py    lecture et écriture des artefacts
agentic_rag_runtime.py      outils contrôlés et adaptateurs fournis
agentic_rag.py              les 8 TODO du TD06
main.py                     exécution des trois scénarios
streamlit_app.py            comparaison visuelle TD05 / TD06
```

Tous les TODO sont concentrés dans :

```text
slrresearch/agentic_rag.py
```

## 5. Installation et exécution

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
pytest -v
```

Test d'un seul TODO :

```bash
pytest -v -k "todo_1"
```

Après avoir terminé les TODO :

```bash
python3 main.py
python3 -m streamlit run streamlit_app.py
```

Les tests et le mode contrôlé ne nécessitent ni Internet, ni Ollama, ni clé
Groq. Les modes réels de l'interface utilisent la configuration de `.env`.

---

## 6. TODO 1 — Évaluer la couverture des facettes

**Fichier :** `slrresearch/agentic_rag.py`  
**Fonction :** `evaluate_facets()`

Une question peut exiger plusieurs informations appelées **facettes**. Par
exemple, Q3 possède les facettes `input_output`, `chain_of_thought` et
`majority_voting`. Une facette est couverte si au moins un de ses mots-clés
apparaît dans les passages mémorisés.

Le texte global est déjà construit :

```python
corpus = " ".join(item["text"] for item in candidates).casefold()
```

La boucle et le test des mots-clés sont déjà écrits dans le fichier Python :

```python
for facet in facets:
    is_covered = any(
        keyword.casefold() in corpus for keyword in facet.keywords
    )
    destination = covered if is_covered else missing
    destination.append(None)
```

**Une seule modification est demandée :** remplacez `None` par l'attribut qui
contient l'identifiant de la facette courante.

```python
destination.append(facet.________)
```

Pour trouver le nom de cet attribut, ouvrez
`slrresearch/agentic_rag_models.py`, puis consultez la classe
`EvidenceFacet`. Vous y trouverez notamment les champs `id`, `label`,
`keywords` et `query_hint`.

Ne mettez pas `facet` dans la liste : `covered_facets` et `missing_facets`
attendent des chaînes de caractères contenant les identifiants.

Résultat attendu :

```python
covered == ["recall_values"]
missing == []
```

Test :

```bash
pytest -v -k "todo_1"
```

## 7. TODO 2 — Prendre une décision à partir de l'état

**Fonction :** `controlled_decider()`

La politique reproduisible suit quatre règles :

| État observé | Action |
|---|---|
| aucune recherche exécutée | `retrieve` avec `lexical` et la question initiale |
| aucune facette manquante | `finish` |
| budget épuisé et facettes manquantes | `abstain` |
| budget disponible et facettes manquantes | `retrieve` avec `semantic` et la requête affinée |

La première branche est fournie. Dans les trois autres branches :

- remplacez les raisons `"À compléter"` par une justification claire ;
- dans la dernière branche, remplacez `tool=None` par l'outil attendu.

Vous ne devez pas modifier `refined_query` : elle est construite à partir des
`query_hint` des facettes encore manquantes.

Les trois zones à compléter ont cette forme :

```python
return RetrievalAction(action="finish", reason="...")
return RetrievalAction(action="abstain", reason="...")
return RetrievalAction(
    action="retrieve",
    tool="________",
    query=refined_query,
    reason="...",
)
```

Indice : après un échec lexical, l'outil suivant doit exploiter la proximité de
sens plutôt que les mots exacts.

Test :

```bash
pytest -v -k "todo_2"
```

## 8. TODO 3 — Exécuter un outil autorisé

**Fonction :** `execute_retrieval_action()`

Les deux contrôles sont déjà fournis :

```python
if action.action != "retrieve" or action.tool is None:
    raise ValueError(...)
if action.tool not in tools:
    raise KeyError(...)
```

Il reste à remplacer `return []` par l'appel de l'outil sélectionné. Le registre
`tools` associe un nom à une fonction :

```python
tools[action.tool]
```

Cette fonction attend trois arguments dans cet ordre :

```text
chunks, requête de l'action, k
```

Vous pouvez procéder en deux lignes :

```python
selected_tool = tools[action.________]
return selected_tool(chunks, action.________, k)
```

Le TODO doit exécuter exactement un outil, une seule fois.

## 9. TODO 4 — Fusionner la mémoire des cycles

**Fonction :** `merge_candidates()`

L'agent doit conserver les passages des cycles précédents sans dupliquer un
`chunk_id`. Si le même chunk est retrouvé deux fois, gardez la version ayant le
meilleur `retrieval_score`.

Complétez les trois zones :

```python
if chunk_id not in by_id or ______:
```

La condition compare le score de `item` à celui déjà enregistré dans
`by_id[chunk_id]`. Utilisez `item.get("retrieval_score", 0.0)`.

La comparaison attendue est de la forme :

```python
score_du_nouvel_item > score_de_l_item_deja_memorise
```

Puis complétez :

```python
key=lambda item: ______
...
)[:______]
```

Le classement est décroissant et la mémoire ne doit pas dépasser `limit`.

Indice : les trois blancs correspondent respectivement à une comparaison de
deux scores, au score servant de clé de tri, puis à la limite de la mémoire.

Test :

```bash
pytest -v -k "todo_4"
```

## 10. TODO 5 — Mettre à jour l'état partagé

**Fonction :** `update_state_after_retrieval()`

La fusion, l'évaluation des facettes et les compteurs sont fournis. Remplacez
uniquement les quatre valeurs incomplètes :

```python
state.covered_facets = ...
state.missing_facets = ...
...
covered_facets=...
missing_facets=...
```

Utilisez les variables locales `covered` et `missing`. La trace doit enregistrer
exactement l'observation qui vient d'être écrite dans l'état.

Il n'y a aucun nouveau calcul à écrire : les quatre blancs réutilisent seulement
les deux variables déjà produites :

```python
state.covered_facets = ________
state.missing_facets = ________
# Puis recopier les mêmes valeurs dans AgentTraceEvent.
```

Test :

```bash
pytest -v -k "todo_5"
```

## 11. TODO 6 — Construire la boucle agentique bornée

**Fonction :** `run_adaptive_retrieval()`

Remplacez le `raise NotImplementedError` par les branches suivantes :

```python
if action.action == "finish":
    state.status = "completed"
    break

if action.action == "abstain":
    state.status = (
        "budget_exhausted"
        if state.iteration >= state.max_iterations
        else "abstained"
    )
    break

if state.iteration >= state.max_iterations:
    state.status = "budget_exhausted"
    break

results = execute_retrieval_action(action, tools, chunks, k)
update_state_after_retrieval(state, action, results, memory_limit)
```

Cette boucle illustre :

```text
décider → contrôler → agir → observer → mémoriser → décider à nouveau
```

Le budget est contrôlé par le programme, pas par le LLM.

Ordre à respecter dans la boucle :

1. traiter `finish` ;
2. traiter `abstain` ;
3. bloquer une nouvelle recherche si le budget est épuisé ;
4. exécuter l'outil ;
5. mettre à jour l'état, puis recommencer la boucle.

Ne créez pas un nouvel état à chaque tour : `update_state_after_retrieval()`
modifie l'objet `state` existant.

Test :

```bash
pytest -v -k "todo_6"
```

## 12. TODO 7 — Comparer la baseline et l'agent

**Fonction :** `compare_baseline_and_agentic()`

Les deux variantes doivent être évaluées avec le même ensemble `relevant`.
Remplacez les deux `None` par deux appels à `retrieval_metrics()`.

Pour la baseline, transmettez les identifiants de `baseline_results` :

```python
[item["chunk_id"] for item in baseline_results]
```

Pour la variante agentique, utilisez :

```python
[item["chunk_id"] for item in agent_state.selected_chunks]
```

Squelette des deux lignes attendues :

```python
baseline_metrics = retrieval_metrics(________, relevant)
agent_metrics = retrieval_metrics(________, relevant)
```

Attention : transmettez des listes de `chunk_id`, pas les dictionnaires complets.

Le rapport calcule ensuite le gain de rappel et le nombre d'appels
supplémentaires. Un meilleur rappel n'est pas gratuit : il faut aussi observer
le coût.

## 13. TODO 8 — Utiliser un décideur LLM structuré

**Fonction :** `build_llm_decider()`

Le contrat `RetrievalAction`, le prompt et l'adaptateur structuré sont fournis :

```python
structured_model = model.with_structured_output(RetrievalAction)
```

Le prompt reçoit aussi, pour chaque facette manquante, son `label`, ses
`keywords` et son `query_hint`. Le LLM ne doit donc pas deviner le vocabulaire
de recherche : il choisit une action structurée et construit une requête à
partir de ces informations. Après le premier échec lexical, l'action attendue
est une recherche `semantic` ciblée.

Remplacez :

```python
return ...
```

par un seul appel de `structured_model` avec `prompt`.

Vous avez déjà utilisé cette méthode dans les TD précédents :

```python
return structured_model.________(prompt)
```

Il ne faut ni parser manuellement du JSON, ni appeler le modèle une deuxième
fois : `with_structured_output(RetrievalAction)` effectue déjà l'adaptation vers
le contrat Pydantic.

Le LLM propose une décision structurée, mais le programme conserve :

- la liste des outils autorisés ;
- le budget maximal ;
- la validation Pydantic ;
- le contrôle des preuves ;
- la possibilité de s'abstenir.

Même avec une sortie structurée, un modèle local peut omettre `tool` ou
`query`. L'interface intercepte alors l'erreur de validation et applique la
politique contrôlée. La colonne `raison` de la trace indique explicitement ce
fallback : l'échec du LLM reste observable au lieu d'arrêter tout le workflow.

## 14. Résultats attendus

Avec le mode entièrement contrôlé et un budget de deux itérations :

| Cas | Baseline TD05 | Variante TD06 |
|---|---|---|
| Q1 du TD05 | peut manquer la preuve anglaise | cible les valeurs de rappel avec le sémantique |
| Q2 du TD05 | écart de vocabulaire français/anglais | peut reformuler et changer d'outil |
| Q3 du TD05 | plusieurs stratégies attendues | contrôle toutes les facettes avant de répondre |

Avec un budget limité à un cycle, Q2 ou Q3 montre aussi l'abstention contrôlée
si la couverture reste incomplète.

L'agentivité n'est donc pas « plus d'appels LLM ». Elle correspond ici au fait
que **la prochaine action dépend de l'observation obtenue pendant l'exécution**.
