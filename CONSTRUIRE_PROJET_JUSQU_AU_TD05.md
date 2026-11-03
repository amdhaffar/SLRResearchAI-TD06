# Construire SLRResearchAI de bout en bout jusqu'au RAG

## 1. Finalité du guide

Le TD05 RAG peut fonctionner seul grâce à des artefacts préremplis. Dans le
projet final, ces fichiers ne doivent plus être considérés comme des données
isolées : ils doivent être produits successivement par les TD précédents.

```text
sujet scientifique
→ questions et requêtes
→ recherche d'articles
→ screening
→ préparation des documents
→ RAG et preuves vérifiées
→ rédaction scientifique
```

Ce guide présente les deux modes de travail :

- **mode TD autonome** : utiliser les artefacts fournis pour se concentrer sur
  le RAG ;
- **mode projet intégré** : remplacer chaque artefact fourni par la sortie
  réelle du composant précédent.

## 2. Un seul projet Python qui grandit

Tous les TD utilisent désormais le même package `slrresearch`. Le TD05 ne
supprime pas le code précédent : il ajoute quatre modules RAG aux composants
déjà acquis.

```text
slrresearch/
├── research_questions.py, search_plan.py       # TD01–TD02
├── tools.py, real_tools.py, react_agent.py      # TD03
├── workflow.py, contracts.py, policies.py       # TD03
├── artifacts.py, screening_agent.py             # TD04
├── llm_screening_agent.py                       # TD04
├── rag_models.py, rag_artifacts.py               # TD05
├── rag_pipeline.py, pdf_loader.py                # TD05
└── config.py                                     # configuration partagée
```

Deux mécanismes assurent la continuité :

1. les **imports Python** permettent aux nouveaux modules de réutiliser les
   classes et fonctions existantes ;
2. les **artefacts JSON** conservent les sorties entre deux exécutions et
   rendent les étapes découplées et reproductibles.

Les artefacts préremplis servent uniquement pendant la séance. Pour le projet,
il suffit d'exécuter les producteurs précédents et d'écrire leurs résultats aux
mêmes chemins. Les contrats et les identifiants restent inchangés.

## 3. Contrats entre les étapes

| Producteur | Artefact | Consommateur | Information transmise |
|---|---|---|---|
| TD02 | `search_plan.json` | TD03 | requêtes par moteur scientifique |
| TD03 | `search_results.json` | TD04 | articles trouvés et dédupliqués |
| TD04 | `screening_decisions.json` | préparation documentaire | décision finale par article |
| Préparation documentaire | `document_manifest.json` | TD05 RAG | disponibilité et emplacement des documents |
| Extraction PDF | `extracted_pages.json` | TD05 RAG | texte conservé page par page |
| TD05 RAG | `verified_evidence.json` | TD de rédaction | preuves, citations et provenance |

L'identifiant `paper_id` constitue la clé de transmission commune. Il ne doit
pas être recréé ou remplacé entre deux étapes.

## 4. Exécuter d'abord le TD en mode autonome

Depuis le dossier du TD :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest -v
python -m streamlit run streamlit_app.py
```

Les fichiers suivants sont déjà fournis dans `artifacts/` :

```text
search_results.json
screening_decisions.json
document_manifest.json
extracted_pages.json
rag_evaluation.json
```

Ils permettent de terminer les dix TODO sans relancer les API scientifiques,
le screening ou le téléchargement.

## 5. Comprendre la jointure d'entrée

La fonction fournie se trouve dans :

```text
slrresearch/rag_artifacts.py
```

Elle est appelée ainsi :

```python
from pathlib import Path
from slrresearch.rag_artifacts import load_rag_handoff

documents, lineage = load_rag_handoff(Path("artifacts"))
```

Elle calcule l'intersection suivante :

```text
identifiants trouvés au TD03
∩ identifiants inclus au TD04
∩ identifiants dont le document est disponible
```

Un article exclu, incertain sans arbitrage final ou indisponible ne doit pas
entrer silencieusement dans le corpus RAG.

## 6. Remplacer l'artefact prérempli du TD03

Le TD03 doit exporter ses résultats dans un fichier respectant au minimum ce
contrat :

```json
{
  "schema_version": "1.0",
  "origin": "TD03",
  "topic": "sujet de la revue",
  "papers": [
    {
      "id": "identifiant stable",
      "title": "titre",
      "abstract": "résumé",
      "doi": "doi éventuel",
      "source": "openalex",
      "url": "adresse du document",
      "open_access": true
    }
  ]
}
```

Pour intégrer une exécution réelle :

1. exécuter le SearchAgent du TD03 ;
2. exporter son artefact ;
3. copier le résultat vers `artifacts/search_results.json` ;
4. conserver tous les `paper_id` sans les renommer ;
5. relancer les tests de transmission.

## 7. Produire réellement l'artefact du TD04

Après le screening déterministe, le screening LLM et l'arbitrage humain, chaque
papier doit posséder une décision finale.

Le contrat attendu est :

```json
{
  "schema_version": "1.0",
  "origin": "TD04",
  "source_artifact": "search_results.json",
  "decisions": [
    {
      "paper_id": "identifiant venant du TD03",
      "final_decision": "include",
      "decided_by": "human",
      "reason": "justification traçable"
    }
  ]
}
```

Règles importantes :

- ne pas exporter `uncertain` comme une décision définitive ;
- demander l'arbitrage humain avant l'export lorsqu'il reste un conflit ;
- ne pas perdre les raisons de la décision ;
- vérifier que chaque `paper_id` existe dans `search_results.json`.

Le fichier final remplace ensuite :

```text
artifacts/screening_decisions.json
```

## 8. Préparer les documents inclus

Pour chaque article définitivement inclus :

1. récupérer légalement le PDF lorsqu'il est disponible ;
2. conserver son URL et son statut ;
3. demander une intervention humaine s'il est derrière un paywall ;
4. extraire son texte page par page avec `extract_pdf_pages()` ;
5. conserver le même `paper_id`.

Le manifeste attendu ressemble à :

```json
{
  "paper_id": "P1",
  "status": "downloaded",
  "source_url": "https://.../paper.pdf",
  "extracted_pages_key": "doc_P1"
}
```

Les statuts possibles sont :

- `downloaded` : le texte peut entrer dans le RAG ;
- `manual_required` : une action humaine est nécessaire ;
- `unavailable` : le document ne doit pas entrer dans le corpus courant.

Dans cette version pédagogique, `extracted_pages.json` remplace les PDF locaux
pour assurer une exécution reproductible. Dans le projet personnel, les pages
peuvent être produites par :

```python
from slrresearch.pdf_loader import extract_pdf_pages

pages = extract_pdf_pages("documents/P1.pdf")
```

Un PDF scanné sans couche textuelle nécessite un OCR avant cette étape.

## 8. Construire le corpus RAG

Une fois `documents` obtenu par `load_rag_handoff()` :

```python
chunks = []
for document in documents:
    chunks.extend(
        chunk_pages(
            paper_id=document.paper_id,
            pages=document.pages,
            size=400,
            overlap=80,
        )
    )
```

Chaque chunk doit conserver :

```text
paper_id + chunk_id + page + text
```

Sans ces métadonnées, une citation ne peut plus être auditée correctement.

## 9. Comparer les retrievers correctement

Le protocole expérimental doit conserver :

```text
même corpus + même question + même k
```

Puis comparer :

- les identifiants des chunks retournés ;
- leur ordre et leur score ;
- `Precision@k` ;
- `Recall@k` lorsque les passages pertinents sont annotés.

`rag_evaluation.json` contient les annotations pédagogiques. Dans un projet
personnel, ces annotations doivent être réalisées par un évaluateur humain sur
un sous-ensemble représentatif.

## 10. Extraire et vérifier la preuve

Le pipeline final suit cet ordre :

```python
selected = retriever(chunks, question, k)
context = build_context(selected)
proposal = extractor(question, context)
evidence = verify_evidence(proposal, selected)
```

Le LLM propose une réponse et une citation. Le programme vérifie ensuite :

- que `chunk_id` appartient au contexte transmis ;
- que `paper_id` et `page` correspondent au chunk ;
- que la citation apparaît réellement dans le passage.

Le statut final est `verified`, `human_review` ou `rejected`.

## 11. Produire l'artefact pour le TD suivant

Après une exécution réussie dans Streamlit, le fichier suivant est créé :

```text
artifacts/verified_evidence.json
```

Il contient notamment :

```json
{
  "origin": "RAG_TD05",
  "evidence": [
    {
      "answer": "résultat synthétique",
      "supporting_quote": "citation exacte",
      "paper_id": "P1",
      "chunk_id": "P1_p2_c1",
      "page": 2,
      "status": "verified"
    }
  ]
}
```

Le prochain TD de rédaction devra consommer cet artefact et non relire
directement une réponse libre du LLM. Une affirmation rédigée devra citer un
`paper_id` présent dans les preuves autorisées.

## 12. Vérifications de bout en bout

Avant de considérer l'intégration terminée, vérifier :

- [ ] les `paper_id` du screening existent dans les résultats de recherche ;
- [ ] chaque papier inclus possède une décision finale ;
- [ ] les documents indisponibles ne sont pas indexés ;
- [ ] chaque chunk conserve article, page et identifiant ;
- [ ] lexical et sémantique utilisent le même corpus et le même `k` ;
- [ ] les métriques ne sont calculées qu'avec des références annotées ;
- [ ] chaque preuve est contrôlée après la génération ;
- [ ] les cas `human_review` sont visibles et non acceptés automatiquement ;
- [ ] `verified_evidence.json` est produit ;
- [ ] le futur rédacteur ne reçoit que les preuves autorisées.

## 13. Commandes de diagnostic

Tester uniquement la transmission des artefacts :

```bash
pytest -v tests/test_artifact_chain.py
```

Tester un TODO précis :

```bash
pytest -v -k "todo_5_"
```

Tester tout le projet :

```bash
pytest -v
```

Observer la continuité dans l'interface :

```bash
python -m streamlit run streamlit_app.py
```

La première zone de l'interface doit afficher le nombre d'articles trouvés,
inclus, disponibles et réellement transmis au RAG.
