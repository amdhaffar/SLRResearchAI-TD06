import json
import re
from collections.abc import Callable

from .contracts import AgentAction, SearchState


DecisionPolicy = Callable[[SearchState, list[str]], dict]


def state_for_decision(state: SearchState) -> dict:
    """Projection minimale de l'état exposée au décideur."""
    return {
        "objective": state.objective,
        "available_queries": state.queries,
        "papers_found": len(state.papers),
        "tools_already_called": [action.tool for action in state.actions_executed],
        "last_observation": (
            state.observations[-1].model_dump(exclude={"papers"})
            if state.observations
            else None
        ),
        "step": state.step,
    }


def build_decision_prompt(state: SearchState, allowed_tools: list[str]) -> str:
    """Construit le prompt de décision contrôlée."""
    projected_state = json.dumps(state_for_decision(state), ensure_ascii=False, indent=2)
    allowed = json.dumps(allowed_tools, ensure_ascii=False)
    return f"""Tu es le décideur d'un SearchAgent scientifique contrôlé.
Tu proposes une seule action structurée. Tu n'exécutes jamais un outil.

OBJECTIF
{state.objective}

ÉTAT UTILE
{projected_state}

OUTILS AUTORISÉS
{allowed}

CONTRAT DE SORTIE JSON
Pour rechercher :
{{"action":"search","tool":"nom autorisé","query":"requête disponible","limit":5}}
Pour arrêter :
{{"action":"finish","tool":null,"query":null,"limit":null}}

RÈGLES
- Choisis seulement un outil autorisé.
- Après une erreur ou un résultat vide, essaie si utile une autre source non appelée.
- Si assez d'articles sont disponibles ou si aucune action utile ne reste, choisis finish.
- Retourne uniquement l'objet JSON, sans Markdown ni explication.
"""


def extract_json_object(text: str) -> dict:
    """Extrait un objet JSON, même entouré d'un bloc Markdown."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Aucun objet JSON trouvé dans la décision")
    return json.loads(cleaned[start : end + 1])


class LLMDecisionPolicy:
    """Adaptateur : le LLM propose un dictionnaire, le runtime validera ensuite."""

    def __init__(self, model):
        self.model = model

    def __call__(self, state: SearchState, allowed_tools: list[str]) -> dict:
        """Effectue un appel LLM puis extrait le JSON, sans tool call direct."""
        response = self.model.invoke(build_decision_prompt(state, allowed_tools))
        return extract_json_object(str(response.content))


class ScriptedPolicy:
    """Politique reproductible pour tester la boucle sans LLM ni Internet."""

    def __init__(self, decisions: list[dict]):
        self.decisions = list(decisions)
        self.calls = 0

    def __call__(self, state: SearchState, allowed_tools: list[str]) -> dict:
        self.calls += 1
        if not self.decisions:
            return {"action": "finish"}
        return self.decisions.pop(0)
