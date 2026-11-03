from dataclasses import asdict
import json
from pathlib import Path

from slrresearch.artifacts import load_search_artifact
from slrresearch.screening_agent import ScreeningAgent

ARTIFACT_PATH = Path("artifacts/search_results.json")
INCLUDE_TERMS = {"slr", "screening", "systematic review"}
EXCLUDE_TERMS = {"medicine", "clinical trial"}

def main() -> None:
    artifact = load_search_artifact(ARTIFACT_PATH)
    print("=== HANDOFF TD03 → TD04 ===")
    print("origine :", artifact.origin)
    print("artefact pédagogique :", artifact.pedagogical_fixture)
    print("articles reçus :", len(artifact.papers))
    print("articles avec résumé :", sum(bool(p.abstract.strip()) for p in artifact.papers))
    result = ScreeningAgent().run(artifact.papers, INCLUDE_TERMS, EXCLUDE_TERMS)
    print("\n=== RECOMMANDATIONS AUTOMATIQUES TD04 ===")
    print(json.dumps({"decisions":[asdict(x) for x in result["decisions"]],"human_queue":[asdict(x) for x in result["human_queue"]]}, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
