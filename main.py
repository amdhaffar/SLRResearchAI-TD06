from pathlib import Path

from slrresearch.agentic_rag import (
    compare_baseline_and_agentic,
    controlled_decider,
    run_adaptive_retrieval,
)
from slrresearch.agentic_rag_artifacts import (
    load_agentic_cases,
    save_comparison_artifact,
)
from slrresearch.agentic_rag_runtime import ProtocolEmbeddings, build_retrieval_tools
from slrresearch.rag_artifacts import load_rag_handoff
from slrresearch.rag_pipeline import chunk_pages


def load_chunks(root: Path):
    documents, _ = load_rag_handoff(root / "artifacts")
    return [
        chunk
        for document in documents
        for chunk in chunk_pages(document.paper_id, document.pages, size=1000, overlap=0)
    ]


def main():
    root = Path(__file__).parent
    chunks = load_chunks(root)
    cases = load_agentic_cases(root / "artifacts" / "agentic_rag_cases.json")
    for case in cases:
        tools = build_retrieval_tools(
            ProtocolEmbeddings(chunks, set(case.relevant_chunk_ids))
        )
        baseline = tools[case.baseline_tool](
            chunks, case.question, case.baseline_k
        )
        state = run_adaptive_retrieval(
            case.question,
            case.facets,
            chunks,
            tools,
            decider=controlled_decider,
            k=case.baseline_k,
            max_iterations=2,
        )
        report = compare_baseline_and_agentic(case, baseline, state)
        print(f"\n=== {case.id} — {case.label} ===")
        print("Baseline :", report.baseline)
        print("Agentique :", report.agentic)
        print("Trace :", [event.model_dump() for event in state.trace])
        save_comparison_artifact(
            report,
            [event.model_dump() for event in state.trace],
            root / "artifacts" / f"agentic_result_{case.id.lower()}.json",
        )


if __name__ == "__main__":
    main()
