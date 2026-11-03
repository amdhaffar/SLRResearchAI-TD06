from pathlib import Path
import json

from slrresearch.rag_artifacts import load_rag_handoff
from slrresearch.rag_pipeline import chunk_pages, lexical_retrieve


def main():
    root = Path(__file__).parent
    documents, lineage = load_rag_handoff(root / "artifacts")
    evaluation = json.loads((root / "artifacts" / "rag_evaluation.json").read_text(encoding="utf-8"))
    chunks = []
    for document in documents:
        chunks.extend(chunk_pages(document.paper_id, document.pages, size=1000, overlap=0))
    question = evaluation["questions"][0]["question"]
    print("Transmission :", lineage["sources"])
    print("Corpus RAG :", lineage["rag_documents"], "document(s)")
    print("Question :", question)
    for result in lexical_retrieve(chunks, question, k=2):
        print(result["chunk_id"], round(result["retrieval_score"], 3), result["text"])


if __name__ == "__main__":
    main()
