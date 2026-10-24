"""Measure retrieval quality (and optionally answer quality) on a small question set.

Run from the project root:
    python scripts/evaluate.py                 # retrieval only, no LLM needed
    python scripts/evaluate.py --llm           # also check the generated answers
    python scripts/evaluate.py --top-k 3

Writes docs/eval_results.md so the numbers can be shown in the README.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import config, evaluation  # noqa: E402
from backend.services import embedding_service, ingest_service, rag_service, vector_store  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--file", default=str(ROOT / "data" / "eval" / "sample_contract_eval.json"))
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument("--llm", action="store_true", help="also grade generated answers")
    parser.add_argument("--verbose", action="store_true", help="print each generated answer (with --llm)")
    args = parser.parse_args()

    spec = json.loads(Path(args.file).read_text())
    document = spec["document"]

    embedding_service.warm_up()
    if document not in {d["filename"] for d in vector_store.list_documents()}:
        sample = ROOT / "data" / "samples" / document
        print(f"Indexing {sample.name} first…")
        with open(sample, "rb") as handle:
            ingest_service.ingest_pdf(document, handle, allow_protected=True)

    print(f"Embeddings: {embedding_service.backend_name()}  top_k={args.top_k}  "
          f"min_relevance={config.MIN_RELEVANCE}\n")

    results = []
    for item in spec["questions"]:
        hits = rag_service.retrieve(item["question"], args.top_k, document)
        result = evaluation.score_retrieval(item, hits, config.MIN_RELEVANCE)
        if args.llm:
            answer = rag_service.answer_question(item["question"], args.top_k, document)["answer"]
            if result["answerable"]:
                result["answer_ok"] = evaluation.contains_fact(answer, item["fact"])
            else:
                result["answer_ok"] = answer == rag_service.NOT_FOUND
        results.append(result)
        mark = "PASS" if result["ok"] else "FAIL"
        extra = f"  answer={'ok' if result['answer_ok'] else 'WRONG'}" if args.llm else ""
        rank = f"rank {result['rank']}" if result["rank"] else "-"
        print(f"[{mark}] {result['question']:<66} top={result['top_score']:.2f} {rank}{extra}")
        if args.llm and args.verbose:
            print(f"       -> {answer!r}")

    s = evaluation.summarize(results)
    lines = [
        f"- Retrieval hit rate @{args.top_k}: **{s['retrieval_hits']}/{s['answerable_total']}** "
        f"({s['hit_rate']:.0%}), MRR {s['mrr']:.2f}",
        f"- Correct abstentions (unanswerable questions): **{s['abstain_correct']}/{s['abstain_total']}**",
    ]
    if args.llm:
        good = sum(1 for r in results if r.get("answer_ok"))
        lines.append(f"- Answers containing the expected fact / correct refusal: **{good}/{len(results)}**")
    print("\n" + "\n".join(l.replace("**", "") for l in lines))

    table = ["| Question | Result | Top score | Rank |", "|---|---|---|---|"]
    for r in results:
        table.append(f"| {r['question']} | {'✅' if r['ok'] else '❌'} | {r['top_score']:.2f} | {r['rank'] or '-'} |")
    out = ROOT / "docs" / "eval_results.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        f"# Evaluation results\n\nDocument: `{document}` · embeddings: `{embedding_service.backend_name()}` "
        f"· top_k={args.top_k} · min_relevance={config.MIN_RELEVANCE}\n\n"
        + "\n".join(lines) + "\n\n" + "\n".join(table) + "\n"
    )
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
