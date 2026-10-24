"""Pure scoring functions for the evaluation script (no I/O, easy to test)."""


def _norm(text):
    return " ".join(text.lower().replace(",", "").split())


def contains_fact(text, fact):
    """Case-, comma- and whitespace-insensitive substring check ('8,000' == '8000')."""
    return _norm(fact) in _norm(text)


def score_retrieval(item, hits, min_relevance):
    """Score one question against the retrieved hits (best first).

    Answerable question: a hit if some retrieved chunk is on the expected page
    AND contains the expected fact. Unanswerable question: correct if nothing
    scores above the relevance threshold (so the app would abstain).
    """
    top_score = hits[0]["score"] if hits else 0.0
    if not item.get("answerable", True):
        return {
            "question": item["question"],
            "answerable": False,
            "ok": top_score < min_relevance,
            "rank": None,
            "top_score": top_score,
        }

    rank = None
    for position, hit in enumerate(hits, start=1):
        if hit["page"] == item["page"] and contains_fact(hit["text"], item["fact"]):
            rank = position
            break
    return {
        "question": item["question"],
        "answerable": True,
        "ok": rank is not None,
        "rank": rank,
        "top_score": top_score,
    }


def summarize(results):
    answerable = [r for r in results if r["answerable"]]
    negatives = [r for r in results if not r["answerable"]]
    hits = [r for r in answerable if r["ok"]]
    return {
        "answerable_total": len(answerable),
        "retrieval_hits": len(hits),
        "hit_rate": len(hits) / len(answerable) if answerable else 0.0,
        "mrr": (sum(1 / r["rank"] for r in hits) / len(answerable)) if answerable else 0.0,
        "abstain_total": len(negatives),
        "abstain_correct": sum(1 for r in negatives if r["ok"]),
    }
