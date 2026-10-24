"""Retrieval-Augmented Generation: retrieve chunks, then make the LLM answer from them only."""
from backend.config import DEFAULT_TOP_K, INSIGHT_TOP_K, MIN_RELEVANCE
from backend.services import embedding_service, llm_service, vector_store

NOT_FOUND = "I could not find this in the uploaded documents."

QA_SYSTEM_PROMPT = f"""You are a careful document analyst.
Answer the question using ONLY the numbered context excerpts provided.
Rules:
- If the excerpts do not contain the answer, reply exactly: "{NOT_FOUND}"
- Cite the excerpts you used with their numbers in square brackets, like [1] or [2][3].
- Never use outside knowledge. Never invent clauses, names, dates or numbers.
- Be concise and precise."""

INSIGHT_SYSTEM_PROMPT = """You are a careful document analyst.
Work ONLY from the numbered context excerpts provided; never use outside knowledge
and never invent details. Cite excerpt numbers in square brackets, like [1] or [2][3].
If the excerpts contain nothing relevant, say so plainly."""

# Each "smart insight" button = a retrieval query + an instruction for the LLM.
INSIGHTS = {
    "summary": {
        "query": "overview purpose parties scope main terms agreement introduction conclusion",
        "task": "Write a structured summary: what the document is, who is involved, "
        "and the main points. Use at most 8 short bullet points.",
    },
    "risks": {
        "query": "risk liability penalty termination indemnity breach damages warranty dispute limitation",
        "task": "List clauses or statements that carry risk or are unfavourable to the "
        "reader (liability, penalties, termination, indemnity, lock-in, ambiguity). "
        "For each: one line quoting or describing it and one line on why it is risky.",
    },
    "key_dates": {
        "query": "date deadline term renewal effective expiry notice period within days payment due",
        "task": "List every date, deadline, duration and notice period mentioned, as "
        "bullet points: what it is, the date or period, and the excerpt number.",
    },
    "obligations": {
        "query": "shall must obligation responsible duty required to deliver pay provide comply",
        "task": "List the obligations, grouped by the party responsible for them. "
        "One bullet per obligation.",
    },
}


def retrieve(query, top_k=DEFAULT_TOP_K, source=None):
    return vector_store.search(embedding_service.embed_query(query), top_k, source)


def _format_context(hits):
    blocks = []
    for number, hit in enumerate(hits, start=1):
        blocks.append(f"[{number}] (Source: {hit['source']}, page {hit['page']})\n{hit['text']}")
    return "\n\n".join(blocks)


def _public_sources(hits):
    return [
        {
            "id": number,
            "source": hit["source"],
            "page": hit["page"],
            "score": hit["score"],
            "text": hit["text"],
        }
        for number, hit in enumerate(hits, start=1)
    ]


def answer_question(question, top_k=DEFAULT_TOP_K, source=None):
    hits = [h for h in retrieve(question, top_k, source) if h["score"] >= MIN_RELEVANCE]
    if not hits:
        # Don't even call the LLM: nothing relevant was retrieved, so any
        # answer would be a guess. This is the main defence against hallucination.
        return {"answer": NOT_FOUND, "sources": []}

    user_prompt = (
        f"Context excerpts:\n\n{_format_context(hits)}\n\n"
        f"Question: {question}\n\nAnswer (cite excerpt numbers like [1]):"
    )
    answer = llm_service.generate(QA_SYSTEM_PROMPT, user_prompt)
    return {"answer": answer, "sources": _public_sources(hits)}


def generate_insight(kind, source):
    spec = INSIGHTS[kind]
    hits = retrieve(spec["query"], INSIGHT_TOP_K, source)
    if not hits:
        return {"answer": NOT_FOUND, "sources": []}

    hits.sort(key=lambda h: (h["page"], h["chunk_index"]))  # read in document order
    user_prompt = (
        f"Context excerpts:\n\n{_format_context(hits)}\n\n"
        f"Task: {spec['task']}"
    )
    answer = llm_service.generate(INSIGHT_SYSTEM_PROMPT, user_prompt)
    return {"answer": answer, "sources": _public_sources(hits)}
