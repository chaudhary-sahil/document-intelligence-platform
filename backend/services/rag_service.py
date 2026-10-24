"""Retrieval-Augmented Generation: retrieve chunks, then make the LLM answer from them only."""
import re

from backend.config import DEFAULT_TOP_K, INSIGHT_TOP_K, MIN_RELEVANCE
from backend.services import embedding_service, llm_service, vector_store

NOT_FOUND = "I could not find this in the uploaded documents."

# The rules below target mistakes seen in testing with a small local model:
# invented clause numbers, "either party" narrowed to one party, and a due
# date confused with an invoice date.
_GROUNDING_RULES = """- Copy numbers, durations, dates, names and amounts exactly as written in the excerpts.
- Keep the subject of each right or duty exactly as written. If the text says "either party" or "each party", write that; never narrow it to one named party.
- Mention clause numbers or headings only if they appear in the excerpt text, exactly as written. Never invent, combine or guess numbers.
- Never use outside knowledge and never speculate beyond what the excerpts state.
- Do not copy the "(Source: ...)" headers into your answer.
- Excerpt numbers are ONLY the numbers in square brackets printed before each excerpt (1 to N). Clause or section numbers inside the text are NOT excerpt numbers: never put them in square brackets.
- State every fact in words. A reply made only of citations, such as "[1]", is wrong: write the answer, then add the citation after it, e.g. "30 days [1]"."""

QA_SYSTEM_PROMPT = f"""You are a careful document analyst.
Answer the question using ONLY the numbered context excerpts provided.
Rules:
- If the excerpts do not contain the answer, reply exactly: "{NOT_FOUND}"
{_GROUNDING_RULES}
- Be concise: 1 to 4 complete sentences unless a list is clearly needed."""

INSIGHT_SYSTEM_PROMPT = f"""You are a careful document analyst.
Work ONLY from the numbered context excerpts provided.
Rules:
{_GROUNDING_RULES}
- If the excerpts contain nothing relevant, say so plainly.
- Format as a markdown bullet list: every item on its own line starting with "- ". No introduction and no closing remarks."""

# Each "smart insight" button = a retrieval query + an instruction for the LLM.
INSIGHTS = {
    "summary": {
        "query": "overview purpose parties scope main terms agreement introduction conclusion",
        "task": "Summarise the document in at most 6 bullets: what kind of document it is, "
        "who the parties or authors are, and the key terms. Use only facts stated in the excerpts.",
    },
    "risks": {
        "query": "risk liability penalty termination indemnity breach damages warranty dispute limitation renewal interest",
        "task": "Identify provisions that could be unfavourable or risky for a party (liability "
        "limits, penalties or late-payment interest, automatic renewal, termination rights, "
        "lock-in, dispute resolution). Include a provision only if an excerpt states it. "
        "For each bullet give: the heading or number exactly as written, what it says using "
        "the exact figures, and in a few words why it may be risky. Say who it affects only as the "
        "text does: if a clause applies to 'either party', write 'either party' and do not "
        "assign it to the Client or the Provider. At most 6 bullets.",
    },
    "key_dates": {
        "query": "date deadline term renewal effective expiry notice period within days payment due invoice",
        "task": "List every date, deadline, duration, notice period and payment term stated. "
        "Write each bullet as 'Item: exact value [n]', for example '- Delivery deadline: 14 days "
        "after order [2]' (this example is only a format guide; never output it). "
        "Distinguish when something is issued from when it is due and do not repeat an item.",
    },
    "obligations": {
        "query": "shall must obligation responsible duty required to deliver pay provide comply",
        "task": "List the obligations grouped by the party responsible. Start each group with "
        "a bold party name on its own line, then one bullet per obligation. Include only duties "
        "the excerpts state. Duties of 'either party' or 'each party' go in a shared group.",
    },
}


def normalize_markdown(text):
    """Turn inline '•' bullets (small models love them) into real markdown lists."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]*[•●▪‣]\s*", "\n- ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_CITE = re.compile(r"\[(\d+)\]")
_LEAK = re.compile(r"(?i)what it is.*exact value|exact value.*excerpt number")
EMPTY_NOTE = (
    "The model did not produce a usable answer. The most relevant passages "
    "are listed under Sources."
)


def clean_citations(text, n):
    """Remove citations like [8] that point at excerpts that don't exist (1..n are valid)."""
    text = _CITE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= n else "", text)
    return re.sub(r"[ \t]+([.,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", text))


def drop_leaked_lines(text):
    """Drop lines where the model repeated our formatting instructions back."""
    return "\n".join(line for line in text.split("\n") if not _LEAK.search(line))


def has_content(text):
    """False for replies that are only citations or punctuation (e.g. just '[1]')."""
    return len(re.sub(r"\[\d+\]|[^A-Za-z0-9]", "", text)) >= 2


def _finalize(answer, n):
    return normalize_markdown(drop_leaked_lines(clean_citations(answer, n)))


def _generate_checked(system_prompt, user_prompt, n):
    """Call the LLM; if it returns nothing but citations, retry once with a stronger nudge."""
    cleaned = _finalize(llm_service.generate(system_prompt, user_prompt), n)
    if has_content(cleaned):
        return cleaned
    nudge = (
        "\n\nYour previous reply contained no actual answer. Reply again and write the "
        "answer in words, then add the excerpt number in square brackets."
    )
    cleaned = _finalize(llm_service.generate(system_prompt, user_prompt + nudge), n)
    return cleaned if has_content(cleaned) else EMPTY_NOTE


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

    n = len(hits)
    user_prompt = (
        f"There are {n} context excerpts, numbered [1] to [{n}].\n\n"
        f"{_format_context(hits)}\n\n"
        f"Question: {question}\n\n"
        "Answer in a complete sentence, then cite the excerpt number, like: 30 days [1]"
    )
    answer = _generate_checked(QA_SYSTEM_PROMPT, user_prompt, n)
    return {"answer": answer, "sources": _public_sources(hits)}


def generate_insight(kind, source):
    spec = INSIGHTS[kind]
    hits = retrieve(spec["query"], INSIGHT_TOP_K, source)
    if not hits:
        return {"answer": NOT_FOUND, "sources": []}

    hits.sort(key=lambda h: (h["page"], h["chunk_index"]))  # read in document order
    n = len(hits)
    user_prompt = (
        f"There are {n} context excerpts, numbered [1] to [{n}].\n\n"
        f"{_format_context(hits)}\n\nTask: {spec['task']}"
    )
    answer = _generate_checked(INSIGHT_SYSTEM_PROMPT, user_prompt, n)
    return {"answer": answer, "sources": _public_sources(hits)}
