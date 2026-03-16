import logging
import re

from langchain_core.messages import HumanMessage

CHITCHAT_PATTERNS = [
    r"^(hi|hello|hey|howdy|sup|yo)\b",
    r"^(good\s)?(morning|afternoon|evening|night)\b",
    r"^how are you",
    r"^what('s| is) up",
    r"^thanks?(\s|!|$)",
    r"^thank you",
    r"^ok(ay)?\s*$",
    r"^bye\b",
    r"^who are you",
    r"^what can you do",
]


def is_chitchat(query: str) -> bool:
    q = query.strip().lower()
    for pattern in CHITCHAT_PATTERNS:
        if re.search(pattern, q):
            return True
    if len(q.split()) <= 3 and "?" not in q:
        return True
    return False


def decompose_query(query: str, model) -> list:
    """
    Asks the LLM to break a complex query into focused sub-questions.
    Returns a list of strings. Falls back to [query] on any failure.
    """
    prompt = (
        "Break the following question into focused sub-questions if it covers "
        "multiple topics. If it is already a single focused question, return it as-is.\n"
        "Return ONLY a numbered list like:\n1. sub-question one\n2. sub-question two\n\n"
        "Question: " + query
    )
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        lines = response.content.strip().splitlines()
        sub_questions = []
        for line in lines:
            line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if line:
                sub_questions.append(line)
        return sub_questions if sub_questions else [query]
    except Exception:
        logging.exception("Query decomposition failed.")
        return [query]


def decide_sources(query: str, has_docs: bool) -> list:
    """
    Simple source decision per sub-query.
    Returns ordered list of sources to try.

    Strategy: Always try RAG first if documents available, then web as fallback.
    The context_builder will process each sub-query independently and combine results.
    """
    if has_docs:
        return ["rag", "web"]  # Try RAG first, then web if RAG finds nothing
    else:
        return ["web"]  # No docs, use web only


def is_query_complex(query: str) -> bool:
    """
    Heuristic to detect if a query is complex (needs reasoning with chain-of-thought)
    vs simple (factual/definitional - direct answer only).

    Complex patterns: 'how to', 'why', 'compare', 'explain', 'analyze', etc.
    Simple patterns: 'what is', 'who is', 'when', 'where', etc.

    Returns True if complex (should use Reasoning mode), False if simple.
    """
    q = query.strip().lower()

    # Simple factual queries - direct answer only
    simple_patterns = [
        r"^what is",
        r"^who is",
        r"^when",
        r"^where",
        r"^definition of",
        r"^list ",
        r"^give me",
        r"^tell me about",
    ]

    for pattern in simple_patterns:
        if re.search(pattern, q):
            return False

    # Complex reasoning queries - benefit from chain-of-thought
    complex_patterns = [
        r"how to",
        r"how do",
        r"how can",
        r"why",
        r"compare",
        r"explain",
        r"analyze",
        r"design",
        r"implement",
        r"solve",
        r"build",
        r"create",
        r"suggest",
        r"recommend",
        r"benefit",
        r"impact",
        r"relationship",
    ]

    for pattern in complex_patterns:
        if re.search(pattern, q):
            return True

    # If more than 10 words and starts with "what", likely complex
    word_count = len(q.split())
    if word_count > 10:
        return True

    # Default: not complex for direct questions
    return False
