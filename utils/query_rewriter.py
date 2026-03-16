import logging


def rewrite_query(query: str, chat_model) -> str:
    """
    Uses a small prompt to produce a cleaner, more retrieval-friendly
    version of the user's query. Falls back to the original on failure.
    Keeps token usage minimal by using a single-turn call with no history.
    """
    prompt = (
        "Rewrite the following question to be more specific and retrieval-friendly "
        "for a search engine or document database. Return only the rewritten question, "
        "no explanation.\n\nOriginal: " + query
    )
    try:
        from langchain_core.messages import HumanMessage
        response = chat_model.invoke([HumanMessage(content=prompt)])
        rewritten = response.content.strip().strip('"').strip("'")
        if rewritten:
            return rewritten
    except Exception:
        logging.exception("Query rewriting failed.")
    return query
