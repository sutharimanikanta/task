from utils.rag import get_context
from utils.router import decide_sources, decompose_query
from utils.web_search import web_search


def build_context(query: str, model, retriever, has_docs: bool) -> tuple:
    """
    1. Decomposes query into sub-questions.
    2. For each sub-question, decides which sources to use.
    3. Tries RAG first, falls back to web search ONLY if RAG returns nothing.
    4. Returns merged context string and a detailed trail list.

    trail is a list of dicts:
      { "sub_query": str, "source": str, "found": bool }
    """
    sub_queries = decompose_query(query, model)
    all_contexts = []
    trail = []

    for sq in sub_queries:
        sources = decide_sources(sq, has_docs)
        found = False

        for source in sources:
            if found:  # Already found an answer, skip remaining sources
                break

            if source == "rag" and retriever is not None:
                context, doc_source = get_context(retriever, sq)
                if context.strip():
                    all_contexts.append(f"[from document: {doc_source}]\n{context}")
                    trail.append(
                        {
                            "sub_query": sq,
                            "source": f"RAG ({doc_source})",
                            "found": True,
                        }
                    )
                    found = True
                else:
                    trail.append({"sub_query": sq, "source": "RAG", "found": False})

            elif source == "web":  # Only try web if RAG didn't find anything
                context, urls = web_search(sq)
                if context.strip():
                    all_contexts.append(f"[from web: {urls}]\n{context}")
                    trail.append(
                        {"sub_query": sq, "source": f"Web ({urls})", "found": True}
                    )
                    found = True
                else:
                    trail.append({"sub_query": sq, "source": "Web", "found": False})

        if not found:
            trail.append({"sub_query": sq, "source": "none", "found": False})

    merged = "\n\n---\n\n".join(all_contexts)
    return merged, sub_queries, trail
