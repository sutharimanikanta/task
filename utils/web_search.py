import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import TAVILY_API_KEY
from tavily import TavilyClient


def web_search(query: str) -> tuple[str, str]:
    """
    Returns (context_text, source_urls).
    Both are empty strings on failure.
    """
    if not TAVILY_API_KEY:
        logging.warning("TAVILY_API_KEY not set.")
        return "", ""
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        res = client.search(query=query, max_results=3)
        results = res.get("results", [])
        if not results:
            return "", ""
        parts = []
        urls = []
        for item in results:
            parts.append(f"{item.get('title', '')}\n{item.get('content', '')}")
            urls.append(item.get("url", ""))
        return "\n\n".join(parts), " | ".join(urls)
    except Exception:
        logging.exception("Web search failed.")
        return "", ""
