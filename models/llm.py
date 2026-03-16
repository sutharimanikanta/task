import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import GROQ_API_KEY
from langchain_groq import ChatGroq


def get_model(model_name: str = "llama-3.3-70b-versatile"):
    if not GROQ_API_KEY:
        logging.warning("GROQ_API_KEY not set.")
        return None
    try:
        return ChatGroq(api_key=GROQ_API_KEY, model=model_name, temperature=0.2)
    except Exception:
        logging.exception("Failed to initialize model: %s", model_name)
        return None
