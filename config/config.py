import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="config/.env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen-qwq-32b",
    "mixtral-8x7b-32768",
]

SMALL_MODEL = "llama-3.1-8b-instant"
