import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# Read .env file
if not load_dotenv(verbose=True, override=True):
    raise Exception("No environment variable has been set from .env.")


# Azure keys
AZURE_OPENAI_API_VERSION: Final[str] = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_API_KEY: Final[str] = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT: Final[str] = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT: Final[str] = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# OpenAI key for embedding model
OPENAI_API_KEY: Final[str] = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL: Final[str] = "text-embedding-ada-002"

# FAISS Vector DB Path
VECTOR_DB: Final[Path] = Path("db")

# Document Storage
DOCS_DIR: Final[Path] = Path("./docs/pdf")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Text split settings
CHUNK_SIZE: Final[int] = 1000
CHUNK_OVERLAP: Final[int] = 50
BATCH_SIZE: Final[int] = 64

# Prompt settings
SYSTEM_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
