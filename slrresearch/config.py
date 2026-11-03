import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    provider: str = os.getenv("LLM_PROVIDER", "ollama").lower()
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def build_model(settings: Settings):
    if settings.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
    if settings.provider == "groq":
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY manquante dans .env")
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0,
        )
    raise ValueError("LLM_PROVIDER doit être 'ollama' ou 'groq'")


def build_embeddings(settings: Settings):
    """Construit le modèle local utilisé par le retriever sémantique du TD05."""
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )
