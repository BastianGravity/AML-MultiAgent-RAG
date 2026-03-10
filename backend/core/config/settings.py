"""
Configuration settings for AML Multi-Agent RAG System.

Manages environment variables and application settings for:
- LLM API integration (OpenAI-compatible, including Groq)
- Vector database configuration
- Document processing parameters
- Model specifications
"""
from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application configuration settings with environment variable support.

    This class manages all configuration parameters for the AML Multi-Agent RAG
    system, automatically loading values from environment variables
    or .env file.
    """
    # API keys
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None

    # Optional OpenAI-compatible base URL (e.g. Groq endpoint)
    llm_api_base_url: Optional[str] = None

    # Vector DB configuration
    qdrant_url: str = "http://localhost:6333"

    # Document processing
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # LLM models
    llm_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # Collection names
    collection_name: str = "aml-documents"

    class Config:
        env_file = ".env"

    @property
    def llm_api_key(self) -> Optional[str]:
        """Return configured OpenAI key, or fall back to Groq key."""
        return self.OPENAI_API_KEY or self.GROQ_API_KEY

    @property
    def resolved_llm_api_base_url(self) -> Optional[str]:
        """Resolve base URL for OpenAI-compatible clients.

        If no explicit base URL is set and only a Groq key exists,
        default to Groq's OpenAI-compatible endpoint.
        """
        if self.llm_api_base_url:
            return self.llm_api_base_url
        if self.GROQ_API_KEY and not self.OPENAI_API_KEY:
            return "https://api.groq.com/openai/v1"
        return None


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()
