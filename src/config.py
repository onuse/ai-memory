"""Configuration management for AI Memory system."""
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Server Configuration
    llama_cpp_url: str = "http://localhost:8080"
    llama_cpp_model: str = "gpt-oss-120b"
    llama_cpp_timeout: int = 120

    # Neo4j Configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"

    # Application Configuration
    app_host: str = "0.0.0.0"
    app_port: int = 3000
    log_level: str = "INFO"

    # Memory Configuration
    min_confidence_threshold: float = 0.5
    max_context_memories: int = 10
    extraction_enabled: bool = True

    # Conversation Configuration
    conversation_reasoning_level: Literal["low", "medium", "high"] = "low"
    extraction_reasoning_level: Literal["low", "medium", "high"] = "medium"

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
