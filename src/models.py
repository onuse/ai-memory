"""Pydantic data models for AI Memory system."""
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class EntityType(str, Enum):
    """Fixed entity types for MVP."""
    PERSON = "Person"
    PLACE = "Place"
    EVENT = "Event"
    CONCEPT = "Concept"
    PREFERENCE = "Preference"


class RelationshipType(str, Enum):
    """Common relationship types."""
    KNOWS = "KNOWS"
    LIVES_IN = "LIVES_IN"
    LOCATED_IN = "LOCATED_IN"
    ATTENDED = "ATTENDED"
    PREFERS = "PREFERS"
    DISLIKES = "DISLIKES"
    RELATED_TO = "RELATED_TO"
    MENTIONED_IN = "MENTIONED_IN"


# === Request/Response Models ===

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for continuity")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="LLM response")
    conversation_id: str = Field(..., description="Conversation ID")
    memories_extracted: Optional[int] = Field(None, description="Number of memories extracted")


# === Memory Models ===

class Entity(BaseModel):
    """Represents an extracted entity."""
    type: EntityType = Field(..., description="Entity type")
    name: str = Field(..., description="Entity name")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class Relationship(BaseModel):
    """Represents a relationship between entities."""
    source: str = Field(..., description="Source entity name")
    target: str = Field(..., description="Target entity name")
    type: str = Field(..., description="Relationship type")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class MemoryExtraction(BaseModel):
    """Result of memory extraction from a conversation turn."""
    entities: List[Entity] = Field(default_factory=list, description="Extracted entities")
    relationships: List[Relationship] = Field(default_factory=list, description="Extracted relationships")
    conversation_context: Optional[str] = Field(None, description="Context snippet from conversation")


# === Conversation Models ===

class Message(BaseModel):
    """A single message in a conversation."""
    role: Literal["user", "assistant", "system"] = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")


class ConversationTurn(BaseModel):
    """A complete conversation turn (user + assistant)."""
    user_message: str = Field(..., description="User's message")
    assistant_response: str = Field(..., description="Assistant's response")
    timestamp: datetime = Field(default_factory=datetime.now, description="Turn timestamp")
    conversation_id: str = Field(..., description="Conversation identifier")


# === Memory Storage Models ===

class StoredEntity(BaseModel):
    """Entity stored in Neo4j with metadata."""
    id: str = Field(..., description="Unique entity ID")
    type: EntityType = Field(..., description="Entity type")
    name: str = Field(..., description="Entity name")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Entity properties")
    confidence: float = Field(..., description="Confidence score")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_mentioned: datetime = Field(..., description="Last mention timestamp")
    mention_count: int = Field(default=1, description="Number of mentions")


class RetrievedMemory(BaseModel):
    """Memory retrieved from Neo4j for context."""
    entity_id: str = Field(..., description="Entity ID")
    entity_type: EntityType = Field(..., description="Entity type")
    name: str = Field(..., description="Entity name")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Entity properties")
    relationships: List[str] = Field(default_factory=list, description="Related entities")
    relevance_score: float = Field(..., description="Relevance to current context")
    last_mentioned: Optional[datetime] = Field(None, description="When last mentioned")
