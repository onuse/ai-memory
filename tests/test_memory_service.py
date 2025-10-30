"""Tests for memory service."""
import pytest
from src.models import Entity, Relationship, MemoryExtraction, EntityType


def test_entity_creation():
    """Test creating an entity."""
    entity = Entity(
        type=EntityType.PERSON,
        name="Test Person",
        properties={"occupation": "engineer"},
        confidence=0.9,
    )

    assert entity.type == EntityType.PERSON
    assert entity.name == "Test Person"
    assert entity.confidence == 0.9
    assert entity.properties["occupation"] == "engineer"


def test_relationship_creation():
    """Test creating a relationship."""
    rel = Relationship(
        source="Person A",
        target="Person B",
        type="KNOWS",
        properties={},
        confidence=0.85,
    )

    assert rel.source == "Person A"
    assert rel.target == "Person B"
    assert rel.type == "KNOWS"
    assert rel.confidence == 0.85


def test_memory_extraction():
    """Test memory extraction model."""
    entities = [
        Entity(
            type=EntityType.PERSON,
            name="Jonas",
            properties={},
            confidence=0.95,
        )
    ]

    relationships = [
        Relationship(
            source="Jonas",
            target="Stockholm",
            type="LIVES_IN",
            properties={},
            confidence=0.9,
        )
    ]

    extraction = MemoryExtraction(
        entities=entities,
        relationships=relationships,
        conversation_context="Test context",
    )

    assert len(extraction.entities) == 1
    assert len(extraction.relationships) == 1
    assert extraction.entities[0].name == "Jonas"


def test_entity_types():
    """Test that all entity types are valid."""
    types = [
        EntityType.PERSON,
        EntityType.PLACE,
        EntityType.EVENT,
        EntityType.CONCEPT,
        EntityType.PREFERENCE,
    ]

    for entity_type in types:
        entity = Entity(
            type=entity_type,
            name=f"Test {entity_type.value}",
            properties={},
            confidence=0.8,
        )
        assert entity.type == entity_type


@pytest.mark.asyncio
async def test_llm_client_import():
    """Test that LLM client can be imported."""
    from src.llm_client import LLMClient

    client = LLMClient()
    assert client.base_url is not None
    assert client.model is not None


@pytest.mark.asyncio
async def test_neo4j_client_import():
    """Test that Neo4j client can be imported."""
    from src.neo4j_client import Neo4jClient

    client = Neo4jClient()
    assert client.uri is not None
    assert client.user is not None


@pytest.mark.asyncio
async def test_memory_service_import():
    """Test that memory service can be imported."""
    from src.memory_service import MemoryService

    service = MemoryService()
    assert service.min_confidence >= 0.0
    assert service.min_confidence <= 1.0
