"""Memory service for extraction, storage, and retrieval of conversation memories."""
import logging
import hashlib
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.models import (
    Entity,
    Relationship,
    MemoryExtraction,
    ConversationTurn,
    RetrievedMemory,
    EntityType,
)
from src.llm_client import LLMClient, get_llm_client
from src.neo4j_client import Neo4jClient, get_neo4j_client
from src.prompts import get_extraction_prompt
from src.config import settings

logger = logging.getLogger(__name__)


class MemoryService:
    """Handles all memory-related operations."""

    def __init__(self):
        self.llm_client: LLMClient = get_llm_client()
        self.neo4j_client: Neo4jClient = get_neo4j_client()
        self.min_confidence = settings.min_confidence_threshold

    # === EXTRACTION ===

    async def extract_memories(
        self, user_message: str, assistant_response: str
    ) -> MemoryExtraction:
        """
        Extract entities and relationships from a conversation turn.

        Uses LLM to analyze conversation and output structured JSON.
        """
        logger.info("Extracting memories from conversation turn")

        try:
            # Build extraction prompt
            prompt = get_extraction_prompt(user_message, assistant_response)

            # Get structured JSON from LLM
            result = await self.llm_client.extract_json(
                messages=[{"role": "user", "content": prompt}],
                reasoning_level=settings.extraction_reasoning_level,
                temperature=0.3,
            )

            # Parse entities
            entities = []
            for entity_data in result.get("entities", []):
                try:
                    entity = Entity(
                        type=EntityType(entity_data["type"]),
                        name=entity_data["name"],
                        properties=entity_data.get("properties", {}),
                        confidence=entity_data["confidence"],
                    )
                    entities.append(entity)
                except Exception as e:
                    logger.warning(f"Failed to parse entity: {entity_data} - {str(e)}")

            # Parse relationships
            relationships = []
            for rel_data in result.get("relationships", []):
                try:
                    relationship = Relationship(
                        source=rel_data["source"],
                        target=rel_data["target"],
                        type=rel_data["type"],
                        properties=rel_data.get("properties", {}),
                        confidence=rel_data["confidence"],
                    )
                    relationships.append(relationship)
                except Exception as e:
                    logger.warning(
                        f"Failed to parse relationship: {rel_data} - {str(e)}"
                    )

            logger.info(
                f"Extracted {len(entities)} entities and {len(relationships)} relationships"
            )

            return MemoryExtraction(
                entities=entities,
                relationships=relationships,
                conversation_context=f"User: {user_message[:100]}...",
            )

        except Exception as e:
            logger.error(f"Memory extraction failed: {str(e)}")
            # Return empty extraction on failure
            return MemoryExtraction(entities=[], relationships=[])

    # === STORAGE ===

    def _generate_entity_id(self, entity_type: EntityType, name: str) -> str:
        """Generate deterministic ID for entity."""
        # Create a stable ID based on type and name
        content = f"{entity_type.value}:{name.lower().strip()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def store_memories(
        self,
        extraction: MemoryExtraction,
        conversation_id: str,
        user_message: str,
        assistant_response: str,
    ) -> int:
        """
        Store extracted memories in Neo4j.

        Returns number of memories stored.
        """
        logger.info("Storing memories in Neo4j")

        stored_count = 0
        entity_id_map: Dict[str, str] = {}  # name -> id mapping

        try:
            # Step 1: Store entities
            for entity in extraction.entities:
                # Skip low confidence entities
                if entity.confidence < self.min_confidence:
                    logger.debug(
                        f"Skipping low confidence entity: {entity.name} ({entity.confidence})"
                    )
                    continue

                entity_id = self._generate_entity_id(entity.type, entity.name)
                entity_id_map[entity.name] = entity_id

                await self.neo4j_client.create_or_update_entity(
                    entity_id=entity_id,
                    entity_type=entity.type,
                    name=entity.name,
                    properties=entity.properties,
                    confidence=entity.confidence,
                )

                stored_count += 1

            # Step 2: Store relationships
            for relationship in extraction.relationships:
                # Skip low confidence relationships
                if relationship.confidence < self.min_confidence:
                    logger.debug(
                        f"Skipping low confidence relationship: {relationship.source} -> {relationship.target}"
                    )
                    continue

                # Get entity IDs (or generate if not already in map)
                source_id = entity_id_map.get(relationship.source)
                target_id = entity_id_map.get(relationship.target)

                # If entities not in this extraction, try to find them in database
                if not source_id or not target_id:
                    # Try to find entities by name
                    names_to_find = []
                    if not source_id:
                        names_to_find.append(relationship.source)
                    if not target_id:
                        names_to_find.append(relationship.target)

                    found_entities = await self.neo4j_client.find_entities_by_names(
                        names_to_find
                    )

                    for entity in found_entities:
                        entity_id_map[entity["name"]] = entity["id"]

                    source_id = entity_id_map.get(relationship.source)
                    target_id = entity_id_map.get(relationship.target)

                # Only create relationship if both entities exist
                if source_id and target_id:
                    await self.neo4j_client.create_relationship(
                        source_id=source_id,
                        target_id=target_id,
                        rel_type=relationship.type,
                        properties=relationship.properties,
                        confidence=relationship.confidence,
                    )
                    stored_count += 1
                else:
                    logger.warning(
                        f"Skipping relationship (missing entities): {relationship.source} -> {relationship.target}"
                    )

            # Step 3: Create conversation node
            if entity_id_map:
                await self.neo4j_client.create_conversation_node(
                    conversation_id=conversation_id,
                    user_message=user_message,
                    assistant_response=assistant_response,
                    entity_ids=list(entity_id_map.values()),
                )

            logger.info(f"Stored {stored_count} memories in Neo4j")
            return stored_count

        except Exception as e:
            logger.error(f"Failed to store memories: {str(e)}")
            return stored_count

    # === RETRIEVAL ===

    async def retrieve_relevant_memories(
        self, user_message: str, max_memories: int = None
    ) -> List[str]:
        """
        Retrieve relevant memories for a user message.

        Returns list of formatted memory strings for context injection.
        """
        if max_memories is None:
            max_memories = settings.max_context_memories

        logger.info(f"Retrieving relevant memories for: {user_message[:50]}...")

        try:
            # Strategy 1: Extract entity names from user message
            entity_names = self._extract_entity_names(user_message)

            memories: List[Dict[str, Any]] = []

            # Strategy 2: Find mentioned entities
            if entity_names:
                found_entities = await self.neo4j_client.find_entities_by_names(
                    entity_names
                )
                memories.extend(found_entities)

            # Strategy 3: Keyword search
            # Extract keywords (simple approach: words > 3 chars, not common words)
            keywords = self._extract_keywords(user_message)
            for keyword in keywords[:3]:  # Top 3 keywords
                search_results = await self.neo4j_client.search_entities(
                    search_term=keyword, limit=3
                )
                memories.extend(search_results)

            # Strategy 4: Get recent entities if not enough memories
            if len(memories) < max_memories:
                recent = await self.neo4j_client.get_recent_entities(
                    limit=max_memories - len(memories),
                    min_confidence=self.min_confidence,
                )
                memories.extend(recent)

            # Deduplicate by entity ID
            seen_ids = set()
            unique_memories = []
            for memory in memories:
                if memory["id"] not in seen_ids:
                    seen_ids.add(memory["id"])
                    unique_memories.append(memory)

            # Sort by relevance (recent mentions + high confidence)
            unique_memories.sort(
                key=lambda m: (
                    m.get("last_mentioned", datetime.min),
                    m.get("confidence", 0),
                ),
                reverse=True,
            )

            # Format for prompt injection
            formatted_memories = []
            for memory in unique_memories[:max_memories]:
                formatted = self._format_memory(memory)
                formatted_memories.append(formatted)

            logger.info(f"Retrieved {len(formatted_memories)} relevant memories")
            return formatted_memories

        except Exception as e:
            logger.error(f"Failed to retrieve memories: {str(e)}")
            return []

    def _extract_entity_names(self, text: str) -> List[str]:
        """Extract potential entity names from text (simple heuristic)."""
        # Look for capitalized words/phrases
        pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
        matches = re.findall(pattern, text)
        return list(set(matches))  # Deduplicate

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text (simple approach)."""
        # Common words to ignore
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "this",
            "that",
            "these",
            "those",
            "what",
            "when",
            "where",
            "why",
            "how",
            "can",
            "could",
            "will",
            "would",
            "should",
            "may",
            "might",
            "must",
        }

        words = re.findall(r"\b[a-z]{4,}\b", text.lower())
        keywords = [w for w in words if w not in stop_words]
        return keywords

    def _format_memory(self, entity: Dict[str, Any]) -> str:
        """Format an entity for prompt injection."""
        name = entity.get("name", "Unknown")
        entity_type = entity.get("type", "Entity")
        confidence = entity.get("confidence", 0)
        mention_count = entity.get("mention_count", 1)

        # Build formatted string
        parts = [f"{name} ({entity_type})"]

        # Add properties if available
        properties = []
        for key, value in entity.items():
            if key not in [
                "id",
                "name",
                "type",
                "confidence",
                "created_at",
                "last_mentioned",
                "mention_count",
            ]:
                properties.append(f"{key}: {value}")

        if properties:
            parts.append(" - " + ", ".join(properties[:3]))  # Top 3 properties

        # Add mention count if significant
        if mention_count > 2:
            parts.append(f" [mentioned {mention_count} times]")

        return "".join(parts)

    # === ASYNC EXTRACTION ===

    async def extract_and_store_async(
        self, user_message: str, assistant_response: str, conversation_id: str
    ):
        """
        Extract and store memories asynchronously (non-blocking).

        This is called after responding to the user.
        """
        try:
            logger.info(f"Starting async memory extraction for {conversation_id}")

            # Extract memories
            extraction = await self.extract_memories(user_message, assistant_response)

            # Store memories
            if extraction.entities or extraction.relationships:
                stored_count = await self.store_memories(
                    extraction=extraction,
                    conversation_id=conversation_id,
                    user_message=user_message,
                    assistant_response=assistant_response,
                )

                logger.info(
                    f"Async extraction complete for {conversation_id}: {stored_count} memories stored"
                )
            else:
                logger.info(
                    f"No memories extracted for {conversation_id}"
                )

        except Exception as e:
            logger.error(
                f"Async memory extraction failed for {conversation_id}: {str(e)}"
            )
            # Don't propagate exception - this is async and shouldn't affect user


# Global memory service instance
_memory_service: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    """Get or create global memory service instance."""
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service
