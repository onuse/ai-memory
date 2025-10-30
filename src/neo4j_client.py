"""Neo4j client for graph database operations."""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from src.config import settings
from src.models import EntityType, StoredEntity, RetrievedMemory

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Client for Neo4j graph database operations."""

    def __init__(self):
        self.uri = settings.neo4j_uri
        self.user = settings.neo4j_user
        self.password = settings.neo4j_password
        self.database = settings.neo4j_database
        self.driver: Optional[AsyncDriver] = None

    async def connect(self):
        """Establish connection to Neo4j."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # Verify connectivity
            await self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

    async def close(self):
        """Close Neo4j connection."""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j connection closed")

    async def initialize_schema(self):
        """Initialize Neo4j schema with constraints and indexes."""
        logger.info("Initializing Neo4j schema...")

        async with self.driver.session(database=self.database) as session:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT conversation_id IF NOT EXISTS FOR (c:Conversation) REQUIRE c.id IS UNIQUE",
            ]

            for constraint in constraints:
                try:
                    await session.run(constraint)
                    logger.debug(f"Created constraint: {constraint}")
                except Exception as e:
                    # Constraint might already exist
                    logger.debug(f"Constraint creation note: {str(e)}")

            # Create indexes for performance
            indexes = [
                "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
                "CREATE INDEX entity_created IF NOT EXISTS FOR (e:Entity) ON (e.created_at)",
                "CREATE INDEX entity_mentioned IF NOT EXISTS FOR (e:Entity) ON (e.last_mentioned)",
                "CREATE INDEX conversation_timestamp IF NOT EXISTS FOR (c:Conversation) ON (c.timestamp)",
            ]

            for index in indexes:
                try:
                    await session.run(index)
                    logger.debug(f"Created index: {index}")
                except Exception as e:
                    logger.debug(f"Index creation note: {str(e)}")

            logger.info("Schema initialization complete")

    async def create_or_update_entity(
        self,
        entity_id: str,
        entity_type: EntityType,
        name: str,
        properties: Dict[str, Any],
        confidence: float,
    ) -> Dict[str, Any]:
        """
        Create a new entity or update existing one.

        Uses MERGE for idempotency - won't create duplicates.
        """
        async with self.driver.session(database=self.database) as session:
            query = """
            MERGE (e:Entity {id: $id})
            ON CREATE SET
                e.type = $type,
                e.name = $name,
                e.created_at = datetime(),
                e.last_mentioned = datetime(),
                e.mention_count = 1,
                e.confidence = $confidence
            ON MATCH SET
                e.last_mentioned = datetime(),
                e.mention_count = e.mention_count + 1,
                e.confidence = ($confidence + e.confidence) / 2
            SET e += $properties
            RETURN e
            """

            params = {
                "id": entity_id,
                "type": entity_type.value,
                "name": name,
                "properties": properties,
                "confidence": confidence,
            }

            result = await session.run(query, params)
            record = await result.single()

            if record:
                logger.debug(f"Created/updated entity: {name} ({entity_type.value})")
                return dict(record["e"])
            else:
                raise Exception(f"Failed to create/update entity: {name}")

    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Dict[str, Any],
        confidence: float,
    ):
        """Create a relationship between two entities."""
        async with self.driver.session(database=self.database) as session:
            # Sanitize relationship type (must be valid Cypher identifier)
            rel_type = rel_type.upper().replace(" ", "_")

            query = f"""
            MATCH (source:Entity {{id: $source_id}})
            MATCH (target:Entity {{id: $target_id}})
            CREATE (source)-[r:{rel_type}]->(target)
            SET r.created_at = datetime(),
                r.confidence = $confidence,
                r += $properties
            RETURN r
            """

            params = {
                "source_id": source_id,
                "target_id": target_id,
                "confidence": confidence,
                "properties": properties,
            }

            result = await session.run(query, params)
            record = await result.single()

            if record:
                logger.debug(f"Created relationship: {source_id} -{rel_type}-> {target_id}")
            else:
                logger.warning(f"Failed to create relationship: {source_id} -{rel_type}-> {target_id}")

    async def find_entities_by_names(self, names: List[str]) -> List[Dict[str, Any]]:
        """Find entities by their names."""
        async with self.driver.session(database=self.database) as session:
            query = """
            MATCH (e:Entity)
            WHERE e.name IN $names
            RETURN e
            ORDER BY e.last_mentioned DESC
            """

            result = await session.run(query, {"names": names})
            records = await result.data()

            entities = [record["e"] for record in records]
            logger.debug(f"Found {len(entities)} entities for {len(names)} names")
            return entities

    async def search_entities(
        self,
        search_term: str,
        entity_type: Optional[EntityType] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search entities by name (case-insensitive partial match).
        """
        async with self.driver.session(database=self.database) as session:
            if entity_type:
                query = """
                MATCH (e:Entity)
                WHERE e.type = $entity_type AND toLower(e.name) CONTAINS toLower($search_term)
                RETURN e
                ORDER BY e.mention_count DESC, e.last_mentioned DESC
                LIMIT $limit
                """
                params = {
                    "search_term": search_term,
                    "entity_type": entity_type.value,
                    "limit": limit,
                }
            else:
                query = """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($search_term)
                RETURN e
                ORDER BY e.mention_count DESC, e.last_mentioned DESC
                LIMIT $limit
                """
                params = {"search_term": search_term, "limit": limit}

            result = await session.run(query, params)
            records = await result.data()

            entities = [record["e"] for record in records]
            logger.debug(f"Found {len(entities)} entities matching '{search_term}'")
            return entities

    async def get_entity_with_relationships(
        self, entity_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get an entity and its immediate relationships."""
        async with self.driver.session(database=self.database) as session:
            query = """
            MATCH (e:Entity {id: $entity_id})
            OPTIONAL MATCH (e)-[r]-(neighbor:Entity)
            RETURN e,
                   collect({
                       type: type(r),
                       direction: CASE WHEN startNode(r) = e THEN 'outgoing' ELSE 'incoming' END,
                       neighbor: neighbor
                   }) as relationships
            """

            result = await session.run(query, {"entity_id": entity_id})
            record = await result.single()

            if record:
                return {
                    "entity": dict(record["e"]),
                    "relationships": [
                        rel for rel in record["relationships"] if rel["neighbor"] is not None
                    ],
                }
            return None

    async def get_recent_entities(
        self, limit: int = 20, min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Get recently mentioned entities above confidence threshold."""
        async with self.driver.session(database=self.database) as session:
            query = """
            MATCH (e:Entity)
            WHERE e.confidence >= $min_confidence
            RETURN e
            ORDER BY e.last_mentioned DESC
            LIMIT $limit
            """

            result = await session.run(
                query, {"limit": limit, "min_confidence": min_confidence}
            )
            records = await result.data()

            entities = [record["e"] for record in records]
            logger.debug(f"Retrieved {len(entities)} recent entities")
            return entities

    async def create_conversation_node(
        self,
        conversation_id: str,
        user_message: str,
        assistant_response: str,
        entity_ids: List[str],
    ):
        """Create a conversation node and link it to mentioned entities."""
        async with self.driver.session(database=self.database) as session:
            query = """
            CREATE (c:Conversation {
                id: $conversation_id,
                timestamp: datetime(),
                user_message: $user_message,
                assistant_response: $assistant_response
            })
            WITH c
            UNWIND $entity_ids AS entity_id
            MATCH (e:Entity {id: entity_id})
            CREATE (c)-[:MENTIONED]->(e)
            RETURN c
            """

            params = {
                "conversation_id": conversation_id,
                "user_message": user_message,
                "assistant_response": assistant_response,
                "entity_ids": entity_ids,
            }

            await session.run(query, params)
            logger.debug(f"Created conversation node: {conversation_id}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        async with self.driver.session(database=self.database) as session:
            query = """
            MATCH (e:Entity)
            WITH count(e) as entity_count
            MATCH ()-[r]->()
            WITH entity_count, count(r) as relationship_count
            MATCH (c:Conversation)
            RETURN entity_count, relationship_count, count(c) as conversation_count
            """

            result = await session.run(query)
            record = await result.single()

            if record:
                return {
                    "entities": record["entity_count"],
                    "relationships": record["relationship_count"],
                    "conversations": record["conversation_count"],
                }
            return {"entities": 0, "relationships": 0, "conversations": 0}


# Global Neo4j client instance
_neo4j_client: Optional[Neo4jClient] = None


def get_neo4j_client() -> Neo4jClient:
    """Get or create global Neo4j client instance."""
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient()
    return _neo4j_client


async def close_neo4j_client():
    """Close global Neo4j client."""
    global _neo4j_client
    if _neo4j_client is not None:
        await _neo4j_client.close()
        _neo4j_client = None
