"""Script to initialize Neo4j database schema."""
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neo4j_client import get_neo4j_client
from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


async def main():
    """Initialize Neo4j schema."""
    logger.info("Starting Neo4j schema initialization...")
    logger.info(f"Connecting to: {settings.neo4j_uri}")

    client = get_neo4j_client()

    try:
        await client.connect()
        await client.initialize_schema()

        # Get stats to verify connection
        stats = await client.get_stats()
        logger.info(f"Current database stats: {stats}")

        logger.info("Schema initialization complete!")

    except Exception as e:
        logger.error(f"Failed to initialize schema: {str(e)}")
        raise
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
