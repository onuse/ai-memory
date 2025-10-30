"""FastAPI application for AI Memory chatbot."""
import logging
import uuid
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.models import ChatRequest, ChatResponse
from src.llm_client import get_llm_client, close_llm_client
from src.neo4j_client import get_neo4j_client, close_neo4j_client
from src.memory_service import get_memory_service
from src.prompts import get_conversation_prompt

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app."""
    # Startup
    logger.info("Starting AI Memory application...")

    # Initialize Neo4j
    neo4j_client = get_neo4j_client()
    await neo4j_client.connect()
    await neo4j_client.initialize_schema()

    logger.info(f"Application started on {settings.app_host}:{settings.app_port}")

    yield

    # Shutdown
    logger.info("Shutting down AI Memory application...")
    await close_llm_client()
    await close_neo4j_client()
    logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="AI Memory",
    description="LLM chatbot with graph database memory",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "AI Memory",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check Neo4j connection
        neo4j_client = get_neo4j_client()
        stats = await neo4j_client.get_stats()

        return {
            "status": "healthy",
            "neo4j": "connected",
            "database_stats": stats,
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Chat endpoint with memory retrieval and extraction.

    Flow:
    1. Retrieve relevant memories from Neo4j
    2. Inject memories into prompt as context
    3. Generate response from LLM
    4. Return response to user immediately
    5. Extract and store memories asynchronously (background task)
    """
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())

        logger.info(f"Chat request: {conversation_id}")

        # Get clients
        llm_client = get_llm_client()
        memory_service = get_memory_service()

        # Step 1: Retrieve relevant memories
        memories = await memory_service.retrieve_relevant_memories(
            request.message, max_memories=settings.max_context_memories
        )

        # Step 2: Build prompt with memory context
        system_prompt = get_conversation_prompt(memories)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message},
        ]

        # Step 3: Get response from LLM
        response = await llm_client.chat_completion(
            messages=messages,
            reasoning_level=settings.conversation_reasoning_level,
            temperature=0.7,
        )

        response_text = response["content"]

        logger.info(f"Generated response for {conversation_id} (with {len(memories)} memories)")

        # Step 4: Schedule async memory extraction in background
        if settings.extraction_enabled:
            background_tasks.add_task(
                memory_service.extract_and_store_async,
                user_message=request.message,
                assistant_response=response_text,
                conversation_id=conversation_id,
            )

        return ChatResponse(
            response=response_text,
            conversation_id=conversation_id,
            memories_extracted=len(memories) if memories else 0,
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get database statistics."""
    try:
        neo4j_client = get_neo4j_client()
        stats = await neo4j_client.get_stats()

        return {
            "database": stats,
            "extraction_enabled": settings.extraction_enabled,
            "min_confidence": settings.min_confidence_threshold,
        }
    except Exception as e:
        logger.error(f"Stats endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
    )
