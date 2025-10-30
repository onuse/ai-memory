# AI Memory - LLM with Graph Memory

A chatbot system that maintains conversational memory using Neo4j graph database and a local LLM (GPT-OSS:120b via llama.cpp).

## Overview

This MVP implements the core functionality of an agentic memory system:
- Chat with a local LLM with persistent memory across sessions
- Automatic extraction of entities and relationships from conversations
- Storage in Neo4j graph database with confidence scoring
- Context retrieval and injection for relevant memories

## Architecture

```
User Message → Retrieve Context → LLM Response → User
                                       ↓
                            Extract Entities (async)
                                       ↓
                            Convert to Cypher
                                       ↓
                            Store in Neo4j
```

## Prerequisites

- **Python 3.11+**
- **llama.cpp server** running GPT-OSS:120b on port 8080
- **Neo4j 5.x** database running on bolt://localhost:7687

## Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd ai-memory
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Initialize Neo4j schema:
```bash
python scripts/setup_neo4j.py
```

6. Run the server:
```bash
uvicorn src.main:app --reload --port 3000
```

## Usage

Send a POST request to `/chat`:
```bash
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hi, my name is Jonas and I live in Stockholm"}'
```

The system will:
1. Retrieve relevant memories from past conversations
2. Generate a response using the LLM
3. Extract entities (Person: Jonas, Place: Stockholm)
4. Store them in Neo4j for future context

## Project Structure

```
ai-memory/
├── src/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── models.py            # Pydantic data models
│   ├── llm_client.py        # LLM API wrapper
│   ├── neo4j_client.py      # Neo4j driver wrapper
│   ├── memory_service.py    # Memory extraction, storage, retrieval
│   └── prompts.py           # System prompts
├── tests/                   # Unit and integration tests
├── scripts/                 # Setup and utility scripts
├── docs/                    # Architecture documentation
└── requirements.txt         # Python dependencies
```

## Entity Types (MVP)

- **Person** - People mentioned in conversations
- **Place** - Locations, cities, countries
- **Event** - Things that happened or will happen
- **Concept** - Ideas, topics, subjects
- **Preference** - User likes, dislikes, preferences

## Next Steps (Beyond MVP)

- Two-pass extraction (staging area + validation)
- Ensemble voting for high-stakes information
- Scheduled maintenance jobs (deduplication, pruning)
- Dynamic schema evolution
- Monitoring dashboard

## Documentation

See `docs/` folder for detailed architecture documentation.
