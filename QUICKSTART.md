## Quick Start Guide

### Prerequisites

Before you begin, make sure you have:

1. **Python 3.11+** installed
2. **Neo4j 5.x** running locally on `bolt://localhost:7687`
3. **llama.cpp server** running with GPT-OSS:120b on `http://localhost:8080`

### Step 1: Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
# Copy example environment file
copy .env.example .env  # Windows
cp .env.example .env    # Linux/Mac

# Edit .env with your settings
# Update Neo4j password if needed
```

### Step 3: Initialize Neo4j

```bash
# Run schema initialization script
python scripts/setup_neo4j.py
```

You should see:
```
INFO - Starting Neo4j schema initialization...
INFO - Connected to Neo4j at bolt://localhost:7687
INFO - Schema initialization complete!
```

### Step 4: Start the Server

```bash
# Run the FastAPI application
python src/main.py

# Or use uvicorn directly
uvicorn src.main:app --reload --port 3000
```

The server will start on `http://localhost:3000`

### Step 5: Test the System

Open a new terminal and run the test client:

```bash
python scripts/test_client.py
```

This will:
1. Check system health
2. Send a series of test messages
3. Verify memory extraction and recall
4. Display database statistics

### Expected Output

```
============================================================
AI Memory - Test Conversation
============================================================

[1] Checking system health...
✓ System healthy: {...}

[2] First message: Introducing Jonas...
User: Hi! My name is Jonas. I live in Stockholm...
Assistant: [Response with greeting]
Conversation ID: [UUID]
Memories in context: 0

[3] Waiting for memory extraction...
Database stats: {
  "entities": 3,
  "relationships": 2,
  "conversations": 1
}

[4] Second message: Asking about location...
User: What city did I mention I live in?
Assistant: You mentioned you live in Stockholm!
Memories in context: 3
✓ Memory recall successful! The system remembered Stockholm.

...
```

### Testing Manually

You can also test with curl:

```bash
# Send a chat message
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hi, my name is Jonas and I live in Stockholm"}'

# Check statistics
curl http://localhost:3000/stats

# Health check
curl http://localhost:3000/health
```

### Common Issues

**Issue: Neo4j connection failed**
- Make sure Neo4j is running: Check Neo4j Desktop or `neo4j status`
- Verify credentials in `.env` file match your Neo4j setup

**Issue: LLM server connection failed**
- Make sure llama.cpp server is running on port 8080
- Test: `curl http://localhost:8080/health`
- Check `LLAMA_CPP_URL` in `.env`

**Issue: Memory extraction not working**
- Check logs for extraction errors
- Verify `EXTRACTION_ENABLED=true` in `.env`
- Ensure LLM supports JSON mode / structured output
- May need to adjust reasoning levels in `.env`

**Issue: Slow responses**
- Reduce `MAX_CONTEXT_MEMORIES` in `.env` (default: 10)
- Set `CONVERSATION_REASONING_LEVEL=low` for faster chat
- Memory extraction happens async, won't block responses

### Next Steps

1. **Explore the database**: Open Neo4j Browser at `http://localhost:7474`
   - Query: `MATCH (n:Entity) RETURN n LIMIT 25`
   - Query: `MATCH (n:Entity)-[r]->(m:Entity) RETURN n, r, m LIMIT 50`

2. **Adjust confidence threshold**: Edit `MIN_CONFIDENCE_THRESHOLD` in `.env`
   - Lower = more memories stored (may include noise)
   - Higher = fewer, higher quality memories

3. **Monitor extraction quality**: Check logs for extraction results
   - Look for "Extracted X entities and Y relationships"
   - Review what's being stored in Neo4j

4. **Build a proper UI**: Create a web interface or CLI chat tool
   - Use the `/chat` endpoint
   - Maintain `conversation_id` across messages
   - Display `memories_extracted` count to show system is learning

5. **Experiment with prompts**: Edit `src/prompts.py`
   - Customize extraction prompt for your use case
   - Adjust conversation system prompt
   - Add few-shot examples for better extraction

### Development

Run tests:
```bash
pytest tests/ -v
```

Format code:
```bash
black src/ tests/
```

Type checking:
```bash
mypy src/
```

### Architecture

```
User Message
    ↓
Retrieve Memories (Neo4j search)
    ↓
Inject Context (build prompt)
    ↓
LLM Response
    ↓
Return to User ← [IMMEDIATE RESPONSE]
    ↓
Extract Entities (async, background)
    ↓
Store in Neo4j
```

The system is **non-blocking**: memory extraction happens in the background and won't slow down responses to the user.

### Monitoring

Key metrics to watch:
- Number of entities/relationships in Neo4j (`/stats` endpoint)
- Memory extraction success rate (check logs)
- Response latency (should be <2s with memory retrieval)
- Confidence scores of stored memories

### Tips

1. **Start simple**: Test with clear, factual statements first
2. **Be explicit**: "My name is X" works better than "I'm X"
3. **Wait for extraction**: Give it 5-10 seconds between messages for extraction to complete
4. **Check Neo4j**: Browse the graph to see what's being stored
5. **Iterate on prompts**: Adjust extraction prompt based on what you see

### Support

- Check logs for detailed error messages
- Review architecture doc: `docs/neo4j-memory-chatbot-architecture.md`
- Check project README: `README.md`
