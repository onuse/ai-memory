# Neo4j Memory Chatbot - Architecture & Implementation Plan

## Executive Summary

This document specifies the architecture and implementation plan for an agentic chatbot system that maintains infinite conversational memory through autonomous graph database management. The system uses GPT-OSS:120b running locally via llama.cpp to analyze conversations, decide what information to store, and retrieve relevant context from a Neo4j graph database. The core innovation is giving the LLM autonomous control over memory decisions while implementing comprehensive validation to mitigate hallucination risks.

**Key Design Principle:** The LLM proposes memory operations autonomously, but a validation layer gates all actual storage to prevent hallucinated information from corrupting the knowledge graph.

## System Overview

### Core Concept

Traditional chatbots have limited memory - they can only "remember" what fits in their context window (typically the last few messages). This system creates the illusion of infinite memory by:

1. **Analyzing every conversation turn** to identify important information (entities, relationships, facts, preferences, events)
2. **Storing structured information in Neo4j** as nodes (entities like people, places, concepts) and relationships (connections between entities)
3. **Retrieving relevant context** from the graph when needed, injecting it into the LLM's prompt
4. **Refining the graph** through scheduled maintenance jobs that consolidate, deduplicate, and reorganize stored memories

The user experiences a chatbot that "remembers" everything from past conversations, even across sessions weeks or months apart.

### Critical Architectural Principles

#### 1. Dynamic Schema Evolution

**The system does NOT use a predefined schema.** Entity types and relationship types emerge organically based on what the user discusses. For a physics researcher, the graph might contain `Experiment`, `Apparatus`, `Measurement` nodes with `MEASURED_BY` and `CONTRADICTS` relationships. For someone discussing cooking, it would be `Recipe`, `Ingredient`, `Technique` with completely different relationships.

The LLM proposes both the entity content AND its type. Over time, the schema evolves:
- Early conversations: LLM freely creates entity types as needed
- After 50-100 conversations: Patterns emerge (most entities are a handful of types)
- Maintenance passes: Consolidate similar types ("Experiment" + "PhysicsExperiment" → "Experiment")
- Long-term: Schema stabilizes around user's actual topics of interest

This means:
- **No entity type constraints** in Neo4j (use generic `Entity` label or dynamic labels)
- **Relationship types are strings** without schema validation
- **Property keys are flexible** (different entity types have different properties)
- **Schema introspection** becomes a feature (query what types exist in YOUR graph)

#### 2. Two-Pass Memory Architecture

To balance conversational latency with memory quality, the system uses **two distinct memory passes**:

**Live Pass (Immediate, Lenient):**
- Triggers: After every conversation turn
- Reasoning: Low (fast, ~2-3 seconds)
- Validation: Syntax only (is JSON valid? Can we parse it?)
- Acceptance: Confidence > 0.3 (lenient threshold)
- Storage: "Staging area" in Neo4j (marked as `unverified: true`)
- Goal: Don't block conversation, capture everything quickly

**Maintenance Pass (Scheduled, Rigorous):**
- Triggers: After 30+ minutes of user inactivity
- Reasoning: High (thorough, can take 30+ seconds per memory)
- Validation: Full pipeline (ensemble voting, fact verification, conflict detection)
- Acceptance: Confidence > 0.6, must pass all validation stages
- Storage: Promotes staging memories to verified graph, rejects bad memories
- Goal: High-quality persistent memory without time pressure

**Why this works:**
- Users get instant responses (live pass is async and fast)
- Memory quality improves in background when system is idle
- Maintenance can take its time - no user waiting
- Failed validations don't create user-visible errors
- System "thinks" about its memories during downtime, like human sleep consolidation

**Downtime Detection:**
- Track last user message timestamp
- If `now() - last_message > 30 minutes`, enter maintenance mode
- Run maintenance pass on staging area (10-50 memories per session)
- Return to idle state when done or user sends new message
- Alternative: Fixed schedule (e.g., 3 AM daily) for comprehensive maintenance

### Architecture Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface                               │
│                    (CLI / Web / API Client)                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ HTTP/WebSocket
┌────────────────────────────────┴────────────────────────────────────┐
│                      Memory Application                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Conversation Orchestration Layer                            │   │
│  │  - User message handling                                     │   │
│  │  - Conversation history management                           │   │
│  │  - Response streaming                                        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Memory Management Layer                                     │   │
│  │  - Memory extraction (calls LLM to identify what to store)   │   │
│  │  - Memory validation (fact-checking, ensemble voting)        │   │
│  │  - Memory storage (executes approved Neo4j operations)       │   │
│  │  - Memory retrieval (finds relevant context for prompts)     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Validation Infrastructure                                   │   │
│  │  - Cypher syntax validation                                  │   │
│  │  - Fact verification (against existing graph + web search)   │   │
│  │  - Ensemble decision voting                                  │   │
│  │  - Hallucination monitoring (CoT analysis)                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Tool Execution Layer                                        │   │
│  │  - Neo4j operations (CREATE, MATCH, MERGE)                   │   │
│  │  - Web search integration (optional, for fact-checking)      │   │
│  │  - Schema introspection                                      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────┬───────────────────────────────────┬───────────────────────┘
          │                                   │
          │ OpenAI-compatible API             │ Neo4j Bolt Protocol
          │ (HTTP)                            │ (bolt://localhost:7687)
          │                                   │
┌─────────┴─────────────────┐   ┌─────────────┴──────────────────────┐
│   llama.cpp Server        │   │   Neo4j Database                   │
│                           │   │                                    │
│  GPT-OSS:120b GGUF       │   │   Graph Schema:                    │
│  Harmony format          │   │   - Entity nodes (Person, Place,   │
│  Port: 8080              │   │     Event, Concept, Preference)    │
│  Reasoning: medium/high  │   │   - Relationships (KNOWS,          │
│  for memory decisions    │   │     LOCATED_IN, HAPPENED_AT,       │
│                           │   │     PREFERS, MENTIONED_IN)         │
│                           │   │   - Conversation nodes (metadata)  │
│                           │   │   - Temporal indexing              │
└───────────────────────────┘   └────────────────────────────────────┘
```

### Technology Stack

**AI Server:**
- **llama.cpp** (latest version with GPT-OSS support)
- **GPT-OSS:120b GGUF Q4_K_M** (~45GB, runs on AMD Ryzen AI Max+ 395)
- **OpenAI-compatible API** exposed on port 8080

**Memory Application:**
- **Python 3.11+** (recommended for LangChain/LangGraph ecosystem)
- **FastAPI** or **Flask** for HTTP/WebSocket server
- **neo4j-driver** (official Python driver)
- **httpx** or **openai** library for LLM communication
- **pydantic** for data validation

**Database:**
- **Neo4j 5.x** (Community or Enterprise Edition)
- **APOC plugin** for advanced graph algorithms
- **Full-text search indexes** for semantic retrieval

**Optional Components:**
- **Tavily API** for web search and fact verification
- **Redis** for caching LLM responses and validation results
- **Grafana + Prometheus** for monitoring hallucination rates

## Detailed Architecture

### 1. Conversation Flow

#### Standard User Interaction (Two-Pass Memory)

```
1. User sends message: "I'm planning a trip to Stockholm next month"

2. Memory Application receives message
   - Retrieves relevant context from Neo4j (past Stockholm mentions, travel preferences)
   - Constructs prompt with context and conversation history
   
3. Calls LLM (reasoning: low, fast response)
   - System: "You are a helpful assistant with access to conversation memories."
   - Context: [Retrieved Neo4j memories about user's travel preferences]
   - User: "I'm planning a trip to Stockholm next month"
   
4. LLM generates response: "That's exciting! When we talked about Scandinavia before..."

5. Response sent to user immediately (end of user-visible latency)

6. LIVE PASS - Fast, lenient extraction (async, non-blocking)
   - Calls LLM again (reasoning: low, fast ~2-3 seconds)
   - System: "Identify entities and relationships. Propose entity types dynamically."
   - LLM proposes: 
     * Entity: "Stockholm Trip" (type: TravelEvent, confidence: 0.75)
     * Relationship: User PLANNING Stockholm Trip (confidence: 0.85)
   
7. Minimal validation (syntax only)
   - Is JSON valid? ✓
   - Can we parse entity structure? ✓
   - Accept both proposals
   
8. Store in staging area (unverified: true)
   - CREATE (e:Entity:TravelEvent {id: "...", name: "Stockholm Trip", unverified: true})
   - CREATE (u:Entity:User)-[:PLANNING {unverified: true}]->(e)
   - Mark with staging_timestamp for maintenance processing

9. Continue conversation - user can send next message immediately

--- 30 minutes later, user stops responding ---

10. MAINTENANCE PASS - Thorough, rigorous validation
   - System detects idle period (last_message > 30 min ago)
   - Query staging area: MATCH (e:Entity {unverified: true})
   - For each staged memory:
     a. Re-extract with high reasoning (get better entity types, properties)
     b. Run full validation pipeline:
        - Fact verification (query existing graph for conflicts)
        - Ensemble voting (extract 3x, require 60% agreement)
        - Confidence threshold (reject if < 0.6)
     c. If passes validation:
        - Remove 'unverified' flag
        - Enhance properties (add inferred data)
        - Update confidence score
     d. If fails validation:
        - Move to rejected_memories archive
        - Log reason for analysis
   
11. Schema consolidation
   - Identify similar entity types: TravelEvent vs TripEvent
   - Propose merges (LLM confirms they're synonyms)
   - Update labels and relationships
   
12. System returns to idle state, ready for next conversation
```

**Key Benefits of Two-Pass Approach:**
- User never waits for validation (< 1 second to see response)
- Low-quality live extraction is "good enough" for immediate context
- Maintenance pass can take minutes without user impact
- Failed validations don't create confusing errors mid-conversation
- Graph quality improves over time without slowing down interaction

### 2. Memory Management Components

#### 2.1 Memory Extraction

**Purpose:** Identify what information from a conversation should be stored.

**Two Modes:**

1. **Live Extraction** (fast, lenient, post-response)
2. **Maintenance Extraction** (thorough, accurate, during idle periods)

**Implementation:**
```python
class MemoryExtractor:
    """Analyzes conversation turns and proposes memory operations."""
    
    async def extract_memories_live(
        self, 
        conversation_turn: ConversationTurn
    ) -> List[MemoryProposal]:
        """
        Fast, lenient extraction for immediate storage in staging area.
        Uses reasoning: low, accepts confidence > 0.3
        
        Args:
            conversation_turn: User message + assistant response
            
        Returns:
            List of proposed memory operations (entities, relationships, facts)
        """
        
        system_prompt = """You are a memory extraction specialist. Analyze the conversation 
        and identify important information to store in a graph database.
        
        CRITICAL: You propose BOTH the entity content AND its type dynamically.
        Do not use predefined types - infer the most appropriate type from context.
        
        Examples:
        - User discusses quantum mechanics → EntityType: "Concept", "Experiment", "Physicist"
        - User discusses cooking → EntityType: "Recipe", "Ingredient", "Technique"
        - User discusses travel → EntityType: "Destination", "Trip", "Activity"
        
        Extract:
        - Entities (with dynamically determined types based on domain)
        - Relationships between entities (dynamic relationship types)
        - Properties relevant to the entity type
        - Confidence level (0.0-1.0)
        
        Return structured JSON following this schema:
        {
          "extractions": [
            {
              "type": "entity|relationship",
              "entity_type": "DynamicallyDeterminedType",  // e.g., "Experiment", "Recipe"
              "properties": {"key": "value"},
              "confidence": 0.0-1.0,
              "reasoning": "brief explanation"
            }
          ]
        }
        
        SPEED: This is a fast pass. Provide immediate extractions without extensive reasoning.
        LENIENCY: Include information even if confidence is moderate (> 0.3).
        """
        
        messages = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": f"User: {conversation_turn.user_message}"},
            {"role": "user", "content": f"Assistant: {conversation_turn.assistant_response}"}
        ]
        
        response = await self.llm_client.chat_completion(
            messages=messages,
            reasoning_level="low",  # Fast extraction
            response_format={"type": "json_object"},
            temperature=0.4  # Slightly higher for creativity in type generation
        )
        
        extractions = json.loads(response.content)
        return [MemoryProposal.from_dict(e, verified=False) for e in extractions["extractions"]]
    
    async def extract_memories_maintenance(
        self, 
        conversation_turn: ConversationTurn,
        staging_proposal: MemoryProposal  # Optional: refine existing staging memory
    ) -> List[MemoryProposal]:
        """
        Thorough extraction during maintenance windows.
        Uses reasoning: high, requires confidence > 0.6
        
        This re-processes staging area memories with better reasoning and validation.
        """
        
        system_prompt = """You are a memory extraction specialist performing THOROUGH analysis 
        during a maintenance window. Re-examine this conversation and propose high-quality 
        memory operations.
        
        ENTITY TYPES: Dynamically determined from domain. Consider:
        - Is this type consistent with existing types in the graph?
        - Could this entity be a more specific or general type?
        - Are there better property names that match the domain?
        
        QUALITY STANDARDS:
        - Only extract information you're highly confident about (> 0.6)
        - Provide detailed reasoning for each extraction
        - Consider relationships to other likely entities
        - Think about how this fits into the broader knowledge graph
        
        If provided with a staging proposal, you may:
        - Confirm it with higher confidence
        - Refine the entity type or properties
        - Split it into multiple entities if it's actually several things
        - Reject it if it doesn't meet quality standards
        
        Return structured JSON with high-quality extractions.
        """
        
        messages = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": f"User: {conversation_turn.user_message}"},
            {"role": "user", "content": f"Assistant: {conversation_turn.assistant_response}"}
        ]
        
        if staging_proposal:
            messages.append({
                "role": "user", 
                "content": f"Staging proposal to refine: {staging_proposal.to_dict()}"
            })
        
        response = await self.llm_client.chat_completion(
            messages=messages,
            reasoning_level="high",  # Thorough reasoning
            response_format={"type": "json_object"},
            temperature=0.2  # Lower temperature for consistency
        )
        
        # Store full chain-of-thought for monitoring
        self.log_cot(response.reasoning_content)
        
        extractions = json.loads(response.content)
        return [MemoryProposal.from_dict(e, verified=True) for e in extractions["extractions"]]
```

**Key Design Decisions:**
- **Dynamic entity types:** LLM proposes type names based on domain context
- **Two-pass extraction:** Fast/lenient for live, slow/rigorous for maintenance  
- **Staging refinement:** Maintenance pass can improve live extractions
- **Confidence thresholds:** 0.3+ for live (lenient), 0.6+ for maintenance (strict)
- **Reasoning levels:** Low for speed, high for quality
- **CoT logging:** Only in maintenance pass (has time to analyze)

#### 2.2 Memory Validation

**Purpose:** Prevent hallucinated or incorrect information from entering the graph.

**Implementation:**
```python
class MemoryValidator:
    """Multi-stage validation pipeline for memory proposals."""
    
    async def validate(
        self, 
        proposals: List[MemoryProposal],
        conversation_context: ConversationTurn
    ) -> List[ValidatedMemoryProposal]:
        """
        Runs proposals through validation pipeline:
        1. Syntax validation (is Cypher valid?)
        2. Schema validation (does entity type exist?)
        3. Fact verification (check against existing graph + web)
        4. Ensemble voting (do multiple extraction passes agree?)
        5. Confidence thresholding (reject low-confidence proposals)
        
        Returns only proposals that pass all checks.
        """
        
        validated = []
        
        for proposal in proposals:
            # Stage 1: Syntax validation
            if not self.validate_syntax(proposal):
                self.log_rejection(proposal, "invalid_syntax")
                continue
            
            # Stage 2: Schema validation
            if not self.validate_schema(proposal):
                self.log_rejection(proposal, "invalid_schema")
                continue
            
            # Stage 3: Fact verification
            verification_result = await self.verify_facts(proposal)
            if not verification_result.is_valid:
                self.log_rejection(proposal, "fact_check_failed", verification_result)
                continue
            
            # Stage 4: Ensemble voting (for high-stakes information)
            if proposal.confidence < 0.8 or proposal.is_factual_claim():
                ensemble_result = await self.ensemble_vote(proposal, conversation_context)
                if ensemble_result.agreement_rate < 0.6:  # 60% of passes must agree
                    self.log_rejection(proposal, "ensemble_disagreement", ensemble_result)
                    continue
            
            # Stage 5: Confidence threshold
            if proposal.confidence < self.min_confidence_threshold:
                self.log_rejection(proposal, "low_confidence")
                continue
            
            validated.append(ValidatedMemoryProposal(
                proposal=proposal,
                verification=verification_result,
                ensemble_agreement=ensemble_result.agreement_rate if ensemble_result else None
            ))
        
        return validated
    
    async def verify_facts(self, proposal: MemoryProposal) -> VerificationResult:
        """
        Verifies factual claims through multiple sources:
        1. Check if entity already exists in Neo4j (consistency check)
        2. Query web search for external verification (optional but recommended)
        3. Check for contradictions with existing graph data
        """
        
        # Check existing graph
        existing = await self.neo4j_client.find_similar_entities(proposal)
        if existing:
            # Entity exists - verify properties match
            conflicts = self.check_property_conflicts(proposal, existing)
            if conflicts:
                return VerificationResult(is_valid=False, reason="property_conflict", conflicts=conflicts)
        
        # For factual claims, query web search
        if proposal.is_factual_claim() and self.web_search_enabled:
            search_results = await self.web_search(proposal.to_search_query())
            verification = self.analyze_search_results(proposal, search_results)
            if not verification.supports_claim:
                return VerificationResult(is_valid=False, reason="web_verification_failed", details=verification)
        
        return VerificationResult(is_valid=True)
    
    async def ensemble_vote(
        self, 
        proposal: MemoryProposal, 
        context: ConversationTurn,
        num_passes: int = 3
    ) -> EnsembleResult:
        """
        Runs extraction multiple times with different prompts/temperatures.
        Proposals are only accepted if multiple passes agree.
        """
        
        extraction_passes = []
        
        for i in range(num_passes):
            # Vary prompt slightly or use different temperature
            variant_proposals = await self.extractor.extract_memories(
                context,
                reasoning_level="high",
                temperature=0.3 + (i * 0.1),  # 0.3, 0.4, 0.5
                prompt_variant=i
            )
            extraction_passes.append(variant_proposals)
        
        # Count how many passes extracted this same information
        agreement_count = self.count_agreements(proposal, extraction_passes)
        agreement_rate = agreement_count / num_passes
        
        return EnsembleResult(
            agreement_rate=agreement_rate,
            agreement_count=agreement_count,
            total_passes=num_passes,
            variant_proposals=extraction_passes
        )
```

**Validation Pipeline Stages:**

1. **Syntax Validation** (fast, deterministic)
   - Parse proposed Cypher queries
   - Verify entity types match schema
   - Check property keys are valid
   - Ensure relationship types exist

2. **Schema Validation** (fast, deterministic)
   - Verify entity types are in allowed set
   - Check relationship types are defined
   - Validate property value types (string, int, date, etc.)

3. **Fact Verification** (slow, heuristic)
   - Query Neo4j for existing similar entities
   - Check for property conflicts with existing data
   - Optional: Query web search for external confirmation
   - Flag contradictions with stored memories

4. **Ensemble Voting** (slow, computational)
   - Re-run extraction 2-3 times with variations
   - Count how many passes propose the same information
   - Require 60%+ agreement for acceptance
   - Used for high-stakes factual claims and low-confidence proposals

5. **Confidence Thresholding** (fast, heuristic)
   - Reject proposals below minimum confidence (e.g., 0.5)
   - Configurable per entity type (Person: 0.7, Preference: 0.5)

**Rejection Logging:**
Every rejected proposal is logged with:
- Original proposal
- Rejection reason and stage
- Chain-of-thought from extraction
- Conversation context
- Timestamp

This creates a dataset for improving prompts and tuning thresholds.

#### 2.3 Memory Storage

**Purpose:** Execute validated memory operations in Neo4j.

**Implementation:**
```python
class MemoryStorage:
    """Executes validated memory operations in Neo4j."""
    
    async def store(self, validated_proposals: List[ValidatedMemoryProposal]):
        """
        Converts validated proposals to Cypher queries and executes them.
        Uses MERGE for entities (idempotent) and CREATE for relationships.
        """
        
        async with self.neo4j_client.session() as session:
            async with session.begin_transaction() as tx:
                try:
                    for proposal in validated_proposals:
                        if proposal.type == "entity":
                            await self.create_entity(tx, proposal)
                        elif proposal.type == "relationship":
                            await self.create_relationship(tx, proposal)
                    
                    # Create conversation node linking all extractions
                    await self.create_conversation_node(tx, validated_proposals)
                    
                    await tx.commit()
                    
                except Exception as e:
                    await tx.rollback()
                    self.log_storage_error(validated_proposals, e)
                    raise
    
    async def create_entity(self, tx, proposal: ValidatedMemoryProposal):
        """
        Creates or merges an entity node.
        Uses MERGE for idempotency - won't create duplicates.
        """
        
        entity_type = proposal.entity_type
        properties = proposal.properties
        
        # Build property string for Cypher
        props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
        
        query = f"""
        MERGE (e:{entity_type} {{{props_str}}})
        ON CREATE SET e.created_at = datetime(), e.confidence = $confidence
        ON MATCH SET e.last_mentioned = datetime(), e.mention_count = coalesce(e.mention_count, 0) + 1
        RETURN e
        """
        
        params = {**properties, "confidence": proposal.confidence}
        
        result = await tx.run(query, params)
        return await result.single()
    
    async def create_relationship(self, tx, proposal: ValidatedMemoryProposal):
        """
        Creates a relationship between entities.
        Assumes entities already exist (created in prior step or earlier conversation).
        """
        
        query = f"""
        MATCH (source:{proposal.source_entity_type} {{id: $source_id}})
        MATCH (target:{proposal.target_entity_type} {{id: $target_id}})
        CREATE (source)-[r:{proposal.relationship_type} {{
            created_at: datetime(),
            confidence: $confidence,
            context: $context
        }}]->(target)
        RETURN r
        """
        
        params = {
            "source_id": proposal.source_id,
            "target_id": proposal.target_id,
            "confidence": proposal.confidence,
            "context": proposal.context_snippet
        }
        
        result = await tx.run(query, params)
        return await result.single()
```

**Neo4j Schema Design:**

```cypher
// DYNAMIC SCHEMA APPROACH - No predefined entity types!
// All entities use a generic Entity label, with type as a property
// This allows the schema to evolve based on what the user discusses

// Base constraint: Every entity needs a unique ID
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;

// Conversation tracking
CREATE CONSTRAINT conversation_id IF NOT EXISTS FOR (c:Conversation) REQUIRE c.id IS UNIQUE;

// Full-text index for semantic search (searches across all entities regardless of type)
CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
FOR (n:Entity)
ON EACH [n.name, n.description, n.type, n.properties];

// Temporal indexes for time-based queries
CREATE INDEX entity_created IF NOT EXISTS FOR (e:Entity) ON (e.created_at);
CREATE INDEX entity_mentioned IF NOT EXISTS FOR (e:Entity) ON (e.last_mentioned);
CREATE INDEX conversation_timestamp IF NOT EXISTS FOR (c:Conversation) ON (c.timestamp);

// Staging area index for maintenance processing
CREATE INDEX entity_staging IF NOT EXISTS FOR (e:Entity) ON (e.unverified);

// Example entity nodes with DYNAMIC types
// The 'type' property determines what kind of entity this is
// Properties are flexible - different types have different properties

(:Entity {
  id: "annelie_001",
  type: "Person",  // Dynamic type, not a Neo4j label
  name: "Annelie",
  relationship_to_user: "fiancée",
  created_at: datetime(),
  last_mentioned: datetime(),
  mention_count: 15,
  confidence: 0.95,
  unverified: false  // Passed maintenance validation
})

(:Entity {
  id: "stockholm_002",
  type: "City",
  name: "Stockholm",
  country: "Sweden",
  created_at: datetime(),
  last_mentioned: datetime(),
  mention_count: 23,
  confidence: 1.0,
  unverified: false
})

// Example of a domain-specific entity type (physics research)
(:Entity {
  id: "double_slit_exp_003",
  type: "Experiment",  // Type emerged from user's discussions
  name: "Double-slit experiment",
  field: "quantum mechanics",
  apparatus: "electron gun, double slit, detector screen",
  key_observation: "interference pattern",
  created_at: datetime(),
  confidence: 0.88,
  unverified: false
})

// Example of staging area entity (unverified)
(:Entity {
  id: "new_concept_004",
  type: "Concept",
  name: "Wave-particle duality",
  related_to: "double_slit_exp_003",
  staging_timestamp: datetime(),
  confidence: 0.45,  // Low confidence from live pass
  unverified: true  // Needs maintenance validation
})

// DYNAMIC RELATIONSHIPS
// Relationship types are also determined dynamically by the LLM
// No predefined relationship type constraints

(:Entity {id: "user_onuse", type: "User"})-[:LIVES_IN]->(:Entity {id: "stockholm_002", type: "City"})
(:Entity {id: "user_onuse", type: "User"})-[:ENGAGED_TO]->(:Entity {id: "annelie_001", type: "Person"})
(:Entity {id: "user_onuse", type: "User"})-[:RESEARCHING]->(:Entity {id: "double_slit_exp_003", type: "Experiment"})

// Relationships can have properties too
(:Entity {id: "double_slit_exp_003", type: "Experiment"})-[
  :DEMONSTRATES {
    confidence: 0.92,
    created_at: datetime()
  }
]->(:Entity {id: "new_concept_004", type: "Concept"})

// Conversation nodes link to mentioned entities
(:Conversation {
  id: "conv_2025-10-30_001",
  timestamp: datetime("2025-10-30T14:23:15"),
  user_message: "Tell me about the double-slit experiment",
  assistant_response: "The double-slit experiment demonstrates...",
  extraction_count: 3
})-[:MENTIONED]->(:Entity {id: "double_slit_exp_003", type: "Experiment"})

// SCHEMA EVOLUTION TRACKING
// After conversations, query what types exist:
MATCH (e:Entity)
RETURN DISTINCT e.type, count(*) as count
ORDER BY count DESC

// Example results after 200 conversations about physics:
// type            | count
// "Experiment"    | 45
// "Concept"       | 38
// "Physicist"     | 22
// "Theory"        | 18
// "Measurement"   | 15
// ...

// Maintenance can consolidate similar types:
// "Experiment" + "PhysicsExperiment" → "Experiment"
// "Physicist" + "Scientist" → "Scientist" (with field: "physics")
```

**Key Schema Principles:**

1. **Generic Entity label** - All entities are `Entity` nodes, differentiated by `type` property
2. **No type constraints** - New entity types can be created freely by the LLM
3. **Flexible properties** - Different entity types have different properties (no rigid schema)
4. **Staging flag** - `unverified: true` marks memories awaiting maintenance validation
5. **Confidence tracking** - All entities and relationships track confidence scores
6. **Mention frequency** - Track how often entities are referenced (relevance signal)
7. **Schema introspection** - Query the graph to see what types actually exist

**Why This Works:**

- Physics researcher's graph fills with Experiments, Measurements, Theories
- Cook's graph fills with Recipes, Ingredients, Techniques
- Traveler's graph fills with Destinations, Activities, Accommodations
- Schema adapts to user's domain without manual configuration
- Maintenance consolidates redundant types over time (e.g., "Trip" + "Journey" → "Trip")

#### 2.4 Memory Retrieval

**Purpose:** Find relevant context from Neo4j to inject into LLM prompts.

**Implementation:**
```python
class MemoryRetrieval:
    """Retrieves relevant memories from Neo4j for conversation context."""
    
    async def retrieve_context(
        self, 
        user_message: str,
        max_memories: int = 10
    ) -> List[Memory]:
        """
        Multi-strategy retrieval:
        1. Entity extraction from user message (what are they talking about?)
        2. Semantic search (full-text index on Neo4j)
        3. Temporal relevance (recent mentions weighted higher)
        4. Graph traversal (entities + 1-hop neighbors for context)
        """
        
        # Extract entities mentioned in current message
        mentioned_entities = await self.extract_entities(user_message)
        
        # Retrieve those entities + their immediate neighborhood
        memories = []
        
        for entity in mentioned_entities:
            # Get the entity node and its 1-hop neighbors
            query = """
            MATCH (e {id: $entity_id})
            OPTIONAL MATCH (e)-[r]-(neighbor)
            RETURN e, collect({relationship: type(r), neighbor: neighbor}) as connections
            ORDER BY e.last_mentioned DESC
            LIMIT $max_per_entity
            """
            
            result = await self.neo4j_client.run(query, {
                "entity_id": entity.id,
                "max_per_entity": 3
            })
            
            memories.extend(self.format_memories(result))
        
        # If not enough context, do semantic search
        if len(memories) < max_memories:
            semantic_results = await self.semantic_search(user_message, max_memories - len(memories))
            memories.extend(semantic_results)
        
        # Sort by relevance score (recency + mention_count + confidence)
        memories.sort(key=lambda m: m.relevance_score, reverse=True)
        
        return memories[:max_memories]
    
    async def extract_entities(self, text: str) -> List[Entity]:
        """
        Quick entity extraction from user message.
        Uses simpler, faster extraction than full memory extraction.
        """
        
        # Could use NER library (spaCy) or simple LLM call
        # For consistency, using LLM with reasoning: low
        
        response = await self.llm_client.chat_completion(
            messages=[{
                "role": "developer",
                "content": "Extract entity names from this text. Return JSON array of strings."
            }, {
                "role": "user",
                "content": text
            }],
            reasoning_level="low",
            temperature=0.2,
            max_tokens=200
        )
        
        entity_names = json.loads(response.content)
        
        # Look up these entities in Neo4j
        return await self.neo4j_client.find_entities_by_names(entity_names)
    
    async def semantic_search(self, query: str, limit: int) -> List[Memory]:
        """
        Uses Neo4j full-text index for semantic search.
        """
        
        cypher = """
        CALL db.index.fulltext.queryNodes('entity_search', $query)
        YIELD node, score
        MATCH (node)-[r]-(neighbor)
        RETURN node, score, collect({relationship: type(r), neighbor: neighbor}) as connections
        ORDER BY score DESC
        LIMIT $limit
        """
        
        result = await self.neo4j_client.run(cypher, {"query": query, "limit": limit})
        return self.format_memories(result)
    
    def format_memories(self, neo4j_result) -> List[Memory]:
        """
        Converts Neo4j results to structured Memory objects for prompt injection.
        """
        
        memories = []
        
        for record in neo4j_result:
            node = record["node"]
            connections = record.get("connections", [])
            
            # Format as natural language for prompt
            memory_text = self.node_to_text(node)
            
            # Add relationship context
            if connections:
                relationship_texts = [
                    f"{node['name']} {conn['relationship'].lower().replace('_', ' ')} {conn['neighbor']['name']}"
                    for conn in connections
                ]
                memory_text += "\n  Related: " + ", ".join(relationship_texts)
            
            memories.append(Memory(
                text=memory_text,
                entity_id=node["id"],
                confidence=node.get("confidence", 0.5),
                last_mentioned=node.get("last_mentioned"),
                relevance_score=self.calculate_relevance(node, record.get("score", 0))
            ))
        
        return memories
    
    def calculate_relevance(self, node, search_score: float) -> float:
        """
        Combines multiple signals into relevance score:
        - Search score (semantic similarity)
        - Recency (when last mentioned)
        - Frequency (mention_count)
        - Confidence (how certain we are about this memory)
        """
        
        recency_score = self.calculate_recency_score(node.get("last_mentioned"))
        frequency_score = min(node.get("mention_count", 1) / 10, 1.0)  # Cap at 10 mentions
        confidence = node.get("confidence", 0.5)
        
        # Weighted combination
        relevance = (
            search_score * 0.4 +
            recency_score * 0.3 +
            frequency_score * 0.2 +
            confidence * 0.1
        )
        
        return relevance
```

**Context Injection Strategy:**

Retrieved memories are injected into the LLM prompt in a structured format:

```python
def construct_prompt_with_context(user_message: str, memories: List[Memory]) -> List[Dict]:
    """
    Builds prompt with conversation context from Neo4j.
    """
    
    if not memories:
        # No relevant memories found
        context_text = "No prior conversation context available."
    else:
        # Format memories as structured context
        context_text = "# Relevant Information from Past Conversations\n\n"
        
        for memory in memories:
            context_text += f"- {memory.text}\n"
            if memory.last_mentioned:
                days_ago = (datetime.now() - memory.last_mentioned).days
                context_text += f"  (mentioned {days_ago} days ago)\n"
        
        context_text += "\n# Current Conversation\n"
    
    messages = [
        {
            "role": "developer",
            "content": """You are a helpful assistant with access to conversation memories.
            Use the provided context from past conversations when relevant to the user's current question.
            If you reference past information, mention when it was discussed to show you remember.
            Be natural - don't robotically list facts, weave memories into conversation organically."""
        },
        {
            "role": "user",
            "content": context_text
        },
        {
            "role": "user",
            "content": user_message
        }
    ]
    
    return messages
```

### 3. Scheduled Maintenance Jobs

**Purpose:** Refine the graph over time to improve quality and performance.

**Maintenance Operations:**

1. **Deduplication** (weekly)
   - Find similar entities with different IDs
   - Merge duplicate nodes, combining properties
   - Update relationships to point to merged node

2. **Consolidation** (weekly)
   - Identify entity clusters (e.g., multiple "Stockholm" mentions)
   - Create canonical entities with aliases
   - Link related entities that were mentioned separately

3. **Pruning** (monthly)
   - Remove low-confidence entities not mentioned in 90+ days
   - Archive old conversation nodes (keep entities, remove conversation detail)
   - Delete orphaned nodes with no relationships

4. **Relationship Inference** (monthly)
   - Use graph algorithms to infer implicit relationships
   - Example: If A knows B, and B knows C, maybe A knows C?
   - Validate inferences before creating relationships

5. **Confidence Adjustment** (continuous)
   - Increase confidence for entities mentioned multiple times
   - Decrease confidence for entities never referenced again
   - Flag entities with conflicting information for review

**Implementation:**
```python
class MaintenanceJobs:
    """Scheduled jobs for graph refinement."""
    
    async def run_deduplication(self):
        """
        Finds and merges duplicate entities using fuzzy matching.
        """
        
        query = """
        // Find potential duplicates (same type, similar names)
        MATCH (e1:Entity)
        MATCH (e2:Entity)
        WHERE e1.type = e2.type
          AND e1.id < e2.id  // Avoid comparing same pair twice
          AND apoc.text.jaroWinklerDistance(e1.name, e2.name) > 0.9
        RETURN e1, e2, apoc.text.jaroWinklerDistance(e1.name, e2.name) as similarity
        ORDER BY similarity DESC
        """
        
        duplicates = await self.neo4j_client.run(query)
        
        for dup in duplicates:
            # Ask LLM to confirm these are actually duplicates
            confirmation = await self.llm_client.confirm_duplicate(dup["e1"], dup["e2"])
            
            if confirmation.is_duplicate:
                await self.merge_entities(dup["e1"], dup["e2"])
    
    async def merge_entities(self, entity1, entity2):
        """
        Merges two duplicate entities:
        1. Combine properties (prefer higher confidence)
        2. Move all relationships to surviving entity
        3. Delete duplicate entity
        """
        
        # Determine which entity survives (higher confidence, more mentions)
        survivor = entity1 if entity1.confidence >= entity2.confidence else entity2
        duplicate = entity2 if survivor == entity1 else entity1
        
        query = """
        MATCH (duplicate {id: $duplicate_id})
        MATCH (survivor {id: $survivor_id})
        
        // Move relationships
        OPTIONAL MATCH (duplicate)-[r]-(other)
        CREATE (survivor)-[r2:${type(r)}]->(other)
        SET r2 = properties(r)
        DELETE r
        
        // Merge properties (keep survivor's, add duplicate's if not present)
        SET survivor += duplicate
        
        // Update metadata
        SET survivor.mention_count = survivor.mention_count + duplicate.mention_count
        SET survivor.merged_from = coalesce(survivor.merged_from, []) + [duplicate.id]
        
        // Delete duplicate
        DETACH DELETE duplicate
        
        RETURN survivor
        """
        
        await self.neo4j_client.run(query, {
            "survivor_id": survivor.id,
            "duplicate_id": duplicate.id
        })
        
        self.log_merge(survivor, duplicate)
```

### 4. Monitoring & Observability

**Purpose:** Track system health and LLM reliability over time.

**Key Metrics:**

1. **Hallucination Rate** (% of rejections due to fact-check failures)
2. **Extraction Agreement** (% of ensemble votes with >60% agreement)
3. **Storage Success Rate** (% of proposals that pass validation)
4. **Retrieval Relevance** (user feedback on context quality)
5. **CoT Quality Scores** (analyze reasoning traces for red flags)
6. **Graph Growth Rate** (entities/relationships added per day)
7. **Duplicate Detection Rate** (how often maintenance finds duplicates)

**Implementation:**
```python
class MonitoringService:
    """Tracks metrics and generates alerts."""
    
    def log_extraction(self, proposals: List[MemoryProposal], cot: str):
        """Logs extraction attempt with full chain-of-thought."""
        
        self.metrics.increment("extractions.total", len(proposals))
        
        # Analyze CoT for hallucination indicators
        hallucination_signals = self.analyze_cot(cot)
        if hallucination_signals.score > 0.5:
            self.metrics.increment("extractions.suspicious_cot")
            self.alert_high_hallucination_risk(proposals, hallucination_signals)
    
    def log_validation_result(self, proposal: MemoryProposal, result: ValidationResult):
        """Logs validation outcomes for analysis."""
        
        self.metrics.increment(f"validation.{result.stage}.{result.outcome}")
        
        if result.outcome == "rejected":
            # Store rejection for prompt improvement
            self.rejection_db.insert({
                "proposal": proposal.to_dict(),
                "reason": result.reason,
                "stage": result.stage,
                "timestamp": datetime.now(),
                "cot": result.chain_of_thought
            })
    
    def analyze_cot(self, cot: str) -> HallucinationSignals:
        """
        Analyzes chain-of-thought for hallucination indicators:
        - Hedging language ("I think", "maybe", "possibly")
        - Lack of justification (assertions without reasoning)
        - Contradictions within the reasoning
        - References to information not in the prompt
        """
        
        signals = HallucinationSignals()
        
        # Count hedge words
        hedge_words = ["i think", "maybe", "possibly", "probably", "seems like"]
        signals.hedge_count = sum(cot.lower().count(word) for word in hedge_words)
        
        # Check for unsupported assertions
        if "based on" not in cot and "because" not in cot:
            signals.lacks_justification = True
        
        # Check for contradictions (simple keyword approach)
        if "however" in cot and "but actually" in cot:
            signals.has_contradictions = True
        
        # Calculate overall score
        signals.score = (
            min(signals.hedge_count * 0.2, 0.5) +
            (0.3 if signals.lacks_justification else 0) +
            (0.5 if signals.has_contradictions else 0)
        )
        
        return signals
    
    def generate_daily_report(self):
        """
        Generates daily summary of system performance:
        - Extraction counts and success rates
        - Validation rejection breakdown
        - Hallucination indicators
        - Graph growth statistics
        - Top entities by mention count
        """
        
        report = {
            "date": datetime.now().date(),
            "extractions": {
                "total": self.metrics.get("extractions.total"),
                "validated": self.metrics.get("validation.final.accepted"),
                "success_rate": self.metrics.get("validation.final.accepted") / self.metrics.get("extractions.total")
            },
            "rejections": {
                "syntax": self.metrics.get("validation.syntax.rejected"),
                "fact_check": self.metrics.get("validation.fact_check.rejected"),
                "ensemble": self.metrics.get("validation.ensemble.rejected"),
                "low_confidence": self.metrics.get("validation.confidence.rejected")
            },
            "hallucination_signals": self.metrics.get("extractions.suspicious_cot"),
            "graph_stats": await self.get_graph_stats()
        }
        
        # Send to monitoring dashboard or email
        await self.send_report(report)
```

## Implementation Plan

### Phase 1: Foundation (Week 1-2)

**Goal:** Get basic infrastructure running - LLM server, Neo4j, simple conversation flow.

**Tasks:**

1. **Set up llama.cpp server**
   - Download GPT-OSS:120b GGUF Q4_K_M from Hugging Face
   - Configure llama.cpp with OpenAI-compatible API
   - Test basic chat completions with Harmony format
   - Verify reasoning levels (low/medium/high) work correctly
   - Document: connection details, model path, performance benchmarks

2. **Set up Neo4j database**
   - Install Neo4j 5.x (Community Edition for local dev)
   - Install APOC plugin for advanced graph operations
   - Create initial schema (entity types, relationship types, indexes)
   - Write test queries to verify schema works
   - Document: connection string, credentials, schema reference

3. **Create basic Python application**
   - Set up FastAPI server with `/chat` endpoint
   - Implement conversation history management (in-memory for now)
   - Create LLM client wrapper for llama.cpp API
   - Create Neo4j client wrapper with connection pooling
   - Test end-to-end: user message → LLM response → store in memory

4. **Implement simple memory storage**
   - After each conversation turn, create conversation node in Neo4j
   - Store: user message, assistant response, timestamp
   - Don't do extraction yet - just log conversations
   - Verify: Can query Neo4j and see conversation history

**Deliverables:**
- Working llama.cpp server responding to API calls
- Neo4j with schema populated
- Python application that can chat and log to database
- README with setup instructions

**Success Criteria:**
- User can send message, get response, see it logged in Neo4j
- Average response time < 5 seconds for short messages
- No crashes or connection errors during 10-message conversation

### Phase 2: Memory Extraction (Week 3-4)

**Goal:** Implement LLM-driven memory extraction with structured output.

**Tasks:**

1. **Build MemoryExtractor class**
   - Implement `extract_memories()` with system prompt
   - Use JSON schema for structured extraction
   - Support reasoning levels (start with medium)
   - Store full chain-of-thought for monitoring
   - Unit tests: verify JSON parsing, handle empty extractions

2. **Define memory proposal data models**
   - `MemoryProposal` class (entity/relationship, properties, confidence)
   - `ConversationTurn` class (user msg, assistant msg, timestamp)
   - Pydantic models for validation
   - Serialization/deserialization for logging

3. **Implement async extraction pipeline**
   - After sending response to user, trigger extraction in background
   - Don't block user on memory operations
   - Use asyncio for concurrent operations
   - Handle errors gracefully (log and continue)

4. **Test extraction quality**
   - Create test conversations with known entities/relationships
   - Verify LLM extracts expected information
   - Measure extraction consistency (run same conversation 3x, check agreement)
   - Document: prompt engineering notes, example extractions

**Deliverables:**
- `MemoryExtractor` class with comprehensive tests
- Example extractions from real conversations
- Performance benchmarks (latency per extraction)

**Success Criteria:**
- LLM successfully extracts entities from 80%+ of test conversations
- JSON parsing succeeds 95%+ of the time
- Extraction doesn't block user responses (runs async)
- Chain-of-thought is logged for all extractions

### Phase 3: Validation Infrastructure (Week 5-6)

**Goal:** Implement multi-stage validation to prevent hallucinations.

**Tasks:**

1. **Build MemoryValidator class**
   - Stage 1: Syntax validation (Cypher parsing, schema checks)
   - Stage 2: Fact verification (Neo4j consistency checks)
   - Stage 3: Ensemble voting (re-extract with variations)
   - Stage 4: Confidence thresholding
   - Log all rejections with reasons

2. **Implement syntax validation**
   - Parse proposed Cypher queries
   - Check entity types against schema
   - Verify relationship types exist
   - Validate property value types
   - Return detailed error messages for debugging

3. **Implement fact verification**
   - Query Neo4j for existing similar entities
   - Check for property conflicts
   - Flag contradictions with stored memories
   - (Optional) Integrate web search for external verification
   - Return verification results with evidence

4. **Implement ensemble voting**
   - Re-run extraction 2-3 times with prompt variations
   - Count agreements (same entity/relationship extracted)
   - Calculate agreement rate
   - Require 60%+ agreement for low-confidence proposals
   - Log all variant extractions for analysis

5. **Create rejection logging database**
   - SQLite table for rejected proposals
   - Store: proposal, reason, stage, CoT, timestamp
   - Build queries to analyze rejection patterns
   - Dashboard to track rejection rates over time

**Deliverables:**
- `MemoryValidator` class with all stages implemented
- Rejection logging database with sample data
- Validation metrics dashboard (simple HTML page)
- Documentation: validation pipeline, tuning thresholds

**Success Criteria:**
- Validation catches 90%+ of intentionally bad proposals (hallucinations)
- False negative rate < 10% (good proposals accepted)
- Validation latency < 10 seconds per proposal (medium reasoning)
- Rejection reasons are clear and actionable

### Phase 4: Memory Storage & Retrieval (Week 7-8)

**Goal:** Store validated memories in Neo4j and retrieve relevant context.

**Tasks:**

1. **Build MemoryStorage class**
   - Convert validated proposals to Cypher queries
   - Use MERGE for entities (idempotency)
   - CREATE relationships with context
   - Handle transactions (all-or-nothing storage)
   - Create conversation nodes linking extractions
   - Error handling and rollback

2. **Implement entity creation**
   - MERGE entities with unique IDs
   - Set created_at, confidence, mention_count
   - Update last_mentioned on subsequent mentions
   - Increment mention_count
   - Handle property conflicts gracefully

3. **Implement relationship creation**
   - Match source and target entities
   - CREATE relationship with properties
   - Store context snippet
   - Set confidence and timestamp
   - Handle missing entities (log warning)

4. **Build MemoryRetrieval class**
   - Extract entities from user message (NER)
   - Query Neo4j for entities + 1-hop neighbors
   - Implement semantic search (full-text index)
   - Calculate relevance scores (recency + frequency + confidence)
   - Format memories for prompt injection

5. **Implement context injection**
   - Retrieve relevant memories before LLM call
   - Format as structured context in system prompt
   - Include temporal information (when mentioned)
   - Limit context size (max tokens)
   - Test: verify LLM uses context appropriately

**Deliverables:**
- `MemoryStorage` and `MemoryRetrieval` classes with tests
- Neo4j populated with test conversations
- Example retrievals with relevance scores
- Before/after comparison (responses with vs without context)

**Success Criteria:**
- Validated proposals are stored successfully 100% of the time
- Retrieval finds relevant context for 70%+ of queries
- LLM references stored context in responses when appropriate
- No duplicate entities created (MERGE works correctly)
- Graph remains queryable and performant (< 100ms for retrieval)

### Phase 5: Scheduled Maintenance (Week 9-10)

**Goal:** Implement jobs to refine graph quality over time.

**Tasks:**

1. **Build MaintenanceJobs class**
   - Deduplication (find and merge duplicates)
   - Consolidation (cluster related entities)
   - Pruning (remove old, low-confidence entities)
   - Confidence adjustment (boost frequently mentioned)
   - Relationship inference (graph algorithms)

2. **Implement deduplication**
   - Find potential duplicates (fuzzy name matching)
   - Use LLM to confirm duplicates (show both entities, ask if same)
   - Merge entities (combine properties, move relationships)
   - Log merges for review
   - Schedule weekly (cron job)

3. **Implement pruning**
   - Query for entities not mentioned in 90+ days
   - Filter by low confidence (< 0.3)
   - Archive before deletion (export to JSON)
   - Delete entities and orphaned nodes
   - Schedule monthly

4. **Implement confidence adjustment**
   - Increase confidence for entities mentioned 5+ times
   - Decrease confidence for entities not mentioned in 60+ days
   - Identify conflicting information (property mismatches)
   - Flag conflicts for manual review
   - Run daily as background task

5. **Create maintenance dashboard**
   - Show graph statistics (entity count, relationship count)
   - Display maintenance history (merges, deletions)
   - List entities flagged for review
   - Scheduled job status and logs
   - Simple web UI (Flask or FastAPI template)

**Deliverables:**
- `MaintenanceJobs` class with all operations
- Scheduled jobs configured (cron or APScheduler)
- Maintenance dashboard accessible via browser
- Documentation: maintenance strategy, tuning parameters

**Success Criteria:**
- Deduplication finds and merges 80%+ of duplicates
- Pruning removes low-value entities without breaking graph
- Confidence adjustment improves retrieval relevance
- Dashboard shows accurate graph statistics
- Maintenance jobs run reliably on schedule

### Phase 6: Monitoring & Optimization (Week 11-12)

**Goal:** Add observability and optimize for production readiness.

**Tasks:**

1. **Build MonitoringService class**
   - Log all extractions with CoT
   - Track validation results by stage
   - Calculate hallucination indicators from CoT
   - Generate daily performance reports
   - Alert on anomalies (sudden hallucination spike)

2. **Implement hallucination analysis**
   - Parse CoT for hedge words, contradictions
   - Calculate hallucination risk score
   - Store high-risk extractions for review
   - Track hallucination rate over time
   - Build training set for prompt improvement

3. **Create metrics dashboard**
   - Real-time extraction and validation metrics
   - Hallucination rate chart (daily/weekly/monthly)
   - Graph growth chart (entities, relationships over time)
   - Validation rejection breakdown (by stage and reason)
   - Top entities by mention count and confidence

4. **Optimize performance**
   - Profile bottlenecks (LLM calls, Neo4j queries, validation)
   - Cache frequent queries (Redis or in-memory)
   - Batch Neo4j operations where possible
   - Tune reasoning levels (low for conversation, high for memory)
   - Optimize Neo4j indexes for common queries

5. **Load testing**
   - Simulate 100+ concurrent users
   - Test system under heavy load
   - Measure: response time, throughput, error rate
   - Identify breaking points
   - Document: capacity limits, scaling recommendations

6. **Documentation**
   - Architecture diagrams (update this document with actuals)
   - API documentation (FastAPI auto-generates)
   - Deployment guide (Docker Compose, systemd services)
   - Troubleshooting guide (common errors, solutions)
   - Prompt engineering notes (how to improve extraction quality)

**Deliverables:**
- `MonitoringService` with full metrics tracking
- Metrics dashboard (Grafana or custom web UI)
- Performance optimization report
- Load testing results
- Complete documentation package

**Success Criteria:**
- Hallucination rate < 20% (down from GPT-OSS:120b baseline of 78%)
- Average response time < 2 seconds (with context retrieval)
- System handles 10+ concurrent users without degradation
- Dashboard shows accurate real-time metrics
- Documentation is clear enough for handoff to another developer

## Project Structure

```
neo4j-memory-chatbot/
├── README.md                          # Setup and usage instructions
├── ARCHITECTURE.md                    # This document
├── requirements.txt                   # Python dependencies
├── docker-compose.yml                 # Neo4j + Redis for local dev
├── .env.example                       # Environment variables template
│
├── src/
│   ├── __init__.py
│   ├── main.py                        # FastAPI application entry point
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── conversation.py            # ConversationTurn, Message models
│   │   ├── memory.py                  # MemoryProposal, Memory, Entity models
│   │   ├── validation.py              # ValidationResult, EnsembleResult models
│   │   └── monitoring.py              # Metrics, HallucinationSignals models
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_client.py              # Wrapper for llama.cpp API
│   │   ├── neo4j_client.py            # Neo4j driver wrapper
│   │   ├── memory_extractor.py        # MemoryExtractor class
│   │   ├── memory_validator.py        # MemoryValidator class
│   │   ├── memory_storage.py          # MemoryStorage class
│   │   ├── memory_retrieval.py        # MemoryRetrieval class
│   │   ├── maintenance_jobs.py        # MaintenanceJobs class
│   │   └── monitoring_service.py      # MonitoringService class
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat.py                    # /chat endpoint
│   │   ├── memory.py                  # /memory endpoints (query, stats)
│   │   └── monitoring.py              # /metrics, /health endpoints
│   │
│   ├── prompts/
│   │   ├── extraction.txt             # Memory extraction system prompt
│   │   ├── conversation.txt           # Conversation system prompt
│   │   ├── validation.txt             # Validation confirmation prompts
│   │   └── examples.json              # Few-shot examples for extraction
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py                # Configuration (env vars, constants)
│   │   └── neo4j_schema.cypher        # Neo4j schema definition
│   │
│   └── utils/
│       ├── __init__.py
│       ├── cypher_validator.py        # Cypher syntax parsing
│       ├── text_processing.py         # Entity extraction, text utilities
│       └── logging.py                 # Structured logging setup
│
├── tests/
│   ├── __init__.py
│   ├── test_extraction.py             # Memory extraction tests
│   ├── test_validation.py             # Validation pipeline tests
│   ├── test_storage.py                # Neo4j storage tests
│   ├── test_retrieval.py              # Context retrieval tests
│   ├── test_maintenance.py            # Maintenance job tests
│   ├── fixtures/
│   │   ├── conversations.json         # Test conversation examples
│   │   └── memories.json              # Expected extraction outputs
│   └── integration/
│       └── test_end_to_end.py         # Full flow integration tests
│
├── scripts/
│   ├── setup_neo4j.py                 # Initialize Neo4j schema
│   ├── run_maintenance.py             # Manual maintenance job trigger
│   ├── analyze_rejections.py          # Analyze validation rejections
│   └── export_graph.py                # Export Neo4j to JSON backup
│
├── monitoring/
│   ├── dashboard.html                 # Metrics dashboard (simple)
│   └── prometheus.yml                 # Prometheus config (if used)
│
└── docs/
    ├── setup.md                       # Detailed setup instructions
    ├── api.md                         # API documentation
    ├── prompts.md                     # Prompt engineering guide
    ├── neo4j_queries.md               # Useful Neo4j query examples
    └── troubleshooting.md             # Common issues and solutions
```

## Configuration

**Environment Variables (`.env`):**

```bash
# LLM Server
LLAMA_CPP_URL=http://localhost:8080
LLAMA_CPP_MODEL=gpt-oss-120b
LLAMA_CPP_TIMEOUT=120

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
NEO4J_DATABASE=neo4j

# Application
APP_HOST=0.0.0.0
APP_PORT=3000
LOG_LEVEL=INFO

# Memory Settings
EXTRACTION_REASONING_LEVEL=medium
CONVERSATION_REASONING_LEVEL=low
MAX_CONTEXT_MEMORIES=10
MIN_CONFIDENCE_THRESHOLD=0.5
ENSEMBLE_VOTING_ENABLED=true
ENSEMBLE_AGREEMENT_THRESHOLD=0.6

# Validation Settings
FACT_VERIFICATION_ENABLED=true
WEB_SEARCH_ENABLED=false  # Optional, requires API key
WEB_SEARCH_API_KEY=

# Maintenance
DEDUPLICATION_SCHEDULE="0 2 * * 0"  # Weekly Sunday 2 AM
PRUNING_SCHEDULE="0 3 1 * *"        # Monthly 1st day 3 AM
CONFIDENCE_ADJUSTMENT_SCHEDULE="0 4 * * *"  # Daily 4 AM

# Monitoring
METRICS_ENABLED=true
DAILY_REPORT_ENABLED=true
ALERT_EMAIL=your-email@example.com
```

## Key Design Decisions

1. **Separation of Concerns:** AI server (llama.cpp) is independent from application logic. Application orchestrates, validates, and stores - doesn't manage the model.

2. **Async-First:** Memory operations run asynchronously after responding to user. Conversational latency never blocked by memory work.

3. **Validation-Gated Autonomy:** LLM proposes memory operations autonomously, but validation layer prevents bad decisions from corrupting the graph.

4. **Structured Outputs:** Use JSON schema for extraction to ensure parseability. Harmony format provides reasoning visibility.

5. **Confidence-Weighted Storage:** Every entity and relationship has a confidence score. Low-confidence information is stored but flagged for verification.

6. **Multi-Stage Validation:** Cheap checks first (syntax), expensive checks last (ensemble voting). Fail fast to save compute.

7. **Ensemble Voting for Facts:** High-stakes information (factual claims, low-confidence proposals) requires agreement across multiple extraction passes.

8. **Chain-of-Thought Logging:** Full reasoning traces are stored for every extraction. Enables hallucination analysis and prompt improvement.

9. **Temporal Relevance:** Recent memories weighted higher in retrieval. Old memories decay in confidence unless re-mentioned.

10. **Scheduled Refinement:** Graph quality improves over time through maintenance. Duplicates merged, low-value entities pruned, confidence adjusted.

## Success Metrics

**System Performance:**
- Average response time: < 2 seconds
- Memory extraction latency: < 5 seconds (async, doesn't block user)
- Context retrieval latency: < 100ms
- Concurrent user capacity: 10+ without degradation

**Memory Quality:**
- Hallucination rate: < 20% (target, vs GPT-OSS:120b baseline 78%)
- Storage success rate: > 60% of proposals pass validation
- Ensemble agreement rate: > 70% for factual claims
- Duplicate entity rate: < 5% after deduplication runs

**User Experience:**
- Context relevance: User confirms retrieved memories are helpful 70%+ of the time
- Conversation continuity: Bot references past information appropriately
- No hallucinated entities: Zero instances of bot inventing non-existent people/places

**Graph Health:**
- Entity count grows linearly with conversations (not exponentially from duplicates)
- Relationship density: 2-5 relationships per entity on average
- Orphaned nodes: < 1% of total nodes
- Query performance: < 100ms for typical retrievals

## Risks & Mitigations

**Risk 1: High Hallucination Rate**
- *Mitigation:* Multi-stage validation, ensemble voting, fact verification, CoT monitoring

**Risk 2: Poor Cypher Generation**
- *Mitigation:* Syntax validation, few-shot examples, schema documentation in prompts

**Risk 3: Graph Quality Degrades Over Time**
- *Mitigation:* Scheduled maintenance jobs, confidence decay, deduplication, pruning

**Risk 4: Slow Performance (LLM Latency)**
- *Mitigation:* Async extraction, low reasoning for conversation, caching, GPU optimization

**Risk 5: Complex Integration Between Components**
- *Mitigation:* Clean API boundaries, comprehensive tests, structured logging, error handling

**Risk 6: Neo4j Query Performance Issues**
- *Mitigation:* Proper indexing, query optimization, limit context size, connection pooling

## Future Enhancements

**Phase 7+ (Beyond Initial Implementation):**

1. **Multi-User Support:**
   - User authentication and isolation
   - Separate graphs per user or shared graph with user ownership
   - Privacy controls (what memories to share across users)

2. **Advanced Memory Types:**
   - Episodic memory (full conversation transcripts)
   - Semantic memory (general knowledge facts)
   - Procedural memory (how to do tasks, recipes, instructions)
   - Emotional memory (sentiment, reactions, preferences intensity)

3. **Memory Visualization:**
   - Interactive graph visualization (D3.js or Neo4j Bloom)
   - Timeline view of conversations
   - Entity detail pages with full history
   - Relationship exploration

4. **Proactive Memory:**
   - Bot suggests related memories during conversation
   - "By the way, we talked about X before..."
   - Reminder system based on temporal patterns

5. **Memory Import/Export:**
   - Export memories to JSON for backup
   - Import memories from other systems
   - Sync across multiple instances

6. **Fine-Tuning:**
   - Collect high-quality extraction examples
   - Fine-tune GPT-OSS:120b on memory extraction task
   - Improve Cypher generation through LoRA

7. **Advanced Validation:**
   - External knowledge base integration (Wikipedia, Wikidata)
   - Fact-checking API integration (Google Fact Check API)
   - Contradiction detection using NLI models

8. **Voice Interface:**
   - Speech-to-text for voice conversations
   - Text-to-speech for responses
   - Store conversation transcripts with audio metadata

## Conclusion

This architecture balances the powerful capabilities of GPT-OSS:120b (tool use, reasoning, local deployment) with the critical weaknesses (hallucination risk, reliability issues) through comprehensive validation infrastructure. The key insight is **validation-gated autonomy**: let the LLM propose memory operations freely, but gate all storage behind multi-stage verification.

The clean separation between AI server and application logic provides flexibility, maintainability, and clear responsibilities. The LLM focuses on conversation and extraction; the application focuses on validation, storage, and orchestration.

With proper implementation of the validation pipeline, this system can achieve:
- Conversational memory that feels infinite and natural
- Hallucination rates < 20% (vs 78% baseline)
- Local deployment with full privacy
- Graph quality that improves over time through refinement
- Clear observability into LLM decision-making

The implementation plan provides a structured path from basic infrastructure to production-ready system over 12 weeks, with clear deliverables and success criteria at each phase.

This document should provide sufficient detail for Claude Code to implement the system, with specific class interfaces, implementation strategies, and architectural decisions explained throughout.
