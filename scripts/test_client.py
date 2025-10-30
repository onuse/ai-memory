"""Simple test client for AI Memory system."""
import asyncio
import httpx
import json
from typing import Optional


class MemoryClient:
    """Client for testing the AI Memory API."""

    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url
        self.conversation_id: Optional[str] = None

    async def chat(self, message: str) -> dict:
        """Send a chat message and get response."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            payload = {"message": message}

            if self.conversation_id:
                payload["conversation_id"] = self.conversation_id

            response = await client.post(
                f"{self.base_url}/chat",
                json=payload,
            )

            response.raise_for_status()
            result = response.json()

            # Store conversation ID for continuity
            self.conversation_id = result.get("conversation_id")

            return result

    async def get_stats(self) -> dict:
        """Get database statistics."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/stats")
            response.raise_for_status()
            return response.json()

    async def health_check(self) -> dict:
        """Check system health."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()


async def run_example_conversation():
    """Run an example conversation to test memory."""
    print("=" * 60)
    print("AI Memory - Test Conversation")
    print("=" * 60)

    client = MemoryClient()

    # Check health
    print("\n[1] Checking system health...")
    try:
        health = await client.health_check()
        print(f"✓ System healthy: {json.dumps(health, indent=2)}")
    except Exception as e:
        print(f"✗ Health check failed: {str(e)}")
        print("Make sure the server is running: uvicorn src.main:app --port 3000")
        return

    # Test conversation 1: Introduce yourself
    print("\n[2] First message: Introducing Jonas...")
    message1 = "Hi! My name is Jonas. I live in Stockholm and I'm working on an AI project."

    try:
        response1 = await client.chat(message1)
        print(f"User: {message1}")
        print(f"Assistant: {response1['response']}")
        print(f"Conversation ID: {response1['conversation_id']}")
        print(f"Memories in context: {response1.get('memories_extracted', 0)}")
    except Exception as e:
        print(f"✗ Chat failed: {str(e)}")
        return

    # Wait a bit for background extraction to complete
    print("\n[3] Waiting for memory extraction...")
    await asyncio.sleep(5)

    # Check stats
    stats = await client.get_stats()
    print(f"Database stats: {json.dumps(stats['database'], indent=2)}")

    # Test conversation 2: Ask about previous info (should recall Stockholm)
    print("\n[4] Second message: Asking about location...")
    message2 = "What city did I mention I live in?"

    try:
        response2 = await client.chat(message2)
        print(f"User: {message2}")
        print(f"Assistant: {response2['response']}")
        print(f"Memories in context: {response2.get('memories_extracted', 0)}")

        if "Stockholm" in response2['response'] or "stockholm" in response2['response']:
            print("✓ Memory recall successful! The system remembered Stockholm.")
        else:
            print("✗ Memory recall failed. Expected mention of Stockholm.")
    except Exception as e:
        print(f"✗ Chat failed: {str(e)}")
        return

    # Wait for extraction
    await asyncio.sleep(5)

    # Test conversation 3: Add more context
    print("\n[5] Third message: Adding more information...")
    message3 = "My fiancée Annelie is from Sweden too. We both love photography."

    try:
        response3 = await client.chat(message3)
        print(f"User: {message3}")
        print(f"Assistant: {response3['response']}")
        print(f"Memories in context: {response3.get('memories_extracted', 0)}")
    except Exception as e:
        print(f"✗ Chat failed: {str(e)}")
        return

    # Wait for extraction
    await asyncio.sleep(5)

    # Test conversation 4: Test recall of multiple facts
    print("\n[6] Fourth message: Testing multi-fact recall...")
    message4 = "Can you remind me about my personal details you know?"

    try:
        response4 = await client.chat(message4)
        print(f"User: {message4}")
        print(f"Assistant: {response4['response']}")
        print(f"Memories in context: {response4.get('memories_extracted', 0)}")

        # Check if key facts are mentioned
        facts_recalled = []
        response_lower = response4['response'].lower()

        if "jonas" in response_lower:
            facts_recalled.append("name")
        if "stockholm" in response_lower:
            facts_recalled.append("location")
        if "annelie" in response_lower:
            facts_recalled.append("fiancée")
        if "photography" in response_lower or "photo" in response_lower:
            facts_recalled.append("interest")

        print(f"\n✓ Facts recalled: {', '.join(facts_recalled) if facts_recalled else 'none'}")

    except Exception as e:
        print(f"✗ Chat failed: {str(e)}")
        return

    # Final stats
    print("\n[7] Final database statistics:")
    final_stats = await client.get_stats()
    print(json.dumps(final_stats, indent=2))

    print("\n" + "=" * 60)
    print("Test conversation complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_example_conversation())
