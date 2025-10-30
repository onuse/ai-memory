"""LLM client for communicating with llama.cpp server."""
import httpx
import json
import logging
from typing import List, Dict, Any, Optional, Literal
from src.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for llama.cpp server with OpenAI-compatible API."""

    def __init__(self):
        self.base_url = settings.llama_cpp_url
        self.model = settings.llama_cpp_model
        self.timeout = settings.llama_cpp_timeout
        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        reasoning_level: Literal["low", "medium", "high"] = "low",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to llama.cpp server.

        Args:
            messages: List of message dicts with 'role' and 'content'
            reasoning_level: Reasoning level for the model (low/medium/high)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            response_format: Optional format specification (e.g., {"type": "json_object"})

        Returns:
            Dict with 'content' (response text) and optionally 'reasoning_content'
        """
        # Build request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Add reasoning level if supported by model
        # Note: This may need adjustment based on actual llama.cpp API
        if reasoning_level != "low":
            payload["reasoning_effort"] = reasoning_level

        # Add JSON mode if requested
        if response_format:
            payload["response_format"] = response_format

        try:
            logger.debug(f"Sending chat completion request: {len(messages)} messages, reasoning={reasoning_level}")

            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()

            result = response.json()

            # Extract response content
            choice = result["choices"][0]
            content = choice["message"]["content"]

            # Extract reasoning if available (Harmony format)
            reasoning_content = choice["message"].get("reasoning_content")

            logger.debug(f"Received response: {len(content)} chars")

            return {
                "content": content,
                "reasoning_content": reasoning_content,
                "usage": result.get("usage", {}),
            }

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from LLM server: {e.response.status_code} - {e.response.text}")
            raise Exception(f"LLM server error: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Request error to LLM server: {str(e)}")
            raise Exception(f"Failed to connect to LLM server: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in chat completion: {str(e)}")
            raise

    async def extract_json(
        self,
        messages: List[Dict[str, str]],
        reasoning_level: Literal["low", "medium", "high"] = "medium",
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Request structured JSON output from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            reasoning_level: Reasoning level for the model
            temperature: Sampling temperature (lower for more deterministic)

        Returns:
            Parsed JSON object

        Raises:
            Exception: If JSON parsing fails
        """
        response = await self.chat_completion(
            messages=messages,
            reasoning_level=reasoning_level,
            temperature=temperature,
            response_format={"type": "json_object"},
        )

        try:
            # Parse JSON response
            content = response["content"]
            parsed = json.loads(content)
            return parsed
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {content}")
            # Try to extract JSON from markdown code blocks if present
            if "```json" in content:
                try:
                    start = content.index("```json") + 7
                    end = content.rindex("```")
                    json_str = content[start:end].strip()
                    parsed = json.loads(json_str)
                    logger.info("Successfully extracted JSON from markdown block")
                    return parsed
                except Exception:
                    pass

            raise Exception(f"Invalid JSON response from LLM: {str(e)}")

    async def simple_chat(
        self,
        system_prompt: str,
        user_message: str,
        reasoning_level: Literal["low", "medium", "high"] = "low",
        temperature: float = 0.7,
    ) -> str:
        """
        Simple chat interface with system prompt and user message.

        Args:
            system_prompt: System prompt setting behavior
            user_message: User's message
            reasoning_level: Reasoning level for the model
            temperature: Sampling temperature

        Returns:
            String response from LLM
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        response = await self.chat_completion(
            messages=messages,
            reasoning_level=reasoning_level,
            temperature=temperature,
        )

        return response["content"]


# Global LLM client instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create global LLM client instance."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


async def close_llm_client():
    """Close global LLM client."""
    global _llm_client
    if _llm_client is not None:
        await _llm_client.close()
        _llm_client = None
