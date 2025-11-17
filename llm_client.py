import os
import json
from typing import List, Optional, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from schemas import LLMSchemas

load_dotenv()

class LLMClient:

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "google/gemini-2.5-flash-lite",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):

        self.api_key = api_key or os.getenv("OPENROUTER_SECRET_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_SECRET_KEY not found. ")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize OpenAI client pointed at OpenRouter
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        messages = []

        if system_message:
            messages.append({
                "role": "system",
                "content": system_message
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens
        )

        return response.choices[0].message.content

    def generate_structured(
        self,
        prompt: str,
        schema_name: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a structured JSON response following a predefined schema.

        Args:
            prompt: The user prompt
            schema_name: Name of the schema to use (e.g., 'intervention', 'repair', 'critique')
            system_message: Optional system message
            temperature: Optional temperature override
            max_tokens: Optional max tokens override

        Returns:
            Parsed JSON object matching the schema

        Raises:
            ValueError: If schema_name is invalid or response parsing fails
        """
        messages = []

        if system_message:
            messages.append({
                "role": "system",
                "content": system_message
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        # Get the response format for structured outputs
        response_format = LLMSchemas.get_response_format(schema_name)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            response_format=response_format
        )

        content = response.choices[0].message.content

        # Parse the JSON response
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse structured response: {e}\nContent: {content}")

class MultiAgentLLM:
    """
    Manages multiple LLM instances for multi-agent critique.

    This allows using different models or configurations for different agents
    in the critique process.
    """

    def __init__(
        self,
        num_agents: int = 3,
        models: Optional[List[str]] = None,
        api_key: Optional[str] = None
    ):
        self.num_agents = num_agents

        # Default models if not specified
        if models is None:
            models = ["google/gemini-2.5-flash-lite"] * num_agents
        elif len(models) < num_agents:
            # Extend with the first model if not enough specified
            models = models + [models[0]] * (num_agents - len(models))

        self.agents = [
            LLMClient(api_key=api_key, model=model)
            for model in models[:num_agents]
        ]

    def get_agent(self, index: int) -> LLMClient:
        if 0 <= index < self.num_agents:
            return self.agents[index]
        raise IndexError(f"Agent index {index} out of range (0-{self.num_agents-1})")

    def generate_all(
        self,
        prompt: str,
        system_message: Optional[str] = None
    ) -> List[str]:
        return [
            agent.generate(prompt, system_message)
            for agent in self.agents
        ]
