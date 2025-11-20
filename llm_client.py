import os
import json
from typing import List, Optional, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from schemas import LLMSchemas

load_dotenv()

class LLMClient:

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "google/gemini-2.5-flash-lite",
        temperature: float = 0.7
    ):

        self.api_key = api_key or os.getenv("OPENROUTER_SECRET_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_SECRET_KEY not found. ")

        self.model = model
        self.temperature = temperature

        # Initialize OpenAI client pointed at OpenRouter
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None
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
            temperature=temperature or self.temperature
        )

        return response.choices[0].message.content

    def generate_structured(
        self,
        prompt: str,
        schema_name: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> BaseModel:

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
            response_format=response_format
        )

        content = response.choices[0].message.content

        try:
            data = json.loads(content)
            return LLMSchemas.parse_response(schema_name, data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nContent: {content}")
        except ValidationError as e:
            raise ValueError(f"Response validation failed: {e}\nContent: {content}")

class MultiAgentLLM:

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
