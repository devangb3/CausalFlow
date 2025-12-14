import os
import json
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from schemas import LLMSchemas
from text_processor import convert_text_to_jsonl

load_dotenv()

class LLMClient:

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "google/gemini-2.5-flash",
        temperature: float = 0.7
    ):

        self.api_key = api_key or os.getenv("OPENROUTER_SECRET_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_SECRET_KEY not found. ")

        self.model = model
        self.temperature = temperature

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
            temperature=temperature or self.temperature,
        )
        return response.choices[0].message.content

    def generate_structured(
        self,
        prompt: str,
        schema_name: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        model_name: Optional[str] = None
    ) -> BaseModel:
        llm_model_name = model_name or self.model
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

        response_format = LLMSchemas.get_response_format(schema_name)

        response = self.client.chat.completions.create(
            model=llm_model_name,
            messages=messages,
            temperature=temperature or self.temperature,
            response_format=response_format
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError("Empty response from LLM")
        try:
            parsed_objects = convert_text_to_jsonl(content)
            
            if not parsed_objects:
                raise ValueError("No valid JSON objects found in response")
            data = parsed_objects[0]
            
            if not isinstance(data, dict):
                raise ValueError(f"Expected JSON object, got {type(data)}")
            
            if "mode" in data and "text" in data and data.get("mode") == "text":
                try:
                    data = json.loads(content.strip())
                except json.JSONDecodeError:
                    raise ValueError("Response is plain text, not valid JSON")
            
            return LLMSchemas.parse_response(schema_name, data)
            
        except ValueError as e:
            content = content.strip()
            data = data
            raise ValueError(f"Failed to parse JSON response: {e}")

class MultiAgentLLM:

    def __init__(
        self,
        num_agents: int = 3,
        models: Optional[List[str]] = None,
        api_key: Optional[str] = None
    ):
        self.num_agents = num_agents

        if models is None:
            models = ["google/gemini-2.5-flash"] * num_agents
        elif len(models) < num_agents:
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
