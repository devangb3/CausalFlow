"""
LLM Client for OpenRouter API.

This module provides a client for interacting with various LLMs through OpenRouter.
"""

import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMClient:
    """
    Client for making LLM API calls through OpenRouter.

    OpenRouter provides access to multiple LLM models through a unified API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4-turbo-preview",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: OpenRouter API key (if None, reads from OPENROUTER_SECRET_KEY env var)
            model: Model identifier (default: GPT-4 Turbo)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
        """
        self.api_key = api_key or os.getenv("OPENROUTER_SECRET_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_SECRET_KEY not found. "
                "Please set it in your .env file or pass it as api_key parameter."
            )

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
        """
        Generate text from a prompt.

        Args:
            prompt: The user prompt
            system_message: Optional system message to guide behavior
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Generated text response
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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens
        )

        return response.choices[0].message.content

    def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text from a list of messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Generated text response
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens
        )

        return response.choices[0].message.content

    def generate_multiple(
        self,
        prompts: List[str],
        system_message: Optional[str] = None
    ) -> List[str]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of prompts
            system_message: Optional system message for all prompts

        Returns:
            List of generated responses
        """
        return [
            self.generate(prompt, system_message)
            for prompt in prompts
        ]

    def __repr__(self) -> str:
        """String representation of the client."""
        return f"LLMClient(model='{self.model}', temperature={self.temperature})"


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
        """
        Initialize multiple LLM agents.

        Args:
            num_agents: Number of agents to create
            models: Optional list of model names (if None, uses same model for all)
            api_key: OpenRouter API key
        """
        self.num_agents = num_agents

        # Default models if not specified
        if models is None:
            models = ["openai/gpt-4-turbo-preview"] * num_agents
        elif len(models) < num_agents:
            # Extend with the first model if not enough specified
            models = models + [models[0]] * (num_agents - len(models))

        self.agents = [
            LLMClient(api_key=api_key, model=model)
            for model in models[:num_agents]
        ]

    def get_agent(self, index: int) -> LLMClient:
        """
        Get a specific agent by index.

        Args:
            index: Agent index (0-based)

        Returns:
            LLM client for that agent
        """
        if 0 <= index < self.num_agents:
            return self.agents[index]
        raise IndexError(f"Agent index {index} out of range (0-{self.num_agents-1})")

    def generate_all(
        self,
        prompt: str,
        system_message: Optional[str] = None
    ) -> List[str]:
        """
        Generate responses from all agents for the same prompt.

        Args:
            prompt: The prompt to send to all agents
            system_message: Optional system message

        Returns:
            List of responses from each agent
        """
        return [
            agent.generate(prompt, system_message)
            for agent in self.agents
        ]

    def __repr__(self) -> str:
        """String representation."""
        return f"MultiAgentLLM(num_agents={self.num_agents})"
