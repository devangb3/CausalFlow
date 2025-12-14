from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SamplerResponse:
    response_text: str
    actual_queried_message_list: List[Dict[str, str]]


class SamplerBase(ABC):

    
    @abstractmethod
    def __call__(self, message_list: List[Dict[str, str]]) -> SamplerResponse:
        """Generate a response from the model.
        
        Args:
            message_list: List of messages in OpenAI format (role, content).
            
        Returns:
            SamplerResponse with the model's response.
        """
        raise NotImplementedError
    
    @abstractmethod
    def _pack_message(self, content: str, role: str = "user") -> Dict[str, str]:
        """Pack content into a message dict.
        
        Args:
            content: The message content.
            role: The message role (user, assistant, system).
            
        Returns:
            Message dict with role and content.
        """
        raise NotImplementedError


@dataclass
class SingleEvalResult:
    """Result for a single evaluation example."""
    html: str
    score: float
    convo: List[Dict[str, str]]
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Aggregated evaluation results."""
    score: float
    metrics: Dict[str, Any]
    htmls: List[str]
    convos: List[List[Dict[str, str]]]


class Eval(ABC):
    """Base class for evaluations."""
    
    @abstractmethod
    def __call__(self, sampler: SamplerBase) -> EvalResult:
        """Run the evaluation with the given sampler.
        
        Args:
            sampler: Model sampler to evaluate.
            
        Returns:
            EvalResult with scores and metrics.
        """
        raise NotImplementedError
