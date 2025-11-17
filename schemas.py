from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class InterventionOutput(BaseModel):
    """Schema for intervention/correction outputs from causal attribution."""

    corrected_reasoning: Optional[str] = Field(
        None,
        description="The corrected reasoning text for the step"
    )
    corrected_tool_name: Optional[str] = Field(
        None,
        description="The corrected tool name (if applicable)"
    )
    corrected_tool_args: Optional[Dict[str, Any]] = Field(
        None,
        description="The corrected tool arguments as a key-value object"
    )
    explanation: str = Field(
        required=True,
        description="Brief explanation of what was corrected and why"
    )


class OutcomePrediction(BaseModel):
    """Schema for predicting if an intervention would fix the failure."""

    would_succeed: bool = Field(
        required=True,
        description="Whether the intervention would likely fix the failure"
    )
    confidence: float = Field(
        required=True,
        ge=0.0,
        le=1.0,
        description="Confidence level between 0.0 and 1.0"
    )
    reasoning: str = Field(
        required=True,
        description="Brief explanation of the prediction"
    )


class RepairOutput(BaseModel):
    """Schema for counterfactual repair outputs."""

    repaired_text: Optional[str] = Field(
        None,
        description="The minimally repaired text content"
    )
    repaired_tool_name: Optional[str] = Field(
        None,
        description="The repaired tool name (if step is a tool call)"
    )
    repaired_tool_args: Optional[Dict[str, Any]] = Field(
        None,
        description="The repaired tool arguments (if step is a tool call)"
    )
    changes_made: List[str] = Field(
        required=True,
        description="List of specific changes made to achieve minimal repair"
    )
    minimality_justification: str = Field(
        required=True,
        description="Explanation of why this is the minimal necessary change"
    )


class CritiqueOutput(BaseModel):
    """Schema for multi-agent critique outputs."""

    agreement: str = Field(
        required=True,
        pattern="^(AGREE|DISAGREE|PARTIAL)$",
        description="Level of agreement with the causal claim"
    )
    confidence: float = Field(
        required=True,
        ge=0.0,
        le=1.0,
        description="Confidence in this critique between 0.0 and 1.0"
    )
    reasoning: str = Field(
        required=True,
        description="Detailed explanation of the critique"
    )
    alternative_explanation: Optional[str] = Field(
        None,
        description="Alternative causal explanation if disagreeing"
    )
    evidence_strength: str = Field(
        required=True,
        pattern="^(STRONG|MODERATE|WEAK)$",
        description="Strength of evidence for the causal claim"
    )


class ToolArgsOutput(BaseModel):
    """Schema for parsing tool arguments from text."""

    parsed_args: Dict[str, Any] = Field(
        required=True,
        description="Extracted tool arguments as key-value pairs"
    )
    confidence: float = Field(
        required=True,
        ge=0.0,
        le=1.0,
        description="Confidence in the parsing between 0.0 and 1.0"
    )


class LLMSchemas:
    """Collection of schema utilities for structured LLM outputs."""

    # Map schema names to Pydantic models
    SCHEMA_MAP: Dict[str, type[BaseModel]] = {
        "intervention": InterventionOutput,
        "outcome_prediction": OutcomePrediction,
        "repair": RepairOutput,
        "critique": CritiqueOutput,
        "tool_args": ToolArgsOutput,
    }

    @staticmethod
    def get_model(schema_name: str) -> type[BaseModel]:
        if schema_name not in LLMSchemas.SCHEMA_MAP:
            raise ValueError(
                f"Unknown schema: {schema_name}. "
                f"Available: {list(LLMSchemas.SCHEMA_MAP.keys())}"
            )
        return LLMSchemas.SCHEMA_MAP[schema_name]

    @staticmethod
    def get_response_format(schema_name: str) -> Dict[str, Any]:
        model = LLMSchemas.get_model(schema_name)

        # Generate JSON schema from Pydantic model
        json_schema = model.model_json_schema()

        # Remove title and other metadata that might interfere
        json_schema.pop("title", None)
        json_schema.pop("description", None)

        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": json_schema
            }
        }

    @staticmethod
    def parse_response(schema_name: str, response_data: Dict[str, Any]) -> BaseModel:
        """
        Parse and validate a response using the appropriate Pydantic model.

        Args:
            schema_name: Name of the schema
            response_data: Raw JSON response data

        Returns:
            Validated Pydantic model instance

        Raises:
            ValueError: If schema_name is unknown or validation fails
        """
        model = LLMSchemas.get_model(schema_name)
        return model(**response_data)
