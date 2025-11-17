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
        ...,
        description="Brief explanation of what was corrected and why"
    )


class OutcomePrediction(BaseModel):
    """Schema for predicting if an intervention would fix the failure."""

    would_succeed: bool = Field(
        ...,
        description="Whether the intervention would likely fix the failure"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level between 0.0 and 1.0"
    )
    reasoning: str = Field(
        ...,
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
        ...,
        description="List of specific changes made to achieve minimal repair"
    )
    minimality_justification: str = Field(
        ...,
        description="Explanation of why this is the minimal necessary change"
    )


class CritiqueOutput(BaseModel):
    """Schema for multi-agent critique outputs."""

    agreement: str = Field(
        ...,
        pattern="^(AGREE|DISAGREE|PARTIAL)$",
        description="Level of agreement with the causal claim"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in this critique between 0.0 and 1.0"
    )
    reasoning: str = Field(
        ...,
        description="Detailed explanation of the critique"
    )
    alternative_explanation: Optional[str] = Field(
        None,
        description="Alternative causal explanation if disagreeing"
    )
    evidence_strength: str = Field(
        ...,
        pattern="^(STRONG|MODERATE|WEAK)$",
        description="Strength of evidence for the causal claim"
    )


class ToolArgsOutput(BaseModel):
    """Schema for parsing tool arguments from text."""

    parsed_args: Dict[str, Any] = Field(
        ...,
        description="Extracted tool arguments as key-value pairs"
    )
    confidence: float = Field(
        ...,
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

        # Generate JSON schema from Pydantic model with mode='serialization'
        # This produces cleaner schemas compatible with more providers
        json_schema = model.model_json_schema(mode='serialization')

        # Clean up schema for compatibility with Google Gemini and other providers
        # Remove $defs and flatten the schema
        if "$defs" in json_schema:
            defs = json_schema.pop("$defs")
            # Inline any references
            LLMSchemas._inline_refs(json_schema, defs)

        # Simplify schema for Google Gemini compatibility
        LLMSchemas._simplify_schema(json_schema)

        # Remove metadata that might interfere
        json_schema.pop("title", None)
        json_schema.pop("description", None)

        # Ensure additionalProperties is set
        if "additionalProperties" not in json_schema:
            json_schema["additionalProperties"] = False

        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": json_schema
            }
        }

    @staticmethod
    def _simplify_schema(schema: Dict[str, Any]) -> None:
        """
        Simplify schema for Google Gemini compatibility.
        Removes anyOf, oneOf, allOf, and simplifies optional fields.
        Modifies schema in-place.

        Args:
            schema: The schema object to simplify
        """
        if isinstance(schema, dict):
            # Remove title from all properties
            schema.pop("title", None)

            # Handle anyOf for optional fields (common pattern: anyOf with null)
            if "anyOf" in schema:
                any_of = schema.pop("anyOf")
                # Find the non-null type
                for option in any_of:
                    if option.get("type") != "null":
                        # Use the non-null type and mark as nullable
                        schema.update(option)
                        # Keep default if it exists
                        break

            # Handle oneOf similarly
            if "oneOf" in schema:
                one_of = schema.pop("oneOf")
                # Just use the first non-null option
                for option in one_of:
                    if option.get("type") != "null":
                        schema.update(option)
                        break

            # Recursively process nested objects
            for key, value in list(schema.items()):
                if isinstance(value, dict):
                    LLMSchemas._simplify_schema(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            LLMSchemas._simplify_schema(item)

    @staticmethod
    def _inline_refs(schema: Dict[str, Any], defs: Dict[str, Any]) -> None:
        """
        Recursively inline $ref references in a schema.
        Modifies schema in-place.

        Args:
            schema: The schema object to process
            defs: The definitions to inline
        """
        if isinstance(schema, dict):
            if "$ref" in schema:
                # Extract the reference name
                ref = schema["$ref"]
                if ref.startswith("#/$defs/"):
                    def_name = ref.replace("#/$defs/", "")
                    if def_name in defs:
                        # Replace the reference with the actual definition
                        definition = defs[def_name].copy()
                        schema.clear()
                        schema.update(definition)
                        # Recursively inline any nested refs
                        LLMSchemas._inline_refs(schema, defs)
            else:
                # Recursively process nested objects
                for key, value in list(schema.items()):
                    if isinstance(value, dict):
                        LLMSchemas._inline_refs(value, defs)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                LLMSchemas._inline_refs(item, defs)

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
