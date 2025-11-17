"""
Centralized JSON schemas for structured outputs from LLM API calls.
All schemas follow OpenRouter's structured outputs format.
"""

from typing import Dict, Any


class LLMSchemas:
    """Collection of JSON schemas for structured LLM outputs."""

    @staticmethod
    def intervention_schema() -> Dict[str, Any]:
        """Schema for intervention/correction outputs from causal attribution."""
        return {
            "type": "object",
            "properties": {
                "corrected_reasoning": {
                    "type": "string",
                    "description": "The corrected reasoning text for the step"
                },
                "corrected_tool_name": {
                    "type": "string",
                    "description": "The corrected tool name (if applicable)"
                },
                "corrected_tool_args": {
                    "type": "object",
                    "description": "The corrected tool arguments as a key-value object",
                    "additionalProperties": True
                },
                "explanation": {
                    "type": "string",
                    "description": "Brief explanation of what was corrected and why"
                }
            },
            "required": ["explanation"],
            "additionalProperties": False
        }

    @staticmethod
    def outcome_prediction_schema() -> Dict[str, Any]:
        """Schema for predicting if an intervention would fix the failure."""
        return {
            "type": "object",
            "properties": {
                "would_succeed": {
                    "type": "boolean",
                    "description": "Whether the intervention would likely fix the failure"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence level between 0.0 and 1.0",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of the prediction"
                }
            },
            "required": ["would_succeed", "confidence", "reasoning"],
            "additionalProperties": False
        }

    @staticmethod
    def repair_schema() -> Dict[str, Any]:
        """Schema for counterfactual repair outputs."""
        return {
            "type": "object",
            "properties": {
                "repaired_text": {
                    "type": "string",
                    "description": "The minimally repaired text content"
                },
                "repaired_tool_name": {
                    "type": "string",
                    "description": "The repaired tool name (if step is a tool call)"
                },
                "repaired_tool_args": {
                    "type": "object",
                    "description": "The repaired tool arguments (if step is a tool call)",
                    "additionalProperties": True
                },
                "changes_made": {
                    "type": "array",
                    "description": "List of specific changes made to achieve minimal repair",
                    "items": {
                        "type": "string"
                    }
                },
                "minimality_justification": {
                    "type": "string",
                    "description": "Explanation of why this is the minimal necessary change"
                }
            },
            "required": ["changes_made", "minimality_justification"],
            "additionalProperties": False
        }

    @staticmethod
    def critique_schema() -> Dict[str, Any]:
        """Schema for multi-agent critique outputs."""
        return {
            "type": "object",
            "properties": {
                "agreement": {
                    "type": "string",
                    "enum": ["AGREE", "DISAGREE", "PARTIAL"],
                    "description": "Level of agreement with the causal claim"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence in this critique between 0.0 and 1.0",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "reasoning": {
                    "type": "string",
                    "description": "Detailed explanation of the critique"
                },
                "alternative_explanation": {
                    "type": "string",
                    "description": "Alternative causal explanation if disagreeing"
                },
                "evidence_strength": {
                    "type": "string",
                    "enum": ["STRONG", "MODERATE", "WEAK"],
                    "description": "Strength of evidence for the causal claim"
                }
            },
            "required": ["agreement", "confidence", "reasoning", "evidence_strength"],
            "additionalProperties": False
        }

    @staticmethod
    def tool_args_schema() -> Dict[str, Any]:
        """Schema for parsing tool arguments from text."""
        return {
            "type": "object",
            "properties": {
                "parsed_args": {
                    "type": "object",
                    "description": "Extracted tool arguments as key-value pairs",
                    "additionalProperties": True
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence in the parsing between 0.0 and 1.0",
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            },
            "required": ["parsed_args", "confidence"],
            "additionalProperties": False
        }

    @staticmethod
    def get_response_format(schema_name: str) -> Dict[str, Any]:
        """
        Get the complete response_format object for OpenRouter API.

        Args:
            schema_name: Name of the schema (e.g., 'intervention', 'repair', 'critique')

        Returns:
            Complete response_format object for API call
        """
        schema_map = {
            "intervention": LLMSchemas.intervention_schema(),
            "outcome_prediction": LLMSchemas.outcome_prediction_schema(),
            "repair": LLMSchemas.repair_schema(),
            "critique": LLMSchemas.critique_schema(),
            "tool_args": LLMSchemas.tool_args_schema(),
        }

        if schema_name not in schema_map:
            raise ValueError(f"Unknown schema: {schema_name}. Available: {list(schema_map.keys())}")

        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema_map[schema_name]
            }
        }
