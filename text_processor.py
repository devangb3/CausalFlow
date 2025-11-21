import json
import re
from typing import List, Dict, Any

def convert_text_to_jsonl(text: str) -> List[Dict[str, Any]]:
    """
    Convert text output from model to JSONL format.
    Handles various formats the model might output:
    - Raw JSON array
    - JSON wrapped in markdown code blocks
    - JSONL format (one JSON object per line)
    - Mixed text with JSON content
    """
    if not text or not text.strip():
        return []
    
    cleaned_text = text.strip()
    
    strategies = [
        ("markdown_json", _parse_markdown_json),
        ("json_array", _parse_json_array),
        ("jsonl_lines", _parse_jsonl_lines),
        ("mixed_content", _parse_mixed_content),
        ("single_json_objects", _parse_single_json_objects),
        ("text", _parse_text_to_json)
    ]
    
    for _, strategy_func in strategies:
        try:
            result = strategy_func(cleaned_text)
            if result:
                return result
        except Exception as e:
            continue
    print(f"Failed to parse text: {text}")
    return []


def _parse_markdown_json(text: str) -> List[Dict[str, Any]]:
    """Parse JSON wrapped in markdown code blocks"""
    # Look for ```json ... ``` or ``` ... ``` patterns
    json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
        except json.JSONDecodeError:
            continue
    
    raise ValueError()


def _parse_json_array(text: str) -> List[Dict[str, Any]]:
    """Parse direct JSON array"""
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise ValueError
    except json.JSONDecodeError:
        raise ValueError()


def _parse_jsonl_lines(text: str) -> List[Dict[str, Any]]:
    """Parse JSONL format (one JSON object per line)"""
    lines = text.strip().split('\n')
    results = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        try:
            data = json.loads(line)
            results.append(data)
        except json.JSONDecodeError:
            pass
    
    if results:
        return results
    else:
        raise ValueError()


def _parse_mixed_content(text: str) -> List[Dict[str, Any]]:
    """Parse mixed content by extracting JSON objects"""
    json_objects = []
    
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    potential_jsons = re.findall(json_pattern, text, re.DOTALL)
    
    for potential_json in potential_jsons:
        try:
            data = json.loads(potential_json)
            if isinstance(data, dict):
                json_objects.append(data)
        except json.JSONDecodeError:
            continue
    
    if json_objects:
        return json_objects
    else:
        raise ValueError()


def _parse_single_json_objects(text: str) -> List[Dict[str, Any]]:
    """Parse individual JSON objects separated by newlines or other delimiters"""
    delimiters = ['\n\n', '\n---\n', '\n***\n', '\n---', '\n***']
    
    for delimiter in delimiters:
        parts = text.split(delimiter)
        results = []
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            try:
                data = json.loads(part)
                if isinstance(data, dict):
                    results.append(data)
                elif isinstance(data, list):
                    results.extend(data)
            except json.JSONDecodeError:
                continue
        
        if results:
            return results
    raise ValueError()
def _parse_text_to_json(text: str) -> List[Dict[str, Any]]:
       return [{"mode": "text", "text": text}]