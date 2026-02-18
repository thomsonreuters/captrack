"""  
JSON schema compliance evaluation metrics.
  
This module evaluates whether AI model outputs conform to specified JSON schemas.  
It handles various response formats (with/without markdown code fences), extracts  
JSON content, and validates it against provided schemas using jsonschema validation.  
"""
  
import json  
import jsonschema  
from jsonschema import validate, ValidationError  
import pandas as pd  
import re  
from typing import Dict, Any, Optional

  
def json_schema_score(results_df: pd.DataFrame) -> float:  
    """  
    Compute accuracy score for JSON schema compliance.
      
    Evaluates what fraction of model responses contain valid JSON that conforms  
    to the specified schema. This is a simplified version that only returns  
    the overall accuracy score.
  
    Args:  
        results_df: DataFrame with columns:  
            - 'outputs': Model responses (strings that may contain JSON)  
            - 'schema': JSON schemas to validate against (dict or JSON string)
  
    Returns:  
        Accuracy score (0.0 to 1.0) representing fraction of responses that  
        contain valid JSON conforming to the schema  
    """  
    responses = results_df['outputs']  
    schemas = results_df['schema']
  
    valid_count = 0  
    total_count = len(responses)
  
    for i, (response, schema) in enumerate(zip(responses, schemas)):  
        if is_valid_json_schema(response, schema):  
            valid_count += 1
  
    accuracy = valid_count / total_count if total_count > 0 else 0.0  
    return accuracy

  
def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:  
    """  
    Extract JSON object from model response, handling various formats.
      
    Handles common response formats including:  
    - Plain JSON objects  
    - JSON wrapped in markdown code fences (```json ... ```)  
    - JSON with "JSON:" prefix  
    - JSON embedded within other text
      
    Uses regex pattern matching to find JSON objects when direct parsing fails.
  
    Args:  
        response: Model output string that may contain JSON
  
    Returns:  
        Parsed JSON object as dictionary, or None if no valid JSON found  
    """  
    if not isinstance(response, str):  
        return None
  
    # Remove common prefixes/suffixes and code fences  
    response = response.strip()  
    response = re.sub(r'^```json\s*|\s*```$', '', response, flags=re.MULTILINE)  
    response = re.sub(r'^JSON:\s*', '', response, flags=re.IGNORECASE)  
    response = response.strip()
  
    # Try to parse the entire response as JSON first  
    try:  
        return json.loads(response)  
    except json.JSONDecodeError:  
        pass
  
    # Try to find JSON object within the response using pattern matching  
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'  
    matches = re.findall(json_pattern, response, re.DOTALL)
  
    for match in matches:  
        try:  
            return json.loads(match)  
        except json.JSONDecodeError:  
            continue
  
    return None

  
def parse_schema(schema_input) -> Optional[Dict[str, Any]]:  
    """  
    Parse schema from various input formats.
      
    Handles schemas provided as dictionaries or JSON strings, converting  
    them to the dictionary format required by jsonschema validation.
  
    Args:  
        schema_input: Schema as dict or JSON string
  
    Returns:  
        Schema as dictionary, or None if parsing fails  
    """  
    if isinstance(schema_input, dict):  
        return schema_input  
    elif isinstance(schema_input, str):  
        try:  
            return json.loads(schema_input)  
        except json.JSONDecodeError:  
            return None  
    else:  
        return None

  
def is_valid_json_schema(response: str, schema) -> bool:  
    """  
    Check if response contains valid JSON that conforms to the given schema.
      
    Performs a three-step validation process:  
    1. Extracts JSON from the response (handling various formats)  
    2. Parses the schema into the required format  
    3. Validates the JSON against the schema using jsonschema library
  
    Args:  
        response: Model output string that may contain JSON  
        schema: JSON schema to validate against (dict or JSON string)
  
    Returns:  
        True if response contains valid JSON conforming to schema, False otherwise  
    """  
    # Extract JSON from response  
    json_data = extract_json_from_response(response)  
    if json_data is None:  
        return False
  
    # Parse schema  
    schema_dict = parse_schema(schema)  
    if schema_dict is None:  
        return False
  
    # Validate against schema  
    try:  
        validate(instance=json_data, schema=schema_dict)  
        return True  
    except ValidationError:  
        return False  
    except Exception:  
        # Handle any other validation errors  
        return False

  
def json_schema_score_detailed(results_df: pd.DataFrame) -> Dict[str, Any]:  
    """  
    Compute accuracy score with detailed breakdown of failures.
      
    Provides comprehensive diagnostics by categorizing failures into:  
    - invalid_json: Response doesn't contain parseable JSON  
    - schema_parse_error: The schema itself couldn't be parsed  
    - schema_validation_failed: JSON is valid but doesn't match schema
      
    This extended version is useful for debugging and understanding  
    common failure modes in structured output generation.
  
    Args:  
        results_df: DataFrame with columns:  
            - 'outputs': Model responses (strings that may contain JSON)  
            - 'schema': JSON schemas to validate against (dict or JSON string)
  
    Returns:  
        Dictionary containing:  
        - accuracy: Overall accuracy score (0.0 to 1.0)  
        - valid_count: Number of valid responses  
        - total_count: Total number of responses evaluated  
        - failure_reasons: Dict with counts for each failure category  
            - invalid_json: Count of responses with no parseable JSON  
            - schema_parse_error: Count of unparseable schemas  
            - schema_validation_failed: Count of JSON that failed schema validation  
    """  
    responses = results_df['outputs']  
    schemas = results_df['schema']
  
    valid_count = 0  
    total_count = len(responses)  
    failure_reasons = {  
        'invalid_json': 0,  
        'schema_validation_failed': 0,  
        'schema_parse_error': 0  
    }
  
    for i, (response, schema) in enumerate(zip(responses, schemas)):  
        # Extract JSON  
        json_data = extract_json_from_response(response)  
        if json_data is None:  
            failure_reasons['invalid_json'] += 1  
            continue
  
        # Parse schema  
        schema_dict = parse_schema(schema)  
        if schema_dict is None:  
            failure_reasons['schema_parse_error'] += 1  
            continue
  
        # Validate against schema  
        try:  
            validate(instance=json_data, schema=schema_dict)  
            valid_count += 1  
        except ValidationError:  
            failure_reasons['schema_validation_failed'] += 1  
        except Exception:  
            failure_reasons['schema_validation_failed'] += 1
  
    accuracy = valid_count / total_count if total_count > 0 else 0.0
  
    return {  
        'accuracy': accuracy,  
        'valid_count': valid_count,  
        'total_count': total_count,  
        'failure_reasons': failure_reasons  
    }  