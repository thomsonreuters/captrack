"""  
Tool use and function calling evaluation metrics.
  
This module evaluates AI models' ability to correctly select and invoke tools/functions  
with appropriate arguments. Supports BFCL (Berkeley Function Calling Leaderboard) and  
MNMS (Multi-Node Multi-Step) dataset formats. Evaluates selection accuracy (correct tool),  
argument accuracy (correct parameters), and integration accuracy (both correct).  
"""
  
import re    
import json    
from typing import Dict, List, Any, Optional, Tuple  
import pandas as pd

  
def compute_bfcl_accuracy(  
    results_df: pd.DataFrame,  
    dataset_format: str = 'bfcl',  
    report_detailed: bool = False,  
    debug: bool = False  
) -> Dict[str, Any]:  
    """  
    Evaluate tool and function calling accuracy on BFCL and MNMS datasets.
  
    Evaluates three key dimensions:  
    1. Selection Accuracy: Whether the correct tool/function was chosen  
    2. Argument Accuracy: Whether correct arguments with proper values were provided  
    3. Integration Accuracy: Overall correctness (both selection and arguments correct)
  
    Also provides per-tool performance breakdown and detailed error analysis  
    categorizing failures by type (wrong tool, missing args, incorrect args, etc.).
  
    Args:  
        results_df: DataFrame with columns:  
            - 'outputs': Model-generated function calls  
            - 'gold': Gold standard function calls  
            - 'id': Optional problem identifier  
        dataset_format: Either 'bfcl' or 'mnms' to specify gold format parsing  
        report_detailed: If True, include per-sample detailed results with parsed calls  
        debug: If True, print detailed parsing information for each sample
  
    Returns:  
        Dictionary containing:  
        - selection_accuracy: Fraction of correct tool selections (0.0-1.0)  
        - argument_accuracy: Fraction of correct argument sets (0.0-1.0)  
        - integration_accuracy: Fraction of fully correct calls (0.0-1.0)  
        - selection_correct: Count of correct tool selections  
        - argument_correct: Count of correct argument sets  
        - integration_correct: Count of fully correct calls  
        - total_items: Total number of function calls evaluated  
        - tool_metrics: Per-tool performance breakdown dict  
        - tool_count: Number of unique tools evaluated  
        - error_analysis: Dict with counts for each error type  
        - error_rate: Overall error rate (0.0-1.0)  
        - perfect_calls: Count of perfect function calls  
        - partial_success_rate: Rate of partially correct calls  
        - detailed_results: Per-sample results (if report_detailed=True)  
    """
  
    if len(results_df) == 0:    
        return get_empty_bfcl_metrics()
  
    # Initialize counters    
    selection_correct = 0    
    argument_correct = 0    
    integration_correct = 0    
    total_items = 0
  
    detailed_results = []    
    tool_performance = {}    
    error_analysis = {    
        'parsing_errors': 0,    
        'wrong_tool': 0,    
        'missing_args': 0,    
        'incorrect_args': 0,    
        'format_errors': 0    
    }
  
    for idx, row in results_df.iterrows():    
        try:    
            problem_id = row.get('id', idx)    
            model_output = row.get('outputs', '').strip()    
            gold_calls = row.get('gold', '')
  
            # Parse function calls based on dataset format    
            if dataset_format == 'mnms':    
                gold_parsed = parse_mnms_gold_format(gold_calls)    
            else:  # bfcl    
                gold_parsed = parse_function_calls(gold_calls)
                
            model_parsed = parse_function_calls(model_output)
  
            if debug:  
                print(f'gold: {gold_calls}')  
                print(f'model: {model_output}')  
                print(f'gold parsed: {gold_parsed}')  
                print(f'model parsed: {model_parsed}')
  
            # Align lengths for comparison (pad shorter list with None)  
            if len(gold_parsed) > len(model_parsed):    
                diff = len(gold_parsed) - len(model_parsed)    
                model_parsed += [None] * diff    
            elif len(gold_parsed) < len(model_parsed):    
                diff = len(model_parsed) - len(gold_parsed)    
                gold_parsed += [None] * diff
  
            # Track per-call metrics for this row    
            row_selection_correct = 0    
            row_argument_correct = 0    
            row_integration_correct = 0
  
            # Handle case where both gold and model have no function calls  
            if len(gold_parsed) == 0 and len(model_parsed) == 0:  
                selection_correct += 1    
                row_selection_correct += 1  
                argument_correct += 1    
                row_argument_correct += 1  
                integration_correct += 1    
                row_integration_correct += 1  
                error_type = None  
                total_items += 1
  
            # Evaluate each function call pair  
            for model, gold in zip(model_parsed, gold_parsed):
  
                # Evaluate selection accuracy    
                selection_acc = evaluate_selection_accuracy(model, gold)    
                if selection_acc:    
                    selection_correct += 1    
                    row_selection_correct += 1
  
                # Evaluate argument accuracy (use dataset-specific evaluator)  
                if dataset_format == 'mnms':    
                    argument_acc = evaluate_argument_accuracy_mnms(model, gold)    
                else:    
                    argument_acc = evaluate_argument_accuracy(model, gold)    
                if argument_acc:    
                    argument_correct += 1    
                    row_argument_correct += 1
  
                # Evaluate integration accuracy (both selection and arguments correct)  
                integration_acc = selection_acc and argument_acc    
                if integration_acc:    
                    integration_correct += 1    
                    row_integration_correct += 1
  
                # Track per-tool performance statistics  
                if gold and gold.get('function_name'):    
                    tool_name = gold['function_name']    
                    if tool_name not in tool_performance:    
                        tool_performance[tool_name] = {    
                            'total': 0, 'selection_correct': 0,    
                            'argument_correct': 0, 'integration_correct': 0    
                        }
  
                    tool_performance[tool_name]['total'] += 1    
                    if selection_acc:    
                        tool_performance[tool_name]['selection_correct'] += 1    
                    if argument_acc:    
                        tool_performance[tool_name]['argument_correct'] += 1    
                    if integration_acc:    
                        tool_performance[tool_name]['integration_correct'] += 1
  
                # Categorize error type for analysis  
                error_type = analyze_error_type(model, gold, model_output)    
                if error_type in error_analysis:    
                    error_analysis[error_type] += 1
  
                total_items += 1
  
            # Store detailed results for this sample  
            detailed_results.append({    
                'id': problem_id,    
                'selection_correct': row_selection_correct,    
                'argument_correct': row_argument_correct,    
                'integration_correct': row_integration_correct,    
                'num_calls': len(gold_parsed),    
                'model_output': model_output,    
                'gold_calls': gold_calls,    
                'model_parsed': model_parsed,    
                'gold_parsed': gold_parsed,    
                'error_type': error_type    
            })
  
        except Exception as e:    
            # Log parsing errors and continue  
            error_analysis['parsing_errors'] += 1    
            detailed_results.append({    
                'id': row.get('id', idx),    
                'selection_correct': 0,    
                'argument_correct': 0,    
                'integration_correct': 0,    
                'error': str(e)    
            })    
            total_items += 1
  
    # Calculate aggregate metrics    
    selection_accuracy = selection_correct / total_items if total_items > 0 else 0.0    
    argument_accuracy = argument_correct / total_items if total_items > 0 else 0.0    
    integration_accuracy = integration_correct / total_items if total_items > 0 else 0.0
  
    # Calculate per-tool metrics    
    tool_metrics = {}    
    for tool_name, stats in tool_performance.items():    
        if stats['total'] > 0:    
            tool_metrics[tool_name] = {    
                'selection_accuracy': stats['selection_correct'] / stats['total'],    
                'argument_accuracy': stats['argument_correct'] / stats['total'],    
                'integration_accuracy': stats['integration_correct'] / stats['total'],    
                'count': stats['total']    
            }
  
    return {    
        # Primary metrics    
        'selection_accuracy': selection_accuracy,    
        'argument_accuracy': argument_accuracy,    
        'integration_accuracy': integration_accuracy,
  
        # Counts    
        'selection_correct': selection_correct,    
        'argument_correct': argument_correct,    
        'integration_correct': integration_correct,    
        'total_items': total_items,
  
        # Per-tool performance    
        'tool_metrics': tool_metrics,    
        'tool_count': len(tool_performance),
  
        # Error analysis    
        'error_analysis': error_analysis,    
        'error_rate': sum(error_analysis.values()) / total_items if total_items > 0 else 0.0,
  
        # Additional insights    
        'perfect_calls': integration_correct,    
        'partial_success_rate': (selection_correct + argument_correct - integration_correct) / total_items if total_items > 0 else 0.0,
  
        # Detailed results    
        'detailed_results': detailed_results if report_detailed else None    
    }

  
def parse_mnms_gold_format(gold_data: Any) -> List[Dict[str, Any]]:  
    """  
    Parse MNMS gold format into standardized function call format.
      
    MNMS uses a structured format with explicit call IDs and task names:  
    [{'id': 0, 'name': 'text classification', 'args': {'text': '...'}}]
      
    This function converts it to the standardized format used by the evaluator,  
    normalizing task names (replacing spaces with underscores) and preserving  
    the MNMS call ID for tracking.
  
    Args:  
        gold_data: MNMS gold format as list of dicts, or string representation  
                  that can be parsed as JSON or Python literal
  
    Returns:  
        List of standardized function call dictionaries with keys:  
        - function_name: Normalized function name (spaces replaced with _)  
        - arguments: Dict of argument name to value mappings  
        - raw_call: Original call string for debugging  
        - parse_error: None if successful, error message if failed  
        - mnms_id: Original MNMS call ID for tracking  
    """  
    if not gold_data:    
        return []
        
    # Handle string representation of list    
    if isinstance(gold_data, str):    
        try:    
            gold_data = json.loads(gold_data)    
        except json.JSONDecodeError:    
            # Try to evaluate as Python literal    
            try:    
                import ast    
                gold_data = ast.literal_eval(gold_data)    
            except:    
                return []    
        
    # Ensure it's a list    
    if not isinstance(gold_data, list):    
        gold_data = [gold_data]
        
    # Convert MNMS format to standardized format    
    standardized_calls = []    
    for call_dict in gold_data:    
        if not isinstance(call_dict, dict):    
            continue
            
        standardized = {    
            'function_name': call_dict.get('name', '').replace(" ", "_"),    
            'arguments': call_dict.get('args', {}),    
            'raw_call': str(call_dict),    
            'parse_error': None,    
            'mnms_id': call_dict.get('id')  # Preserve MNMS call ID    
        }
            
        standardized_calls.append(standardized)
        
    return standardized_calls

  
def parse_function_calls(calls_text: Any) -> List[Dict[str, Any]]:  
    """  
    Parse multiple function calls from text or list input.
      
    Handles various input formats:  
    - String with multiple calls separated by newlines  
    - List of call strings  
    - List of already-parsed dictionaries  
    - Special values like "NONE" or "no function call"
      
    Args:  
        calls_text: Function calls as string, list of strings, or list of dicts
  
    Returns:  
        List of parsed function call dictionaries, empty list if no valid calls  
    """  
    if not calls_text or calls_text == "NONE" or (isinstance(calls_text, list) and len(calls_text) > 0 and calls_text[0].lower() == "no function call"):  
        return []
  
    # Handle list input (already parsed or list of strings)  
    if isinstance(calls_text, list):    
        # Check if it's already in parsed dict format    
        if calls_text and isinstance(calls_text[0], dict) and 'function_name' in calls_text[0]:    
            return calls_text    
        # Otherwise treat as list of call strings    
        lines = calls_text    
    else:    
        # Split by lines and parse each call    
        lines = [line.strip() for line in calls_text.split('\n') if line.strip()]
        
    parsed_calls = []    
    for line in lines:    
        parsed = parse_function_call(line)  
        if parsed:    
            parsed_calls.append(parsed)
        
    return parsed_calls

  
def parse_function_call(call_str: str) -> Optional[Dict[str, Any]]:  
    """  
    Parse a single function call string into structured components.
      
    Extracts function name and arguments from standard function call syntax:  
    function_name(arg1=val1, arg2=val2, ...)
      
    Handles parsing errors gracefully by returning a dict with parse_error field.
  
    Args:  
        call_str: Function call string in standard syntax
  
    Returns:  
        Dictionary with keys:  
        - function_name: Extracted function name (None if parsing failed)  
        - arguments: Dict of parsed arguments  
        - raw_call: Original call string for debugging  
        - parse_error: None if successful, error message otherwise
          
        Returns None if call_str is empty or "NONE"  
    """  
    if not call_str or call_str.strip() == "NONE":    
        return None
  
    call_str = call_str.strip()
  
    # Extract function name and arguments using regex    
    # Pattern: function_name(arg1=val1, arg2=val2, ...)    
    pattern = r'^([a-zA-Z_][a-zA-Z0-9_\s]*)\s*\((.*)\)$'    
    match = re.match(pattern, call_str)
  
    if not match:    
        return {    
            'function_name': None,    
            'arguments': {},    
            'raw_call': call_str,    
            'parse_error': 'Invalid function call format'    
        }
  
    function_name = match.group(1).strip()    
    args_str = match.group(2).strip()
  
    # Parse arguments    
    arguments = {}    
    if args_str:    
        try:    
            arguments = parse_arguments(args_str)    
        except Exception as e:    
            return {    
                'function_name': function_name,    
                'arguments': {},    
                'raw_call': call_str,    
                'parse_error': f'Argument parsing failed: {str(e)}'    
            }
  
    return {    
        'function_name': function_name,    
        'arguments': arguments,    
        'raw_call': call_str,    
        'parse_error': None    
    }

  
def parse_arguments(args_str: str) -> Dict[str, Any]:  
    """  
    Parse argument string into dictionary of key-value pairs.
      
    Handles complex argument structures including nested lists, dicts,  
    and quoted strings. Splits arguments by commas while respecting  
    nested structures and quoted strings.
  
    Args:  
        args_str: Argument string like "arg1=val1, arg2=val2, arg3=[1,2,3]"
  
    Returns:  
        Dictionary mapping argument names to parsed values  
    """  
    arguments = {}
  
    # Split by comma, respecting nested structures    
    arg_parts = split_arguments(args_str)
  
    for part in arg_parts:    
        part = part.strip()    
        if '=' not in part:    
            continue
  
        key, value = part.split('=', 1)    
        key = key.strip()    
        value = value.strip()
  
        # Parse value into appropriate Python type  
        parsed_value = parse_value(value)    
        arguments[key] = parsed_value
  
    return arguments

  
def split_arguments(args_str: str) -> List[str]:  
    """  
    Split arguments by comma while respecting nested structures and quotes.
      
    Tracks nesting depth for parentheses, brackets, and quote state to  
    avoid splitting on commas inside nested structures or quoted strings.  
    This ensures "func(a=[1,2,3], b='hello, world')" splits correctly.
  
    Args:  
        args_str: Argument string potentially containing nested structures
  
    Returns:  
        List of individual argument strings (e.g., ["a=[1,2,3]", "b='hello, world'"])  
    """  
    parts = []    
    current_part = ""    
    paren_depth = 0    
    bracket_depth = 0    
    in_quotes = False    
    quote_char = None
  
    for char in args_str:    
        if char in ['"', "'"] and not in_quotes:    
            in_quotes = True    
            quote_char = char    
        elif char == quote_char and in_quotes:    
            in_quotes = False    
            quote_char = None    
        elif not in_quotes:    
            if char == '(':    
                paren_depth += 1    
            elif char == ')':    
                paren_depth -= 1    
            elif char == '[':    
                bracket_depth += 1    
            elif char == ']':    
                bracket_depth -= 1    
            elif char == ',' and paren_depth == 0 and bracket_depth == 0:    
                # Found a top-level comma - split here  
                parts.append(current_part.strip())    
                current_part = ""    
                continue
  
        current_part += char
  
    # Add the last part  
    if current_part.strip():    
        parts.append(current_part.strip())
  
    return parts

  
def parse_value(value_str: str) -> Any:  
    """  
    Parse a value string into appropriate Python type.
      
    Attempts to parse values in order of specificity:  
    1. Quoted strings -> str (remove quotes)  
    2. Booleans -> bool (true/false)  
    3. None/null -> None  
    4. Integers -> int  
    5. Floats -> float  
    6. Lists -> list (recursive parsing)  
    7. Dicts -> dict (JSON parsing)  
    8. Default -> str (return as-is)
  
    Args:  
        value_str: String representation of a value
  
    Returns:  
        Parsed value in appropriate Python type  
    """  
    value_str = value_str.strip()
  
    # Handle quoted strings    
    if (value_str.startswith('"') and value_str.endswith('"')) or (value_str.startswith("'") and value_str.endswith("'")):  
        return value_str[1:-1]  # Remove quotes
  
    # Handle boolean values    
    if value_str.lower() == 'true':    
        return True    
    elif value_str.lower() == 'false':    
        return False    
    elif value_str.lower() == 'none' or value_str.lower() == 'null':    
        return None
  
    # Handle numeric values    
    try:    
        # Try integer first    
        if '.' not in value_str and 'e' not in value_str.lower():    
            return int(value_str)    
        else:    
            return float(value_str)    
    except ValueError:    
        pass
  
    # Handle lists    
    if value_str.startswith('[') and value_str.endswith(']'):    
        try:    
            list_content = value_str[1:-1].strip()    
            if not list_content:    
                return []
  
            items = split_arguments(list_content)    
            return [parse_value(item) for item in items]    
        except:    
            pass
  
    # Handle dictionaries    
    if value_str.startswith('{') and value_str.endswith('}'):    
        try:    
            return json.loads(value_str)    
        except:    
            pass
  
    # Default to string    
    return value_str

  
def evaluate_selection_accuracy(  
    model_parsed: Optional[Dict],  
    gold_parsed: Optional[Dict]  
) -> bool:  
    """  
    Evaluate if the correct tool/function was selected.
      
    Compares function names with normalization (lowercase, space->underscore)  
    to handle minor formatting variations. Handles None cases where no  
    function call is expected or provided.
  
    Args:  
        model_parsed: Parsed model function call dict (or None)  
        gold_parsed: Parsed gold function call dict (or None)
  
    Returns:  
        True if selection is correct (including both being None), False otherwise  
    """  
    # Handle case where no function call is expected  
    if not gold_parsed or not gold_parsed.get('function_name'):    
        return model_parsed is None or not model_parsed.get('function_name')
  
    # Handle case where model didn't make a call  
    if not model_parsed or not model_parsed.get('function_name'):    
        return False
  
    # Normalize function names for comparison (handle spaces, case)    
    model_name = model_parsed['function_name'].strip().lower().replace(' ', '_')    
    gold_name = gold_parsed['function_name'].strip().lower().replace(' ', '_')
        
    return model_name == gold_name

  
def evaluate_argument_accuracy(  
    model_parsed: Optional[Dict],  
    gold_parsed: Optional[Dict]  
) -> bool:  
    """  
    Evaluate if the arguments are correct for BFCL format.
      
    Checks that:  
    1. All required arguments from gold are present in model output  
    2. Argument values match (with appropriate type tolerance)  
    3. No extra arguments are present (strict matching)
  
    Args:  
        model_parsed: Parsed model function call dict (or None)  
        gold_parsed: Parsed gold function call dict (or None)
  
    Returns:  
        True if all arguments are correct, False otherwise  
    """  
    # Handle None cases  
    if not gold_parsed or not model_parsed:    
        return gold_parsed is None and model_parsed is None
  
    gold_args = gold_parsed.get('arguments', {})    
    model_args = model_parsed.get('arguments', {})
  
    # Check if all required arguments are present and correct    
    for key, gold_value in gold_args.items():    
        if key not in model_args:  
            return False
  
        model_value = model_args[key]
  
        # Compare values with type tolerance    
        if not values_equal(model_value, gold_value):    
            return False
  
    # Require exact match (no extra arguments allowed)  
    return set(model_args.keys()) == set(gold_args.keys())

  
def evaluate_argument_accuracy_mnms(  
    model_parsed: Optional[Dict],  
    gold_parsed: Optional[Dict]  
) -> bool:  
    """  
    Evaluate if the arguments are correct for MNMS format.
      
    More lenient than BFCL evaluation to handle MNMS-specific patterns:  
    - Node references like '<node-0>.image' accept any non-empty string  
    - Prompt/text arguments use fuzzy matching for semantic equivalence  
    - Other arguments use standard exact matching
      
    This accommodates MNMS's multi-step workflow where exact string matching  
    is too strict for intermediate node references.
  
    Args:  
        model_parsed: Parsed model function call dict (or None)  
        gold_parsed: Parsed gold function call dict (or None)
  
    Returns:  
        True if arguments are correct with MNMS-aware matching, False otherwise  
    """  
    if not gold_parsed or not model_parsed:    
        return gold_parsed is None and model_parsed is None
        
    gold_args = gold_parsed.get('arguments', {})    
    model_args = model_parsed.get('arguments', {})
        
    # Check if all required arguments are present    
    if set(gold_args.keys()) != set(model_args.keys()):    
        return False
        
    # Compare each argument with MNMS-aware matching    
    for key, gold_value in gold_args.items():    
        if key not in model_args:    
            return False
            
        model_value = model_args[key]
            
        # Special handling for MNMS node references    
        if isinstance(gold_value, str) and '<node-' in gold_value:    
            # Gold has node reference pattern like '<node-0>.image'    
            # Accept any non-empty string as valid reference    
            if not isinstance(model_value, str) or not model_value.strip():    
                return False    
            # Model provided some reference/value, consider it valid    
            continue
            
        # For prompt/text arguments, use fuzzy matching    
        if key in ['prompt', 'text'] and isinstance(gold_value, str) and isinstance(model_value, str):    
            if values_equal_fuzzy(model_value, gold_value):    
                continue    
            else:    
                return False
            
        # Standard comparison for other values    
        if not values_equal(model_value, gold_value):    
            return False
        
    return True

  
def values_equal_fuzzy(val1: str, val2: str, threshold: float = 0.7) -> bool:  
    """  
    Fuzzy string comparison for semantic equivalence.
      
    Uses token-based Jaccard similarity to compare strings. This allows  
    for minor variations in wording while catching significant differences.  
    Useful for evaluating prompt/text arguments where exact matching is too strict.
  
    Args:  
        val1: First string to compare  
        val2: Second string to compare  
        threshold: Minimum Jaccard similarity score to consider equal (default: 0.7)
  
    Returns:  
        True if strings are semantically similar (similarity >= threshold), False otherwise  
    """  
    # Normalize strings    
    v1 = val1.lower().strip()    
    v2 = val2.lower().strip()
        
    # Exact match    
    if v1 == v2:    
        return True
        
    # Tokenize and compare    
    tokens1 = set(re.findall(r'\w+', v1))    
    tokens2 = set(re.findall(r'\w+', v2))
        
    # Calculate Jaccard similarity: |intersection| / |union|  
    if not tokens1 and not tokens2:    
        return True    
    if not tokens1 or not tokens2:    
        return False
        
    intersection = tokens1.intersection(tokens2)    
    union = tokens1.union(tokens2)    
    similarity = len(intersection) / len(union)
        
    return similarity >= threshold

  
def values_equal(val1: Any, val2: Any, tolerance: float = 1e-6) -> bool:  
    """  
    Compare two values with appropriate tolerance for different types.
      
    Handles comparison for:  
    - None values (identity check)  
    - Numeric values (with floating-point tolerance)  
    - Strings (with whitespace normalization)  
    - Lists (element-wise recursive comparison)  
    - Dicts (key-value recursive comparison)  
    - Other types (default equality)
  
    Args:  
        val1: First value to compare  
        val2: Second value to compare  
        tolerance: Tolerance for numeric comparisons (default: 1e-6)
  
    Returns:  
        True if values are equal within tolerance, False otherwise  
    """  
    # Handle None values    
    if val1 is None or val2 is None:    
        return val1 is val2
  
    # Handle numeric comparisons with tolerance    
    try:    
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):    
            return abs(float(val1) - float(val2)) <= tolerance    
    except (ValueError, TypeError):    
        pass
  
    # Handle string comparisons (with whitespace normalization)  
    if isinstance(val1, str) and isinstance(val2, str):    
        return val1.strip() == val2.strip()
  
    # Handle list comparisons (element-wise)  
    if isinstance(val1, list) and isinstance(val2, list):    
        if len(val1) != len(val2):    
            return False    
        return all(values_equal(v1, v2, tolerance) for v1, v2 in zip(val1, val2))
  
    # Handle dict comparisons (key-value pairs)  
    if isinstance(val1, dict) and isinstance(val2, dict):    
        if set(val1.keys()) != set(val2.keys()):    
            return False    
        return all(values_equal(val1[k], val2[k], tolerance) for k in val1.keys())
  
    # Default comparison    
    return val1 == val2

  
def analyze_error_type(  
    model_parsed: Optional[Dict],  
    gold_parsed: Optional[Dict],  
    model_output: str  
) -> str:  
    """  
    Analyze and categorize the type of error in the model's response.
      
    Categorizes errors into specific types for detailed error analysis:  
    - no_response: Empty output  
    - none_response: Explicit "NONE" response  
    - parsing_errors: Failed to parse function call syntax  
    - format_errors: Parsed but with format issues  
    - unexpected_call: Made a call when none was expected  
    - wrong_tool: Incorrect function selected  
    - missing_args: Required arguments missing  
    - incorrect_args: Argument values incorrect  
    - correct: No error detected
  
    Args:  
        model_parsed: Parsed model function call dict (or None)  
        gold_parsed: Parsed gold function call dict (or None)  
        model_output: Raw model output string for context
  
    Returns:  
        String identifier for the error type  
    """  
    # Check for parsing failures  
    if not model_parsed:    
        if not model_output or model_output.strip() == "":    
            return 'no_response'    
        elif model_output.strip() == "NONE":    
            return 'none_response'    
        else:    
            return 'parsing_errors'
  
    # Check for format errors during parsing  
    if model_parsed.get('parse_error'):    
        return 'format_errors'
  
    # Check for unexpected function call  
    if not gold_parsed:    
        return 'unexpected_call'
  
    # Check selection accuracy (normalize names for comparison)  
    model_name = model_parsed.get('function_name', '').strip().lower().replace(' ', '_')    
    gold_name = gold_parsed.get('function_name', '').strip().lower().replace(' ', '_')
        
    if model_name != gold_name:    
        return 'wrong_tool'
  
    # Check argument issues    
    model_args = model_parsed.get('arguments', {})    
    gold_args = gold_parsed.get('arguments', {})
  
    # Check for missing required arguments  
    missing_args = set(gold_args.keys()) - set(model_args.keys())    
    if missing_args:    
        return 'missing_args'
  
    # Check for incorrect argument values  
    for key in gold_args:    
        if key in model_args and not values_equal(model_args[key], gold_args[key]):    
            return 'incorrect_args'
  
    return 'correct'

  
def get_empty_bfcl_metrics() -> Dict[str, Any]:  
    """  
    Return empty metrics structure for edge cases with no data.
      
    Provides a consistent metrics dictionary when the input dataframe  
    is empty or contains no valid samples.
  
    Returns:  
        Dictionary with same structure as compute_bfcl_accuracy() but with  
        all counts set to 0, all rates set to 0.0, and empty collections  
    """  
    return {    
        'selection_accuracy': 0.0,    
        'argument_accuracy': 0.0,    
        'integration_accuracy': 0.0,    
        'selection_correct': 0,    
        'argument_correct': 0,    
        'integration_correct': 0,    
        'total_items': 0,    
        'tool_metrics': {},    
        'tool_count': 0,    
        'error_analysis': {    
            'parsing_errors': 0,    
            'wrong_tool': 0,    
            'missing_args': 0,    
            'incorrect_args': 0,    
            'format_errors': 0    
        },    
        'error_rate': 0.0,    
        'perfect_calls': 0,    
        'partial_success_rate': 0.0,    
        'detailed_results': []    
    }

  
def compute_mnms_accuracy(  
    results_df: pd.DataFrame,  
    report_detailed: bool = False  
) -> Dict[str, Any]:  
    """  
    Compute accuracy for MNMS (Multi-Node Multi-Step) dataset.
      
    Convenience wrapper around compute_bfcl_accuracy() that automatically  
    sets the dataset format to 'mnms' for proper gold format parsing and  
    MNMS-aware argument matching.
  
    Args:  
        results_df: DataFrame with MNMS results  
        report_detailed: Whether to include detailed per-sample results
  
    Returns:  
        Dictionary with tool calling metrics (same structure as compute_bfcl_accuracy)  
    """  
    return compute_bfcl_accuracy(  
        results_df,  
        dataset_format='mnms',  
        report_detailed=report_detailed  
    )  