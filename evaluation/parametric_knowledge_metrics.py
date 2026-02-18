"""  
Parametric knowledge evaluation metrics.
  
This module evaluates AI models' ability to answer questions using their parametric  
(internal) knowledge. Includes metrics for open-ended QA, multiple choice, mathematical  
reasoning (GSM8K, LiveMathBench), and code generation (HumanEval, MBPP).  
"""
  
import re  
import pandas as pd  
from typing import Optional, Tuple, Dict, Any, List  
from pathlib import Path  
import sys  
import io  
import traceback
  
from .llm_judge import llm_judge_eval

  
def ensure_dir(p):  
    """  
    Create directory if it doesn't exist, including parent directories.
      
    Args:  
        p: Path to directory (string or Path object)  
    """  
    Path(p).mkdir(parents=True, exist_ok=True)

  
def compute_open_accuracy(results_df: pd.DataFrame, debug: bool = False) -> Dict[str, Any]:  
    """  
    Compute accuracy for open-ended question answering tasks.
      
    Evaluates whether model responses contain the correct answer(s) using substring  
    matching. Handles multiple gold answer formats (dict, list, string) and checks  
    if any gold answer appears in the model's response.
      
    Args:  
        results_df: DataFrame with columns:  
            - 'outputs': Model responses (strings)  
            - 'gold': Correct answers in various formats (dict with 'text' or 'answer' key,  
                     list of answers, or single string)  
        debug: If True, print detailed matching information for each sample
          
    Returns:  
        Dictionary containing:  
        - accuracy: Fraction of correct answers (0.0 to 1.0)  
        - format_score: Fraction of non-empty responses (0.0 to 1.0)  
        - correct: Number of correct responses  
        - total: Total number of samples  
        - valid_responses: Number of non-empty responses  
    """  
    response = results_df['outputs']    
    gold = results_df['gold']
        
    correct = 0    
    total = 0    
    valid_responses = 0
        
    for idx, (resp, gold_answers) in enumerate(zip(response, gold)):    
        total += 1
            
        # Handle None or empty responses    
        if resp is None or (isinstance(resp, str) and resp.strip() == ''):    
            if debug:    
                print(f"Row {idx}: Empty/None response")    
            continue
            
        # Convert response to string and normalize    
        resp_str = str(resp).lower().strip()    
        valid_responses += 1
            
        # Extract gold answers from different formats    
        gold_normalized = []
            
        if isinstance(gold_answers, dict):    
            # Handle dictionary format with 'text' key  
            if 'text' in gold_answers:    
                texts = gold_answers['text']    
                if isinstance(texts, list):    
                    gold_normalized = [str(ans).lower().strip() for ans in texts]    
                else:    
                    gold_normalized = [str(texts).lower().strip()]    
            # Handle dictionary format with 'answer' key  
            elif 'answer' in gold_answers:    
                answers = gold_answers['answer']    
                if isinstance(answers, list):    
                    gold_normalized = [str(ans).lower().strip() for ans in answers]    
                else:    
                    gold_normalized = [str(answers).lower().strip()]    
            else:    
                if debug:    
                    print(f"Row {idx}: Unknown dict format: {gold_answers}")    
                continue
                    
        elif isinstance(gold_answers, list):    
            # Handle list format  
            gold_normalized = [str(ans).lower().strip() for ans in gold_answers]
                
        elif isinstance(gold_answers, str):    
            # Try to parse string as list if it looks like one  
            try:  
                gold_answers = eval(gold_answers)  
            except Exception as e:  
                print(e)  
                pass
  
            if isinstance(gold_answers, list):    
                gold_normalized = [str(ans).lower().strip() for ans in gold_answers]  
            else:  
                # Handle single string    
                gold_normalized = [gold_answers.lower().strip()]
                
        else:    
            if debug:    
                print(f"Row {idx}: Unknown gold format: {type(gold_answers)}")    
            continue
              
        if debug:  
            print(f"Gold: {gold_normalized}")  
            print(f"Resp: {resp_str}")
              
        # Check if any gold answer appears in the response (substring match)  
        is_correct = False    
        matched_answer = None
            
        for gold_ans in gold_normalized:    
            if gold_ans in resp_str:    
                is_correct = True    
                matched_answer = gold_ans    
                break
            
        if is_correct:    
            correct += 1    
            if debug:    
                print(f"Row {idx}: CORRECT - Found '{matched_answer}' in '{resp_str[:100]}'")    
        else:    
            if debug:    
                print(f"Row {idx}: INCORRECT - Response: '{resp_str[:100]}', Expected one of: {gold_normalized}")
        
    # Calculate metrics    
    accuracy = correct / total if total > 0 else 0.0    
    response_format_score = valid_responses / total if total > 0 else 0.0
        
    return {    
        'accuracy': accuracy,     
        'format_score': response_format_score,    
        'correct': correct,    
        'total': total,    
        'valid_responses': valid_responses    
    }

  
def compute_mc_accuracy(  
    results_df: pd.DataFrame,  
    judge: str = "gpt-4o-mini@openai",  
    debug: bool = False  
) -> Dict[str, Any]:  
    """  
    Compute accuracy for multiple choice questions.
      
    Uses exact match for properly formatted responses (single letter A-D),  
    and falls back to LLM judge for malformed responses.
      
    Args:  
        results_df: DataFrame with columns:  
            - 'pp-outputs': Preprocessed model responses  
            - 'gold': Correct answer letters  
            - 'choices': Multiple choice options  
        judge: Model ID for LLM judge fallback evaluation  
        debug: If True, print detailed judge evaluation information
          
    Returns:  
        Dictionary containing:  
        - accuracy: Overall accuracy score (0.0 to 1.0)  
        - format_score: Fraction of properly formatted responses (0.0 to 1.0)  
    """  
    response = results_df['pp-outputs']  
    choices = results_df['choices']  
    gold = results_df['gold']  
    gold_unique = set(gold)
  
    # Separate properly formatted from malformed responses  
    format_ok = [i for i in results_df.index if check_format(results_df.loc[i]['pp-outputs'])]  
    to_check = [i for i in results_df.index if not check_format(results_df.loc[i]['pp-outputs'])]
  
    # Exact match for properly formatted responses  
    em_correct = sum([response[i].upper() == gold[i].upper() for i in format_ok])
  
    # LLM judge evaluation for malformed responses  
    if len(to_check) > 0:  
        results_to_check = results_df.loc[to_check].reset_index()  
        judge_response = llm_judge_eval(results_to_check, "single_turn_mc", judge)
          
        if debug:  
            print('Debug LLM judge evaluation:')  
            for j in range(len(results_to_check)):  
                print(f"Output: {results_to_check['pp-outputs'][j]}")  
                print(f"Gold: {results_to_check['gold'][j]}")  
                print(f"Choices: {results_to_check['choices'][j]}")  
                print(f"Judge result: {judge_response[j]}")  
                print()
                  
        judge_correct = sum([j == 'True' for j in judge_response])  
    else:  
        judge_correct = 0
  
    # Calculate overall metrics  
    if len(gold) > 0:  
        accuracy = (em_correct + judge_correct) / len(gold)  
        response_format_score = len(format_ok) / len(gold)  
    else:  
        accuracy = 0.0  
        response_format_score = 0.0
  
    return {'accuracy': accuracy, 'format_score': response_format_score}

  
def check_format(output: str) -> bool:  
    """  
    Check if output is a valid single-letter multiple choice answer.
      
    Args:  
        output: Model response string
          
    Returns:  
        True if output is a single character string (valid MC format), False otherwise  
    """  
    return isinstance(output, str) and len(output) == 1

  
def extract_answer_from_output(output: str) -> Optional[float]:  
    """  
    Extract the final numerical answer from model output.
      
    Tries multiple extraction methods in order of reliability:  
    1. "Answer: <number>" pattern  
    2. "#### <number>" pattern (GSM8K format)  
    3. Final number at end of response  
    4. Any standalone number in last 3 lines
      
    Args:  
        output: Model response string containing mathematical reasoning
          
    Returns:  
        Extracted numerical answer as float, or None if no answer found  
    """  
    if not isinstance(output, str):  
        return None
  
    # Method 1: Look for "Answer: <number>" pattern (most reliable)  
    answer_pattern = r'Answer:\s*([+-]?\d+(?:\.\d+)?)'  
    match = re.search(answer_pattern, output, re.IGNORECASE)  
    if match:  
        try:  
            return float(match.group(1))  
        except ValueError:  
            pass
  
    # Method 2: Look for "#### <number>" pattern (GSM8K format)  
    gsm_pattern = r'####\s*([+-]?\d+(?:\.\d+)?)'  
    match = re.search(gsm_pattern, output)  
    if match:  
        try:  
            return float(match.group(1))  
        except ValueError:  
            pass
  
    # Method 3: Look for final number at the end of the response  
    lines = output.strip().split('\n')  
    for line in reversed(lines):  
        line = line.strip()  
        if line:  
            # Try to extract number from the line  
            number_pattern = r'([+-]?\d+(?:\.\d+)?)'  
            numbers = re.findall(number_pattern, line)  
            if numbers:  
                try:  
                    return float(numbers[-1])  # Take the last number in the line  
                except ValueError:  
                    continue
  
    # Method 4: Look for any standalone number in the last few lines  
    last_part = '\n'.join(output.strip().split('\n')[-3:])  # Last 3 lines  
    number_pattern = r'\b([+-]?\d+(?:\.\d+)?)\b'  
    numbers = re.findall(number_pattern, last_part)  
    if numbers:  
        try:  
            return float(numbers[-1])  # Take the last number found  
        except ValueError:  
            pass
  
    return None

  
def extract_answer_from_gold(gold: str) -> Optional[float]:  
    """  
    Extract the numerical answer from gold standard response.
      
    Gold format typically ends with "#### <number>" in GSM8K format.  
    Falls back to extracting the last number if pattern not found.
      
    Args:  
        gold: Gold standard answer string
          
    Returns:  
        Extracted numerical answer as float, or None if no answer found  
    """  
    if not isinstance(gold, str):  
        return None
  
    # Look for "#### <number>" pattern at the end  
    gsm_pattern = r'####\s*([+-]?\d+(?:\.\d+)?)'  
    match = re.search(gsm_pattern, gold)  
    if match:  
        try:  
            return float(match.group(1))  
        except ValueError:  
            pass
  
    # Fallback: look for the last number in the gold response  
    number_pattern = r'([+-]?\d+(?:\.\d+)?)'  
    numbers = re.findall(number_pattern, gold)  
    if numbers:  
        try:  
            return float(numbers[-1])  
        except ValueError:  
            pass
  
    return None

  
def compute_gsm8k_accuracy(  
    results_df: pd.DataFrame,  
    tolerance: float = 1e-6,  
    report_detailed: bool = False  
) -> Dict[str, Any]:  
    """  
    Compute GSM8K mathematical reasoning accuracy with detailed failure analysis.
      
    Extracts numerical answers from both model outputs and gold standards,  
    then compares them with floating-point tolerance. Categorizes failures  
    into extraction failures vs. incorrect answers.
      
    Args:  
        results_df: DataFrame with columns:  
            - 'outputs': Model responses with reasoning and final answer  
            - 'gold': Gold standard answers in GSM8K format  
        tolerance: Numerical tolerance for floating-point comparison (default: 1e-6)  
        report_detailed: If True, include per-sample results in output
          
    Returns:  
        Dictionary containing:  
        - accuracy: Overall accuracy score (0.0 to 1.0)  
        - correct_count: Number of correct answers  
        - total_count: Total number of samples  
        - failure_reasons: Dict with counts for:  
            - no_predicted_answer: Failed to extract answer from output  
            - no_gold_answer: Failed to extract answer from gold  
            - wrong_answer: Extracted answer but incorrect  
        - detailed_results: Per-sample results (if report_detailed=True)  
    """  
    outputs = results_df['outputs']  
    golds = results_df['gold']  
    golds = [str(g) for g in golds]
  
    correct_count = 0  
    total_count = len(outputs)  
    failure_reasons = {  
        'no_predicted_answer': 0,  
        'no_gold_answer': 0,  
        'wrong_answer': 0  
    }
  
    results = []
  
    for i, (output, gold) in enumerate(zip(outputs, golds)):  
        predicted_answer = extract_answer_from_output(output)  
        gold_answer = extract_answer_from_gold(gold)
  
        result = {  
            'index': i,  
            'predicted': predicted_answer,  
            'gold': gold_answer,  
            'correct': False  
        }
  
        if predicted_answer is None:  
            failure_reasons['no_predicted_answer'] += 1  
        elif gold_answer is None:  
            failure_reasons['no_gold_answer'] += 1  
        elif numbers_approximately_equal(predicted_answer, gold_answer, tolerance):  
            correct_count += 1  
            result['correct'] = True  
        else:  
            failure_reasons['wrong_answer'] += 1
  
        results.append(result)
  
    accuracy = correct_count / total_count if total_count > 0 else 0.0
  
    return {  
        'accuracy': accuracy,  
        'correct_count': correct_count,  
        'total_count': total_count,  
        'failure_reasons': failure_reasons,  
        'detailed_results': results if report_detailed else None  
    }

  
def numbers_approximately_equal(a: float, b: float, tolerance: float = 1e-6) -> bool:  
    """  
    Check if two numbers are approximately equal within tolerance.
      
    Uses absolute difference comparison to handle floating-point precision issues.
      
    Args:  
        a: First number  
        b: Second number  
        tolerance: Maximum allowed absolute difference (default: 1e-6)
          
    Returns:  
        True if |a - b| <= tolerance, False otherwise  
    """  
    return abs(a - b) <= tolerance

  
def compute_humaneval_accuracy(results_df: pd.DataFrame) -> Dict[str, Any]:  
    """  
    Compute HumanEval code generation accuracy.
      
    Executes generated code against test cases to verify correctness.  
    Handles code extraction from markdown formatting and provides detailed  
    failure diagnostics including syntax errors and test failures.
      
    Args:  
        results_df: DataFrame with columns:  
            - 'outputs': Generated code (may include markdown formatting)  
            - 'ctx': Context containing [test_code, entry_point] or  
            - 'gold': Dict with 'test' and 'entry_point' keys
              
    Returns:  
        Dictionary containing:  
        - accuracy: Fraction of correct solutions (0.0 to 1.0)  
        - correct_count: Number of passing solutions  
        - total_count: Total number of problems  
        - failure_reasons: List of dicts with failure details including:  
            - id: Problem identifier  
            - reason: Error message (syntax error, test failure, etc.)  
            - code: Generated code that failed  
            - entry_point: Function name being tested  
    """  
    correct_count = 0  
    total_count = len(results_df)  
    failure_reasons = []
  
    for idx, row in results_df.iterrows():  
        try:  
            # Extract the structured fields  
            output = row['outputs']  
            ctx = row['ctx']
              
            # Handle different context formats  
            if isinstance(ctx, list):  
                test_code = ctx[0]  
                entry_point = ctx[1]  
            else:  
                test_code = row['gold']["test"]  
                entry_point = row['gold']["entry_point"]
  
            # Execute the model's generated code with timeout protection  
            is_correct, error_msg = execute_with_timeout(output, test_code, entry_point)
  
            if is_correct:  
                correct_count += 1  
            else:  
                failure_reasons.append({  
                    'id': row.get('id', idx),  
                    'reason': error_msg,  
                    'code': output,  
                    'entry_point': entry_point  
                })
  
        except Exception as e:  
            failure_reasons.append({  
                'id': row.get('id', idx),  
                'reason': f"Processing error: {str(e)}",  
                'code': row.get('output', ''),  
                'entry_point': row.get('entry_point', '')  
            })
  
    accuracy = correct_count / total_count if total_count > 0 else 0.0
  
    return {  
        'accuracy': accuracy,  
        'correct_count': correct_count,  
        'total_count': total_count,  
        'failure_reasons': failure_reasons  
    }

  
def execute_humaneval_test(  
    generated_code: str,  
    test_code: str,  
    entry_point: str  
) -> Tuple[bool, str]:  
    """  
    Execute HumanEval test using the provided test code and entry point.
      
    Preprocesses generated code to remove markdown formatting, executes it  
    in an isolated namespace, then runs the test suite.
      
    Args:  
        generated_code: Model-generated code (may include markdown fences)  
        test_code: Test cases to validate the generated code  
        entry_point: Name of the function to test
          
    Returns:  
        Tuple of (is_correct: bool, message: str) where message explains  
        the result (success or specific failure reason)  
    """  
    try:  
        # Preprocess the generated code to remove markdown formatting  
        cleaned_code = preprocess_code_output(generated_code)
  
        # Create a single namespace for both globals and locals to avoid scoping issues  
        namespace = {}
  
        # Execute the generated code  
        exec(cleaned_code, namespace, namespace)
  
        # Check if the entry point function exists  
        if entry_point and entry_point not in namespace:  
            return False, f"Entry point function '{entry_point}' not found in generated code"
  
        # Execute the test code in the same namespace  
        if isinstance(test_code, list):  
            test_code = '\n'.join(test_code)  
        exec(test_code, namespace, namespace)
  
        # Call the check function with the candidate  
        if entry_point:  
            if 'check' in namespace:  
                namespace['check'](namespace[entry_point])  
            else:  
                return False, "No 'check' function found in test code"
  
        # If we get here without exceptions, all tests passed  
        return True, "All tests passed"
  
    except AssertionError as e:  
        return False, f"Test assertion failed: {str(e)}"  
    except SyntaxError as e:  
        return False, f"Syntax error in generated code: {str(e)}"  
    except Exception as e:  
        return False, f"Runtime error: {str(e)}"

  
def preprocess_code_output(code_string: str) -> str:  
    """  
    Clean model output by removing markdown code block formatting.
      
    Handles various markdown formats including:  
    - ```python ... ```  
    - ``` ... ```  
    - Triple quote strings
      
    Args:  
        code_string: Raw model output that may contain markdown formatting
          
    Returns:  
        Cleaned code string ready for execution  
    """  
    if not isinstance(code_string, str):  
        return str(code_string) if code_string is not None else ""
  
    # Remove markdown code block delimiters  
    cleaned = code_string.strip()
  
    # Remove opening code block markers  
    if cleaned.startswith('```python'):  
        cleaned = cleaned[9:]  # Remove '```python'  
    elif cleaned.startswith('```'):  
        cleaned = cleaned[3:]   # Remove generic '```'
  
    # Remove closing code block markers  
    if cleaned.endswith('```'):  
        cleaned = cleaned[:-3]  # Remove closing '```'
  
    # Strip any remaining whitespace  
    cleaned = cleaned.strip()
  
    # Handle cases where the code is wrapped in triple quotes  
    if cleaned.startswith('"""') and cleaned.endswith('"""'):  
        cleaned = cleaned[3:-3].strip()  
    elif cleaned.startswith("'''") and cleaned.endswith("'''"):  
        cleaned = cleaned[3:-3].strip()
  
    # Remove any leading/trailing whitespace again  
    cleaned = cleaned.strip()
  
    return cleaned

  
def execute_with_timeout(  
    generated_code: str,  
    test_code: str,  
    entry_point: str,  
    timeout: int = 10  
) -> Tuple[bool, str]:  
    """  
    Execute code with timeout protection to prevent infinite loops.
      
    Uses SIGALRM signal to interrupt execution after timeout period.
      
    Args:  
        generated_code: Model-generated code to execute  
        test_code: Test cases to validate the code  
        entry_point: Function name to test  
        timeout: Maximum execution time in seconds (default: 10)
          
    Returns:  
        Tuple of (is_correct: bool, message: str) indicating test result  
    """  
    import signal
  
    def timeout_handler(signum, frame):  
        raise TimeoutError("Code execution timed out")
  
    try:  
        signal.signal(signal.SIGALRM, timeout_handler)  
        signal.alarm(timeout)
  
        result = execute_humaneval_test(generated_code, test_code, entry_point)
  
        signal.alarm(0)  
        return result
  
    except TimeoutError:  
        return False, "Code execution timed out"  
    except Exception as e:  
        return False, f"Execution error: {str(e)}"  
    finally:  
        signal.alarm(0)

  
def compute_livemathbench_accuracy(  
    results_df: pd.DataFrame,  
    judge: str = "gpt-4o-mini@openai",  
    debug: bool = False  
) -> Dict[str, Any]:  
    """  
    Compute LiveMathBench accuracy using rule-based evaluation with LLM judge fallback.
      
    Attempts rule-based matching first (exact match, numeric comparison, normalized  
    expressions) and falls back to LLM judge for ambiguous cases. This hybrid approach  
    balances precision with coverage.
      
    Args:  
        results_df: DataFrame with columns:  
            - 'outputs': Model responses containing mathematical answers  
            - 'gold': Correct answers (dict with 'text' key, list, or string)  
        judge: Model ID for LLM judge fallback evaluation  
        debug: If True, print detailed evaluation information
          
    Returns:  
        Dictionary containing:  
        - accuracy: Overall accuracy score (0.0 to 1.0)  
        - correct_count: Total number of correct answers  
        - total_count: Total number of samples  
        - rule_based_accuracy: Accuracy for rule-based matches  
        - rule_based_correct: Count of rule-based correct answers  
        - rule_based_total: Count of samples evaluated with rules  
        - llm_judge_accuracy: Accuracy for LLM judge evaluations  
        - llm_judge_correct: Count of LLM judge correct answers  
        - llm_judge_total: Count of samples evaluated by LLM judge  
        - detailed_results: Per-sample results with method used  
    """  
    outputs = results_df['outputs']    
    golds = results_df['gold']
        
    correct_count = 0    
    total_count = len(outputs)    
    rule_based_correct = 0    
    rule_based_total = 0    
    llm_judge_correct = 0    
    llm_judge_total = 0
        
    # Track which rows need LLM judge evaluation    
    needs_llm_judge = []    
    llm_judge_indices = []
        
    results = []
        
    for idx, (output, gold) in enumerate(zip(outputs, golds)):    
        # Extract answer from model output    
        predicted_answer = extract_livemathbench_answer(output)
            
        # Normalize gold answer from various formats  
        if isinstance(gold, dict) and 'text' in gold:    
            gold_answer = gold['text'][0] if isinstance(gold['text'], list) else gold['text']    
        elif isinstance(gold, list):    
            gold_answer = gold[0] if len(gold) > 0 else str(gold)    
        else:    
            gold_answer = str(gold)
            
        if predicted_answer is None:    
            if debug:    
                print(f"Row {idx}: Could not extract answer from output")    
            results.append({    
                'index': idx,    
                'predicted': None,    
                'gold': gold_answer,    
                'correct': False,    
                'method': 'extraction_failed'    
            })    
            continue
            
        # Try rule-based evaluation    
        is_correct, method = evaluate_livemathbench_rule_based(    
            predicted_answer, gold_answer, debug=debug, row_idx=idx    
        )
            
        if method != 'needs_llm_judge':    
            # Rule-based evaluation succeeded    
            rule_based_total += 1    
            if is_correct:    
                correct_count += 1    
                rule_based_correct += 1
                
            results.append({    
                'index': idx,    
                'predicted': predicted_answer,    
                'gold': gold_answer,    
                'correct': is_correct,    
                'method': method    
            })    
        else:    
            # Need LLM judge evaluation    
            needs_llm_judge.append({    
                'predicted': predicted_answer,    
                'gold': gold_answer,    
                'full_output': output    
            })    
            llm_judge_indices.append(idx)    
            results.append({    
                'index': idx,    
                'predicted': predicted_answer,    
                'gold': gold_answer,    
                'correct': False,  # Will be updated after LLM judge    
                'method': 'pending_llm_judge'    
            })
        
    # Run LLM judge evaluation for remaining cases    
    if len(needs_llm_judge) > 0:    
        if debug:    
            print(f"\nRunning LLM judge for {len(needs_llm_judge)} cases...")
            
        # Create dataframe for LLM judge    
        judge_df = pd.DataFrame({    
            'model_answer': [item['predicted'] for item in needs_llm_judge],    
            'gold_answer': [item['gold'] for item in needs_llm_judge]    
        })
            
        # Run LLM judge evaluation using coverage scoring  
        judge_results = llm_judge_eval(    
            judge_df,    
            mode="ragtruth_accuracy",  # Uses coverage score    
            model_id=judge,    
            max_tokens=512,    
            temperature=0.0,    
            verbose=debug    
        )
            
        # Update results with LLM judge scores (threshold: 0.8 for correctness)  
        llm_judge_total = len(judge_results)    
        for i, (judge_idx, score) in enumerate(zip(llm_judge_indices, judge_results)):    
            if score is not None and score >= 0.8:    
                is_correct = True    
                llm_judge_correct += 1    
                correct_count += 1    
            else:    
                is_correct = False
                
            results[judge_idx]['correct'] = is_correct    
            results[judge_idx]['method'] = 'llm_judge'    
            results[judge_idx]['llm_score'] = score
                
            if debug:    
                print(f"Row {judge_idx}: LLM Judge score={score}, correct={is_correct}")
        
    # Calculate metrics    
    accuracy = correct_count / total_count if total_count > 0 else 0.0    
    rule_based_accuracy = rule_based_correct / rule_based_total if rule_based_total > 0 else 0.0    
    llm_judge_accuracy = llm_judge_correct / llm_judge_total if llm_judge_total > 0 else 0.0
        
    return {    
        'accuracy': accuracy,    
        'correct_count': correct_count,    
        'total_count': total_count,    
        'rule_based_accuracy': rule_based_accuracy,    
        'rule_based_correct': rule_based_correct,    
        'rule_based_total': rule_based_total,    
        'llm_judge_accuracy': llm_judge_accuracy,    
        'llm_judge_correct': llm_judge_correct,    
        'llm_judge_total': llm_judge_total,    
        'detailed_results': results    
    }

    
def extract_livemathbench_answer(output: str) -> Optional[str]:  
    """  
    Extract the final answer from LiveMathBench model output.
      
    Tries multiple extraction methods:  
    1. "Answer: <expression>" pattern  
    2. LaTeX boxed answer: \boxed{...}  
    3. Last non-empty line after step-by-step reasoning
      
    Args:  
        output: Model response string containing mathematical reasoning
          
    Returns:  
        Extracted answer string, or None if no answer found  
    """  
    if not isinstance(output, str):    
        return None
        
    # Method 1: Look for "Answer: " pattern (most reliable)    
    answer_pattern = r'Answer:\s*(.+?)(?:\n|$)'    
    match = re.search(answer_pattern, output, re.IGNORECASE)    
    if match:    
        answer = match.group(1).strip()    
        # Remove trailing punctuation    
        answer = answer.rstrip('.,;')    
        return answer
        
    # Method 2: Look for boxed answer (LaTeX format)    
    boxed_pattern = r'\\boxed\{([^}]+)\}'    
    match = re.search(boxed_pattern, output)    
    if match:    
        return match.group(1).strip()
        
    # Method 3: Look for last line after "Step N:" reasoning  
    lines = output.strip().split('\n')    
    for line in reversed(lines):    
        line = line.strip()    
        # Skip empty lines and step labels    
        if not line or line.startswith('Step'):    
            continue    
        # If we find a non-step line, consider it the answer    
        if line:    
            return line
        
    return None

    
def evaluate_livemathbench_rule_based(  
    predicted: str,  
    gold: str,  
    debug: bool = False,  
    row_idx: Optional[int] = None  
) -> Tuple[bool, str]:  
    """  
    Rule-based evaluation for LiveMathBench answers.
      
    Attempts multiple matching strategies in order:  
    1. Exact string match (case-insensitive)  
    2. Numeric comparison (for numerical answers)  
    3. Normalized mathematical expression match  
    4. Substring/word match (for text answers)  
    5. Mathematical equivalence (fractions, percentages, etc.)
      
    Args:  
        predicted: Extracted model answer  
        gold: Gold standard answer  
        debug: If True, print matching details  
        row_idx: Row index for debug messages
          
    Returns:  
        Tuple of (is_correct: bool, method: str) where method indicates  
        which matching strategy succeeded or 'needs_llm_judge' if all failed  
    """  
    if not predicted or not gold:    
        return False, 'empty_answer'
        
    pred_normalized = predicted.strip().lower()    
    gold_normalized = gold.strip().lower()
        
    # Rule 1: Exact string match    
    if pred_normalized == gold_normalized:    
        if debug:    
            print(f"Row {row_idx}: Exact match")    
        return True, 'exact_match'
        
    # Rule 2: Numeric comparison    
    pred_num = extract_number_from_string(predicted)    
    gold_num = extract_number_from_string(gold)
        
    if pred_num is not None and gold_num is not None:    
        if numbers_approximately_equal(pred_num, gold_num, tolerance=1e-6):    
            if debug:    
                print(f"Row {row_idx}: Numeric match ({pred_num} ≈ {gold_num})")    
            return True, 'numeric_match'    
        else:    
            if debug:    
                print(f"Row {row_idx}: Numeric mismatch ({pred_num} ≠ {gold_num})")    
            return False, 'numeric_mismatch'
        
    # Rule 3: Normalized mathematical expression comparison    
    pred_math_normalized = normalize_math_expression(predicted)    
    gold_math_normalized = normalize_math_expression(gold)
        
    if pred_math_normalized == gold_math_normalized:    
        if debug:    
            print(f"Row {row_idx}: Normalized math match")    
        return True, 'normalized_match'
        
    # Rule 4: Check if answer is contained (for text answers)    
    # Only if both are non-numeric text    
    if pred_num is None and gold_num is None:    
        # Check if gold answer is contained in prediction (case insensitive)    
        if gold_normalized in pred_normalized:    
            if debug:    
                print(f"Row {row_idx}: Substring match")    
            return True, 'substring_match'
            
        # Check word-level match for name-type answers    
        pred_words = set(pred_normalized.split())    
        gold_words = set(gold_normalized.split())
            
        if gold_words and gold_words.issubset(pred_words):    
            if debug:    
                print(f"Row {row_idx}: Word subset match")    
            return True, 'word_match'
        
    # Rule 5: Check for common mathematical equivalences    
    if check_mathematical_equivalence(predicted, gold):    
        if debug:    
            print(f"Row {row_idx}: Mathematical equivalence")    
        return True, 'math_equivalence'
        
    # If no rule-based method works, fall back to LLM judge    
    if debug:    
        print(f"Row {row_idx}: Needs LLM judge evaluation")    
    return False, 'needs_llm_judge'

    
def extract_number_from_string(text: str) -> Optional[float]:  
    """  
    Extract a numeric value from a string, handling various formats.
      
    Handles:  
    - Direct numbers: "42", "3.14"  
    - Numbers with LaTeX: "$42$", "\pi"  
    - Fractions: "3/4"  
    - Scientific notation: "1.5e-3"
      
    Args:  
        text: String that may contain a number
          
    Returns:  
        Extracted number as float, or None if no clear numeric value found  
    """  
    if not isinstance(text, str):    
        return None
        
    # Remove common LaTeX commands    
    text = text.replace('\\', '')    
    text = text.replace('$', '')
        
    # Try to parse as direct number    
    try:    
        return float(text.strip())    
    except ValueError:    
        pass
        
    # Look for number patterns (integers, decimals, scientific notation)  
    number_pattern = r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'    
    matches = re.findall(number_pattern, text)
        
    if matches:    
        try:    
            # Return the last number found (usually the final answer)    
            return float(matches[-1])    
        except ValueError:    
            pass
        
    # Try to evaluate simple fractions    
    fraction_pattern = r'([+-]?\d+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)'    
    match = re.search(fraction_pattern, text)    
    if match:    
        try:    
            numerator = float(match.group(1))    
            denominator = float(match.group(2))    
            if denominator != 0:    
                return numerator / denominator    
        except (ValueError, ZeroDivisionError):    
            pass
        
    return None

    
def normalize_math_expression(expr: str) -> str:  
    """  
    Normalize mathematical expressions for comparison.
      
    Removes whitespace, standardizes LaTeX commands, and normalizes  
    bracket types to enable string-based comparison of equivalent expressions.
      
    Args:  
        expr: Mathematical expression string (may contain LaTeX)
          
    Returns:  
        Normalized expression string  
    """  
    if not isinstance(expr, str):    
        return str(expr)
        
    # Convert to lowercase    
    normalized = expr.lower()
        
    # Remove all whitespace    
    normalized = re.sub(r'\s+', '', normalized)
        
    # Standardize LaTeX spacing commands  
    latex_replacements = {    
        '\\left': '',    
        '\\right': '',    
        '\\,': '',    
        '\\:': '',    
        '\\;': '',    
        '\\!': '',    
        '\\quad': '',    
        '\\qquad': '',    
    }
        
    for old, new in latex_replacements.items():    
        normalized = normalized.replace(old, new)
        
    # Remove dollar signs    
    normalized = normalized.replace('$', '')
        
    # Standardize brackets to parentheses  
    normalized = normalized.replace('[', '(').replace(']', ')')    
    normalized = normalized.replace('{', '(').replace('}', ')')
        
    return normalized

    
def check_mathematical_equivalence(pred: str, gold: str) -> bool:  
    """  
    Check for common mathematical equivalences hard to detect with string matching.
      
    Handles:  
    - Fraction to decimal conversion (e.g., "1/2" vs "0.5")  
    - Percentage to decimal (e.g., "50%" vs "0.5")  
    - Simple algebraic evaluation (e.g., "2*3" vs "6")
      
    Args:  
        pred: Predicted answer string  
        gold: Gold answer string
          
    Returns:  
        True if answers are mathematically equivalent, False otherwise  
    """  
    # Check for fraction equivalences (e.g., "1/2" vs "0.5")  
    pred_as_decimal = try_fraction_to_decimal(pred)    
    gold_as_decimal = try_fraction_to_decimal(gold)
        
    if pred_as_decimal is not None and gold_as_decimal is not None:    
        if numbers_approximately_equal(pred_as_decimal, gold_as_decimal):    
            return True
        
    # Check for percentage equivalences (e.g., "50%" vs "0.5")  
    pred_from_percent = try_percent_to_decimal(pred)    
    gold_from_percent = try_percent_to_decimal(gold)
        
    if pred_from_percent is not None and gold_from_percent is not None:    
        if numbers_approximately_equal(pred_from_percent, gold_from_percent):    
            return True
        
    # Check for simple algebraic equivalences (e.g., "2*3" vs "6")  
    pred_evaluated = try_safe_eval(pred)    
    gold_evaluated = try_safe_eval(gold)
        
    if pred_evaluated is not None and gold_evaluated is not None:    
        if numbers_approximately_equal(pred_evaluated, gold_evaluated):    
            return True
        
    return False

    
def try_fraction_to_decimal(text: str) -> Optional[float]:  
    """  
    Try to convert fraction notation to decimal.
      
    Args:  
        text: String that may represent a fraction (e.g., "3/4")
          
    Returns:  
        Decimal value of fraction, or None if not a valid fraction  
    """  
    if not isinstance(text, str):    
        return None
        
    fraction_pattern = r'^([+-]?\d+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)$'    
    match = re.match(fraction_pattern, text.strip())
        
    if match:    
        try:    
            numerator = float(match.group(1))    
            denominator = float(match.group(2))    
            if denominator != 0:    
                return numerator / denominator    
        except (ValueError, ZeroDivisionError):    
            pass
        
    return None

    
def try_percent_to_decimal(text: str) -> Optional[float]:  
    """  
    Try to convert percentage to decimal.
      
    Args:  
        text: String that may represent a percentage (e.g., "50%")
          
    Returns:  
        Decimal value (e.g., 0.5 for "50%"), or None if not a valid percentage  
    """  
    if not isinstance(text, str):    
        return None
        
    percent_pattern = r'^([+-]?\d+(?:\.\d+)?)\s*%$'    
    match = re.match(percent_pattern, text.strip())
        
    if match:    
        try:    
            return float(match.group(1)) / 100.0    
        except ValueError:    
            pass
        
    return None

    
def try_safe_eval(text: str) -> Optional[float]:  
    """  
    Safely evaluate simple mathematical expressions.
      
    Only allows basic arithmetic operations (+, -, *, /, parentheses).  
    Prevents code injection by restricting allowed characters and using  
    empty builtins namespace.
      
    Args:  
        text: String containing mathematical expression (e.g., "2*3+4")
          
    Returns:  
        Evaluated result as float, or None if evaluation fails or is unsafe  
    """  
    if not isinstance(text, str):    
        return None
        
    # Only allow safe characters (numbers and basic operators)  
    if not re.match(r'^[0-9+\-*/().\s]+$', text):    
        return None
        
    # Remove whitespace    
    text = text.replace(' ', '')
        
    # Check for dangerous patterns    
    if '__' in text or 'import' in text or 'eval' in text:    
        return None
        
    try:    
        # Use eval with restricted namespace (no builtins)  
        result = eval(text, {"__builtins__": {}}, {})    
        if isinstance(result, (int, float)):    
            return float(result)    
    except:    
        pass
        
    return None  