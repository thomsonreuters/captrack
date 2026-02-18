"""  
Advanced mathematical reasoning evaluation metrics.
  
This module provides comprehensive evaluation for mathematical reasoning tasks including  
MATH and SuperGPQA datasets. It evaluates final answer accuracy, step-by-step reasoning  
validity, logical coherence, and intermediate consistency using both rule-based methods  
and LLM-as-a-judge evaluation.  
"""
  
import re  
import ast  
import math  
import sympy as sp  
from sympy import sympify, latex, simplify  
from sympy.parsing.latex import parse_latex as sympy_parse_latex  
from typing import Dict, List, Tuple, Any, Union, Optional  
import pandas as pd

  
def compute_math_supergpqa_accuracy(  
    results_df: pd.DataFrame,  
    dataset_type: str = 'MATH',  
    judge: str = "gpt-4o-mini@openai",  
    report_detailed: bool = False,  
    debug: bool = False  
) -> Dict[str, Any]:  
    """  
    Comprehensive evaluation for MATH and SuperGPQA datasets.
      
    Evaluates multiple dimensions of mathematical reasoning:  
    - Final answer accuracy (exact match or mathematical equivalence)  
    - Step validity (mathematical correctness of each reasoning step)  
    - Logical coherence (flow and consistency across steps)  
    - Intermediate consistency (values remain consistent throughout)
      
    Uses hybrid evaluation combining rule-based methods with LLM judge for  
    complex cases requiring semantic understanding.
  
    Args:  
        results_df: DataFrame with columns:  
            - 'outputs': Model responses with reasoning and final answer  
            - 'gold': Correct answers  
            - 'id': Optional problem identifier  
            - 'choices': Multiple choice options (SuperGPQA only)  
        dataset_type: Either 'MATH' or 'SUPERGPQA' (default: 'MATH')  
        judge: Model ID for LLM judge evaluation  
        report_detailed: If True, include per-sample detailed results  
        debug: If True, print detailed evaluation information
  
    Returns:  
        Dictionary containing:  
        - accuracy: Overall accuracy score (0.0 to 1.0)  
        - correct_count: Number of correct final answers  
        - total_count: Total number of samples  
        - failure_reasons: List of dicts with failure details per sample  
        - reasoning_metrics: Dict with:  
            - avg_reasoning_score: Overall reasoning quality (0.0 to 1.0)  
            - avg_step_validity: Average step correctness (0.0 to 1.0)  
            - avg_logical_coherence: Average logical flow quality (0.0 to 1.0)  
            - avg_step_consistency: Average consistency score (0.0 to 1.0)  
            - avg_num_steps: Average number of reasoning steps  
        - detailed_results: Per-sample results (if report_detailed=True)  
    """
  
    correct_count = 0  
    total_count = len(results_df)  
    detailed_results = []  
    failure_reasons = []
  
    # Collect all evaluation requests for batch processing  
    evaluation_requests = []
  
    for idx, row in results_df.iterrows():  
        try:  
            # Extract basic information  
            problem_id = row.get('id', idx)  
            gold_answer = row.get('gold', '')  
            model_output = row['outputs']
  
            # Parse reasoning steps for all samples  
            reasoning_steps = parse_reasoning_steps(model_output)
  
            # Store request data for batch processing  
            request_data = {  
                'idx': idx,  
                'problem_id': problem_id,  
                'gold_answer': gold_answer,  
                'model_output': model_output,  
                'reasoning_steps': reasoning_steps,  
                'dataset_type': dataset_type.upper()  
            }
  
            if dataset_type.upper() == 'SUPERGPQA':  
                request_data['choices'] = row.get('choices', '')
  
            evaluation_requests.append(request_data)
  
        except Exception as e:  
            failure_reasons.append({  
                'id': row.get('id', idx),  
                'reason': f"Processing error: {str(e)}",  
                'step_errors': [],  
                'reasoning_score': 0.0  
            })
  
    # Process all samples with batch LLM judge calls  
    if evaluation_requests:  
        detailed_results = batch_evaluate_samples(evaluation_requests, judge, debug)
  
        # Count correct answers  
        for result in detailed_results:  
            if result['final_accuracy']:  
                correct_count += 1  
            else:  
                failure_reasons.append({  
                    'id': result['id'],  
                    'reason': result['failure_reason'],  
                    'step_errors': result['step_errors'],  
                    'reasoning_score': result['reasoning_score']  
                })
  
    # Calculate aggregate metrics  
    accuracy = correct_count / total_count if total_count > 0 else 0.0
  
    # Calculate reasoning quality metrics  
    if detailed_results:  
        avg_reasoning_score = sum(r['reasoning_score'] for r in detailed_results) / len(detailed_results)  
        avg_step_validity = sum(r['step_validity_score'] for r in detailed_results) / len(detailed_results)  
        avg_logical_coherence = sum(r['logical_coherence_score'] for r in detailed_results) / len(detailed_results)  
        avg_step_consistency = sum(r['consistency_score'] for r in detailed_results) / len(detailed_results)  
        avg_num_steps = sum(r['total_steps'] for r in detailed_results) / len(detailed_results)  
    else:  
        avg_reasoning_score = avg_step_validity = avg_logical_coherence = avg_step_consistency = avg_num_steps = 0.0
  
    return {  
        'accuracy': accuracy,  
        'correct_count': correct_count,  
        'total_count': total_count,  
        'failure_reasons': failure_reasons,  
        'reasoning_metrics': {  
            'avg_reasoning_score': avg_reasoning_score,  
            'avg_step_validity': avg_step_validity,  
            'avg_logical_coherence': avg_logical_coherence,  
            'avg_step_consistency': avg_step_consistency,  
            'avg_num_steps': avg_num_steps  
        },  
        'detailed_results': detailed_results if report_detailed else None  
    }

  
def batch_evaluate_samples(  
    evaluation_requests: List[Dict],  
    judge: str,  
    debug: bool = False  
) -> List[Dict]:  
    """  
    Batch process evaluation requests using LLM judge for efficiency.
      
    Prepares batch requests for consistency and coherence evaluation,  
    calls LLM judge once for all samples, then maps results back.  
    This approach significantly reduces API calls and latency.
  
    Args:  
        evaluation_requests: List of dicts with keys:  
            - problem_id, gold_answer, model_output, reasoning_steps, dataset_type  
        judge: Model ID for LLM judge  
        debug: If True, print detailed processing information
  
    Returns:  
        List of detailed result dictionaries, one per sample, containing  
        accuracy, reasoning scores, step errors, and failure reasons  
    """
  
    # Step 1: Prepare all consistency and coherence evaluation requests  
    consistency_requests = []  
    consistency_dataset_types = []  
    coherence_requests = []  
    coherence_dataset_types = []
  
    for req_idx, request in enumerate(evaluation_requests):  
        model_output = request['model_output']  
        reasoning_steps = request['reasoning_steps']  
        dataset_type = request['dataset_type']
  
        if reasoning_steps:  
            # Add to consistency batch  
            consistency_requests.append(model_output)  
            consistency_dataset_types.append(dataset_type)
  
            # Add coherence evaluation for entire response  
            coherence_requests.append({  
                'model_output': model_output,  
                'dataset_type': dataset_type,  
                'sample_idx': req_idx  
            })  
            coherence_dataset_types.append(dataset_type)
  
    # Step 2: Batch process with LLM judge  
    try:  
        # Get consistency scores for all samples  
        consistency_scores = batch_llm_judge_consistency(  
            consistency_requests, consistency_dataset_types, judge  
        ) if consistency_requests else []
  
        # Get coherence scores for all samples  
        coherence_scores = batch_llm_judge_logical_coherence_full(  
            coherence_requests, judge  
        ) if coherence_requests else []
  
    except Exception as e:  
        print(f"Error in batch LLM judge evaluation: {e}")  
        # Fallback to neutral scores  
        consistency_scores = [0.5] * len(consistency_requests)  
        coherence_scores = [0.5] * len(coherence_requests)
  
    # Step 3: Map results back to samples and compute final metrics  
    detailed_results = []
  
    for req_idx, request in enumerate(evaluation_requests):  
        try:  
            result = process_single_sample_with_batch_results(  
                request,  
                consistency_scores[req_idx] if req_idx < len(consistency_scores) else 0.5,  
                coherence_scores[req_idx] if req_idx < len(coherence_scores) else 0.5,  
                debug  
            )  
            detailed_results.append(result)
  
        except Exception as e:  
            # Fallback result for failed processing  
            detailed_results.append({  
                'id': request['problem_id'],  
                'final_accuracy': False,  
                'model_answer': '',  
                'gold_answer': request['gold_answer'],  
                'reasoning_score': 0.0,  
                'step_validity_score': 0.0,  
                'logical_coherence_score': 0.0,  
                'consistency_score': 0.0,  
                'step_errors': [],  
                'total_steps': len(request['reasoning_steps']),  
                'valid_steps': 0,  
                'failure_reason': f"Processing error: {str(e)}"  
            })
  
    return detailed_results

  
def process_single_sample_with_batch_results(  
    request: Dict,  
    consistency_score: float,  
    coherence_score: float,  
    debug: bool = False  
) -> Dict[str, Any]:  
    """  
    Process a single sample using pre-computed batch LLM judge results.
      
    Combines rule-based step evaluation with LLM judge scores for  
    consistency and coherence to produce comprehensive metrics.
  
    Args:  
        request: Dict with problem_id, gold_answer, model_output,   
                reasoning_steps, dataset_type, and optionally choices  
        consistency_score: Pre-computed consistency score from LLM judge (0.0-1.0)  
        coherence_score: Pre-computed coherence score from LLM judge (0.0-1.0)  
        debug: If True, print detailed evaluation information
  
    Returns:  
        Dictionary with final_accuracy, reasoning scores, step errors,  
        and dataset-specific metrics (e.g., elimination_analysis for SuperGPQA)  
    """
  
    problem_id = request['problem_id']  
    gold_answer = request['gold_answer']  
    model_output = request['model_output']  
    reasoning_steps = request['reasoning_steps']  
    dataset_type = request['dataset_type']
  
    # Extract final answer from model output  
    if dataset_type == 'MATH':  
        model_final_answer = extract_math_final_answer(model_output)  
        gold_final_answer = extract_math_final_answer(gold_answer)  
        final_accuracy = compare_mathematical_expressions(model_final_answer, gold_final_answer)  
    elif dataset_type == 'SUPERGPQA':  
        model_final_answer = extract_supergpqa_final_answer(model_output)  
        gold_final_answer = gold_answer.strip()  
        final_accuracy = (model_final_answer == gold_final_answer)  
    else:  
        model_final_answer = ""  
        gold_final_answer = gold_answer  
        final_accuracy = False
  
    # Evaluate each step using heuristic methods (for step validity)  
    step_evaluations = []  
    step_errors = []
  
    for i, step in enumerate(reasoning_steps):  
        if dataset_type == 'MATH':  
            step_eval = evaluate_math_step(step, i)  
        else:  
            choices = request.get('choices', '')  
            choice_options = parse_multiple_choice_options(choices)  
            step_eval = evaluate_reasoning_step(step, i, choice_options)
  
        step_evaluations.append(step_eval)
  
        if not step_eval['is_valid']:  
            step_errors.append({  
                'step_number': i + 1,  
                'step_text': step,  
                'error_type': step_eval['error_type'],  
                'error_description': step_eval['error_description']  
            })
  
    # Calculate step validity score  
    valid_steps = sum(1 for eval in step_evaluations if eval['is_valid'])  
    step_validity_score = valid_steps / len(step_evaluations) if step_evaluations else 0.0
  
    # Use pre-computed coherence score  
    logical_coherence_score = coherence_score if coherence_score is not None else 0.5
  
    # Overall reasoning score (weighted combination)  
    reasoning_score = (step_validity_score +  
                      logical_coherence_score +  
                      consistency_score) / 3
  
    if debug:  
        print(f'Sample {problem_id}:')  
        print(f'  model: {model_final_answer}, gold: {gold_final_answer}')  
        print(f'  reasoning_steps: {reasoning_steps}')  
        print(f'  final_accuracy: {final_accuracy}')  
        print(f'  step_validity_score: {step_validity_score}')  
        print(f'  logical_coherence_score: {logical_coherence_score}')  
        print(f'  consistency_score: {consistency_score}')
  
    result = {  
        'id': problem_id,  
        'final_accuracy': final_accuracy,  
        'model_answer': model_final_answer,  
        'gold_answer': gold_final_answer,  
        'reasoning_score': reasoning_score,  
        'step_validity_score': step_validity_score,  
        'logical_coherence_score': logical_coherence_score,  
        'consistency_score': consistency_score,  
        'step_errors': step_errors,  
        'total_steps': len(reasoning_steps),  
        'valid_steps': valid_steps,  
        'failure_reason': get_failure_reason(final_accuracy, step_errors, reasoning_score)  
    }
  
    # Add dataset-specific metrics  
    if dataset_type == 'SUPERGPQA':  
        choices = request.get('choices', '')  
        choice_options = parse_multiple_choice_options(choices)  
        reasoning_text = extract_reasoning_section(model_output)  
        elimination_analysis = analyze_option_elimination(reasoning_text, choice_options, gold_final_answer)  
        result['elimination_analysis'] = elimination_analysis  
        result['elimination_quality_score'] = elimination_analysis['quality_score']
  
    return result

  
def batch_llm_judge_consistency(  
    model_output: List[str],  
    dataset_types: List[str] = None,  
    judge: str = None  
) -> List[float]:  
    """  
    Batch evaluate mathematical consistency using LLM judge.
      
    Evaluates whether intermediate values and transformations remain  
    consistent throughout the reasoning chain for both MATH and SuperGPQA.
  
    Args:  
        model_output: List of complete model responses with reasoning  
        dataset_types: List of dataset types ('MATH' or 'SUPERGPQA') per response  
        judge: Model ID for LLM judge
  
    Returns:  
        List of consistency scores (0.0 to 1.0), one per sample  
    """
  
    if not model_output:  
        return []
  
    # Default to MATH if no dataset types provided  
    if dataset_types is None:  
        dataset_types = ['MATH'] * len(model_output)
  
    # Prepare DataFrame for LLM judge  
    df_data = []  
    for i, (out, dataset_type) in enumerate(zip(model_output, dataset_types)):  
        df_data.append({  
            'id': f'consistency_{i}',  
            'model_output': out,  
            'dataset_type': dataset_type  
        })
  
    df = pd.DataFrame(df_data)
  
    try:  
        from .llm_judge import llm_judge_eval
  
        # Use the existing llm_judge_eval function for batch processing  
        judge_responses = llm_judge_eval(df, "mathematical_consistency", judge)
  
        # Extract scores from responses  
        scores = []  
        for response in judge_responses:  
            score = response if response else None  
            scores.append(score if score is not None else 0.5)
  
        return scores
  
    except Exception as e:  
        print(f"Error in batch LLM judge consistency evaluation: {e}")  
        return [0.5] * len(model_output)

  
def batch_llm_judge_logical_coherence_full(  
    coherence_requests: List[Dict],  
    judge: str  
) -> List[float]:  
    """  
    Batch evaluate logical coherence using LLM judge for full responses.
      
    Evaluates the overall logical flow and coherence of the entire reasoning  
    chain in a single pass, rather than evaluating individual transitions.  
    This provides a more holistic assessment of reasoning quality.
  
    Args:  
        coherence_requests: List of dicts with keys:  
            - model_output: Complete response with reasoning  
            - dataset_type: 'MATH' or 'SUPERGPQA'  
            - sample_idx: Index for tracking
  
    Returns:  
        List of coherence scores (0.0 to 1.0), one per sample  
    """
  
    if not coherence_requests:  
        return []
  
    # Prepare DataFrame for LLM judge  
    df_data = []  
    for req in coherence_requests:  
        df_data.append({  
            'id': f'coherence_{req["sample_idx"]}',  
            'model_output': req['model_output'],  
            'dataset_type': req['dataset_type']  
        })
  
    df = pd.DataFrame(df_data)
  
    try:  
        from .llm_judge import llm_judge_eval
  
        # Use the logical_coherence_full mode for single-pass evaluation  
        judge_responses = llm_judge_eval(df, "logical_coherence_full", judge)
  
        # Extract scores from responses  
        scores = []  
        for response in judge_responses:  
            score = response if response else None  
            scores.append(score if score is not None else 0.5)
  
        return scores
  
    except Exception as e:  
        print(f"Error in batch LLM judge coherence evaluation: {e}")  
        return [0.5] * len(coherence_requests)

  
def extract_math_final_answer(text: str) -> str:  
    """  
    Extract the final answer from MATH format response.
      
    Tries multiple extraction methods in order of reliability:  
    1. "Answer: <expression>" pattern (model outputs)  
    2. \boxed{...} pattern (LaTeX formatted answers)  
    3. Common conclusion patterns (Therefore, Thus, etc.)  
    4. Standalone mathematical expressions at the end
      
    Args:  
        text: Model response or gold answer containing mathematical reasoning
          
    Returns:  
        Extracted final answer string, or original text if no clear answer found  
    """
  
    # Clean up the text first  
    text = text.strip()
  
    # Method 1: Look for "Answer: " pattern (most reliable for model outputs)  
    answer_match = re.search(r'Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE | re.MULTILINE)  
    if answer_match:  
        return answer_match.group(1).strip()
  
    # Method 2: Look for \boxed{...} pattern (common in gold answers)  
    boxed_matches = list(re.finditer(r'\\boxed\{', text))  
    if boxed_matches:  
        # Find the last \boxed{} occurrence (most likely the final answer)  
        last_match = boxed_matches[-1]  
        start_pos = last_match.end() - 1  # Position of the opening brace
  
        # Find the matching closing brace  
        brace_count = 0  
        pos = start_pos  
        while pos < len(text):  
            if text[pos] == '{':  
                brace_count += 1  
            elif text[pos] == '}':  
                brace_count -= 1  
                if brace_count == 0:  
                    return text[start_pos + 1:pos].strip()  
            pos += 1
  
    # Method 3: Look for common final answer patterns  
    final_patterns = [  
        r'Therefore,?\s*(.+?)(?:\n|$)',  
        r'Final answer:?\s*(.+?)(?:\n|$)',  
        r'The answer is:?\s*(.+?)(?:\n|$)',  
        r'Thus,?\s*(.+?)(?:\n|$)'  
    ]
  
    for pattern in final_patterns:  
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)  
        if match:  
            return match.group(1).strip()
  
    # Method 4: Look for standalone mathematical expressions at the end  
    lines = text.split('\n')  
    for line in reversed(lines):  
        line = line.strip()  
        if line and not line.startswith('Step') and not line.startswith('##'):  
            # Check if it looks like a mathematical expression  
            if (line.startswith('$') and line.endswith('$')) or bool(re.match(r'^[0-9\-+*/^().,\s\\{}$a-zA-Z]+$', line)):  
                return line
  
    return text

  
def extract_supergpqa_final_answer(text: str) -> str:  
    """  
    Extract final answer (single letter) from SuperGPQA format response.
      
    Looks for patterns indicating the chosen multiple choice option.
      
    Args:  
        text: Model response containing reasoning and final answer
          
    Returns:  
        Single uppercase letter (A-J) representing the chosen answer,  
        or original text if no clear answer pattern found  
    """
      
    # Look for "Answer: X" pattern where X is a single letter  
    answer_patterns = [  
        r'Answer:\s*([A-J])\s*(?:\n|$)',  
        r'The answer is\s*([A-J])',  
        r'Therefore,?\s*the answer is\s*([A-J])',  
        r'Final answer:?\s*([A-J])'  
    ]
  
    for pattern in answer_patterns:  
        match = re.search(pattern, text, re.IGNORECASE)  
        if match:  
            return match.group(1).upper()
  
    return text

  
def parse_reasoning_steps(text: str) -> List[str]:  
    """  
    Parse reasoning steps from model output.
      
    Handles multiple formats:  
    - Explicit "Step N:" format  
    - Sentence-based reasoning (for unstructured responses)
      
    Args:  
        text: Model response containing reasoning
          
    Returns:  
        List of reasoning step strings, cleaned and normalized  
    """
  
    # Clean up the text  
    text = text.strip()
  
    # Remove "Reasoning:" prefix if present  
    text = re.sub(r'^Reasoning:\s*', '', text, flags=re.IGNORECASE)
  
    # Remove "Answer:" section and everything after  
    text = re.split(r'\nAnswer:|Answer:', text, flags=re.IGNORECASE)[0]
  
    # Pattern 1: "Step X:" format (most common in model outputs)  
    step_matches = re.findall(r'Step\s+(\d+):\s*(.+?)(?=Step\s+\d+:|Answer:|The final answer|$)',  
                             text, re.DOTALL | re.IGNORECASE)  
    if step_matches:  
        steps = []  
        for step_num, step_content in step_matches:  
            content = step_content.strip()  
            content = re.sub(r'\s+', ' ', content)  
            if content:  
                steps.append(content)  
        return steps
  
    # Pattern 2: Split by sentences for unstructured reasoning  
    sentences = re.split(r'\.(?:\s+|$)', text)  
    steps = []
  
    for sentence in sentences:  
        sentence = sentence.strip()  
        # More lenient filtering for SuperGPQA  
        if (len(sentence) > 8 and  
            not sentence.startswith(('Step', '##')) and  
            not re.match(r'^\d+[\.)]\s*', sentence)):  
            # Clean up whitespace  
            sentence = re.sub(r'\s+', ' ', sentence)  
            steps.append(sentence)
  
    return steps

  
def evaluate_math_step(step: str, step_number: int) -> Dict[str, Any]:  
    """  
    Evaluate validity of a mathematical reasoning step.
      
    Checks for:  
    - Sufficient detail and length  
    - Mathematical content (expressions, equations)  
    - Reasoning indicators (keywords like "solve", "substitute")  
    - Logical structure (conditionals, conclusions)  
    - Mathematical operation keywords
      
    Args:  
        step: Single reasoning step text  
        step_number: Step index for tracking
          
    Returns:  
        Dictionary with:  
        - is_valid: Boolean indicating step validity  
        - error_type: Type of error if invalid  
        - error_description: Detailed error explanation  
        - has_math: Whether step contains mathematical content  
        - has_logical_structure: Whether step has logical flow  
        - reasoning_indicators: List of reasoning keywords found  
    """
      
    try:  
        step_clean = step.strip()
  
        # Rule 1: Check for sufficient detail  
        if len(step_clean) < 8:  
            return {  
                'is_valid': False,  
                'error_type': 'insufficient_detail',  
                'error_description': "Step too brief to contain meaningful reasoning",  
                'has_math': False,  
                'has_logical_structure': False,  
                'reasoning_indicators': []  
            }
  
        # Rule 2: Extract and validate mathematical content  
        math_expressions = extract_mathematical_expressions(step_clean)  
        equation_patterns = [  
            r'[a-zA-Z0-9\s]*=\s*[a-zA-Z0-9\s\+\-\*/\(\)\.]+',  # Equations  
            r'\d+[\+\-\*/]\d+',  # Basic arithmetic  
            r'[a-zA-Z]\s*=\s*\d+',  # Variable assignments  
            r'\\frac\{[^}]+\}\{[^}]+\}',  # Fractions  
            r'\([^)]*[\+\-\*/][^)]*\)',  # Parenthetical expressions  
            r'\d+\^\d+',  # Exponents  
            r'sqrt\([^)]+\)',  # Square roots  
        ]
  
        has_math_content = bool(math_expressions) or any(re.search(pattern, step_clean) for pattern in equation_patterns)
  
        # Rule 3: Check for mathematical reasoning indicators  
        math_reasoning_indicators = [  
            'substitute', 'solve', 'calculate', 'simplify', 'factor', 'expand',  
            'multiply', 'divide', 'add', 'subtract', 'equals', 'therefore',  
            'thus', 'hence', 'so', 'since', 'given', 'let', 'assume',  
            'isolate', 'rearrange', 'combine', 'cancel', 'cross multiply'  
        ]
  
        reasoning_indicators = [indicator for indicator in math_reasoning_indicators  
                              if indicator in step_clean.lower()]
  
        # Rule 4: Check for logical mathematical structure  
        logical_structure_patterns = [  
            r'if\s+.*then',  # Conditional statements  
            r'since\s+.*[,.]',  # Causal reasoning  
            r'because\s+.*[,.]',  # Explanatory reasoning  
            r'therefore\s+.*',  # Conclusions  
            r'thus\s+.*',  # Conclusions  
            r'so\s+.*',  # Conclusions  
            r'given\s+.*[,.]',  # Given information  
            r'let\s+.*[,.]',  # Variable definitions  
        ]
  
        has_logical_structure = any(re.search(pattern, step_clean.lower()) for pattern in logical_structure_patterns)
  
        # Rule 5: Check for mathematical operation keywords  
        has_operation_keywords = has_mathematical_operation_keywords(step_clean)
  
        # Determine validity based on rules  
        validity_score = 0  
        error_reasons = []
  
        # Mathematical content  
        if has_math_content:  
            validity_score += 1  
        else:  
            error_reasons.append("lacks mathematical content")
  
        # Reasoning indicators  
        if reasoning_indicators:  
            validity_score += 1  
        else:  
            error_reasons.append("lacks mathematical reasoning keywords")
  
        # Logical structure  
        if has_logical_structure or has_operation_keywords:  
            validity_score += 1  
        else:  
            error_reasons.append("lacks logical mathematical structure")
  
        # Final validity determination  
        is_valid = validity_score > 0
  
        if not is_valid:  
            error_type = "weak_mathematical_reasoning"  
            error_description = f"Step fails validity criteria: {'; '.join(error_reasons)}"  
        else:  
            error_type = None  
            error_description = ""
  
        return {  
            'is_valid': is_valid,  
            'error_type': error_type,  
            'error_description': error_description,  
            'has_math': has_math_content,  
            'has_logical_structure': has_logical_structure,  
            'reasoning_indicators': reasoning_indicators  
        }
  
    except Exception as e:  
        return {  
            'is_valid': False,  
            'error_type': 'processing_error',  
            'error_description': f"Error evaluating step: {str(e)}",  
            'has_math': False,  
            'has_logical_structure': False,  
            'reasoning_indicators': []  
        }

  
def has_mathematical_operation_keywords(step_text: str) -> bool:  
    """  
    Check for mathematical operation keywords indicating valid reasoning.
      
    Covers a comprehensive list of mathematical operations across domains:  
    arithmetic, algebra, calculus, geometry, statistics, etc.
      
    Args:  
        step_text: Reasoning step text to check
          
    Returns:  
        True if step contains mathematical operation keywords, False otherwise  
    """
      
    step_lower = step_text.lower()
  
    operation_keywords = [  
        # Basic arithmetic operations  
        'from both sides', 'both sides', 'add to both sides', 'subtract from both sides',  
        'multiply both sides', 'divide both sides', 'cross multiply', 'cross multiplication',
  
        # Algebraic operations  
        'factor out', 'common factor', 'factor', 'factoring', 'distribute', 'distributive',  
        'foil', 'expand', 'combine like terms', 'collect terms', 'simplify', 'simplification',  
        'isolate', 'rearrange', 'transpose', 'substitute', 'substitution',
  
        # Equation solving  
        'quadratic formula', 'completing the square', 'square root', 'cube root',  
        'logarithm', 'log', 'exponential', 'power rule', 'exponent',
  
        # Trigonometry  
        'trigonometric', 'sine', 'cosine', 'tangent', 'sin', 'cos', 'tan',  
        'inverse trig', 'arcsin', 'arccos', 'arctan', 'pythagorean theorem',  
        'law of sines', 'law of cosines', 'unit circle',
  
        # Calculus  
        'derivative', 'differentiate', 'integration', 'integrate', 'antiderivative',  
        'chain rule', 'product rule', 'quotient rule', 'fundamental theorem',  
        'limit', 'continuity', 'optimization', 'critical point',
  
        # Geometry  
        'pythagorean', 'distance formula', 'midpoint formula', 'slope formula',  
        'area formula', 'volume formula', 'perimeter', 'circumference',  
        'parallel', 'perpendicular', 'congruent', 'similar',
  
        # Linear algebra  
        'matrix', 'determinant', 'inverse matrix', 'transpose', 'eigenvalue',  
        'eigenvector', 'dot product', 'cross product', 'linear combination',
  
        # Number theory  
        'prime factorization', 'greatest common divisor', 'gcd', 'lcm',  
        'least common multiple', 'modular arithmetic', 'remainder',
  
        # Statistics and probability  
        'mean', 'median', 'mode', 'standard deviation', 'variance',  
        'probability', 'combination', 'permutation', 'factorial',
  
        # Set theory and logic  
        'union', 'intersection', 'complement', 'subset', 'element of',  
        'implies', 'if and only if', 'contradiction', 'proof by induction',
  
        # Advanced operations  
        'partial derivative', 'gradient', 'divergence', 'curl', 'laplacian',  
        'fourier transform', 'taylor series', 'maclaurin series',  
        'binomial theorem', 'l\'hopital\'s rule',
  
        # General mathematical processes  
        'cancel', 'cancel out', 'eliminate', 'reduce', 'rationalize',  
        'normalize', 'approximate', 'round', 'truncate', 'estimate',  
        'verify', 'check', 'validate', 'confirm', 'test',
  
        # Comparison and equality  
        'equal', 'equivalent', 'identical', 'greater than', 'less than',  
        'maximum', 'minimum', 'optimize', 'extremum',
  
        # Transformation operations  
        'reflect', 'rotate', 'translate', 'scale', 'transform',  
        'map', 'function composition', 'inverse function'  
    ]
  
    # Direct keyword matching  
    for keyword in operation_keywords:  
        if keyword in step_lower:  
            return True
  
    # Additional pattern matching for mathematical operations  
    math_patterns = [  
        r'\b\w+\s+formula\b',  # Any formula  
        r'\brule\s+of\s+\w+\b',  # Rules (chain rule, etc.)  
        r'\btheorem\b',  # Any theorem  
        r'\bmethod\b',  # Any method  
        r'\btechnique\b',  # Any technique  
        r'\bapproach\b',  # Any approach  
    ]
  
    for pattern in math_patterns:  
        if re.search(pattern, step_lower):  
            return True
  
    return False

  
def evaluate_reasoning_step(  
    step: str,  
    step_number: int,  
    choice_options: List[str]  
) -> Dict[str, Any]:  
    """  
    Evaluate validity of a SuperGPQA reasoning step.
      
    Checks for:  
    - Sufficient detail  
    - References to multiple choice options  
    - Elimination reasoning (ruling out incorrect options)  
    - Logical reasoning structure
      
    Args:  
        step: Single reasoning step text  
        step_number: Step index for tracking  
        choice_options: List of available choice letters (e.g., ['A', 'B', 'C'])
          
    Returns:  
        Dictionary with:  
        - is_valid: Boolean indicating step validity  
        - error_type: Type of error if invalid  
        - error_description: Detailed error explanation  
        - mentions_choices: Whether step references options  
        - choice_mentions: List of choice letters mentioned  
        - has_elimination: Whether step eliminates options  
        - has_reasoning_structure: Whether step has logical flow  
        - reasoning_indicators: List of reasoning keywords found  
    """
      
    try:  
        step_clean = step.strip()
  
        # Rule 1: Check for sufficient detail  
        if len(step_clean) < 8:  
            return {  
                'is_valid': False,  
                'error_type': 'insufficient_detail',  
                'error_description': "Step too brief to contain meaningful reasoning",  
                'mentions_choices': False,  
                'has_elimination': False,  
                'has_reasoning_structure': False,  
                'reasoning_indicators': []  
            }
  
        # Rule 2: Enhanced choice references  
        choice_mentions = []  
        mentions_choices = False
  
        # Flexible choice detection patterns  
        choice_patterns = [  
            r'option\s+([A-J])',  
            r'choice\s+([A-J])',  
            r'answer\s+([A-J])',  
            r'\b([A-J])\b(?:\s+(?:gives|has|is|shows))',  
            r'([A-J])\s+(?:gives|has|is|shows|contains)',  
            r'options?\s+([A-J](?:\s*,\s*[A-J])*(?:\s+and\s+[A-J])?)',  
            r'choices?\s+([A-J](?:\s*,\s*[A-J])*(?:\s+and\s+[A-J])?)',  
            r'answers?\s+([A-J](?:\s*,\s*[A-J])*(?:\s+and\s+[A-J])?)',  
            r'\b([A-J](?:\s*,\s*[A-J])+)(?:\s+(?:all|both|each|have|are|show|contain))',  
            r'\b([A-J])\s+(?:and|or)\s+([A-J])\b',  
            r'\b([A-J])\s+(?:through|to|thru)\s+([A-J])\b',  
            r'\b([A-J])-([A-J])\b'  
        ]
  
        for pattern in choice_patterns:  
            matches = re.findall(pattern, step_clean, re.IGNORECASE)  
            for match in matches:  
                if isinstance(match, tuple):  
                    for m in match:  
                        if m and m.upper() not in choice_mentions:  
                            choice_mentions.append(m.upper())  
                            mentions_choices = True  
                elif match.upper() not in choice_mentions:  
                    choice_mentions.append(match.upper())  
                    mentions_choices = True
  
        # Rule 3: Enhanced elimination reasoning  
        elimination_keywords = [  
            'wrong', 'incorrect', 'missing', 'lacks', 'doesn\'t have',  
            'eliminate', 'rule out', 'cannot be', 'not correct',  
            'impossible', 'not possible', 'exclude', 'dismiss', 'reject',  
            'not applicable', 'does not apply', 'inconsistent', 'invalid'  
        ]
  
        has_elimination = any(keyword in step_clean.lower() for keyword in elimination_keywords)
  
        # Rule 4: Enhanced reasoning structure indicators  
        reasoning_structure_indicators = [  
            'because', 'since', 'therefore', 'thus', 'hence', 'so',  
            'given', 'we need', 'looking at', 'considering', 'based on',  
            'this shows', 'this means', 'we can see', 'it appears',  
            'gives', 'has', 'contains', 'shows', 'indicates'  
        ]
  
        reasoning_indicators = [indicator for indicator in reasoning_structure_indicators  
                              if indicator in step_clean.lower()]  
        has_reasoning_structure = bool(reasoning_indicators)
  
        # Final validity determination  
        is_valid = mentions_choices or has_reasoning_structure or has_elimination
  
        error_reasons = []  
        if not mentions_choices:  
            error_reasons.append('Does not mention choices')  
        if not has_reasoning_structure:  
            error_reasons.append('Has no reasoning structure')  
        if not has_elimination:  
            error_reasons.append('Has no elimination')
  
        if not is_valid:  
            error_type = "weak_reasoning"  
            error_description = f"Step fails validity criteria: {'; '.join(error_reasons)}"  
        else:  
            error_type = None  
            error_description = ""
  
        return {  
            'is_valid': is_valid,  
            'error_type': error_type,  
            'error_description': error_description,  
            'mentions_choices': mentions_choices,  
            'choice_mentions': choice_mentions,  
            'has_elimination': has_elimination,  
            'has_reasoning_structure': has_reasoning_structure,  
            'reasoning_indicators': reasoning_indicators  
        }
  
    except Exception as e:  
        return {  
            'is_valid': False,  
            'error_type': 'processing_error',  
            'error_description': f"Error evaluating step: {str(e)}",  
            'mentions_choices': False,  
            'has_elimination': False,  
            'has_reasoning_structure': False,  
            'reasoning_indicators': []  
        }

  
def compare_mathematical_expressions(expr1: str, expr2: str) -> bool:  
    """  
    Compare two mathematical expressions for equivalence.
      
    Uses multiple comparison methods:  
    1. Direct string comparison (after cleaning)  
    2. Text-only comparison (for names, words)  
    3. Unit-aware comparison (degrees, etc.)  
    4. Numerical comparison (for numbers)  
    5. Tuple/coordinate comparison  
    6. Symbolic comparison using SymPy
      
    Args:  
        expr1: First mathematical expression  
        expr2: Second mathematical expression
          
    Returns:  
        True if expressions are mathematically equivalent, False otherwise  
    """
      
    if not expr1 or not expr2:  
        return False
  
    # Strip whitespace for initial comparison  
    expr1_clean = expr1.strip()  
    expr2_clean = expr2.strip()
  
    try:  
        # 1. Direct string comparison (after basic cleaning)  
        clean1 = clean_mathematical_expression(expr1_clean)  
        clean2 = clean_mathematical_expression(expr2_clean)  
        if clean1 == clean2:  
            return True
  
        # 2. Check for text-only expressions (like names)  
        if (re.match(r'^[a-zA-Z]+$', clean1) and re.match(r'^[a-zA-Z]+$', clean2)):  
            return clean1 == clean2
  
        # 3. Check for expressions with units or special notation  
        if ('°' in expr1_clean) != ('°' in expr2_clean):  
            return False  
        if ('degrees' in expr1_clean.lower()) != ('degrees' in expr2_clean.lower()):  
            return False
  
        # 4. Try numerical comparison for simple numbers  
        try:  
            val1_str = re.sub(r'[,$\s]', '', expr1_clean)  
            val2_str = re.sub(r'[,$\s]', '', expr2_clean)
  
            if (re.match(r'^-?\d*\.?\d+$', val1_str) and  
                re.match(r'^-?\d*\.?\d+$', val2_str)):  
                val1 = float(val1_str)  
                val2 = float(val2_str)  
                return abs(val1 - val2) < 1e-10  
        except:  
            pass
  
        # 5. Check if both expressions represent coordinate pairs/tuples  
        tuple_pattern = r'\(\s*([^,]+),\s*([^)]+)\s*\)'  
        match1 = re.search(tuple_pattern, clean1)  
        match2 = re.search(tuple_pattern, clean2)
  
        if match1 and match2:  
            comp1_1, comp1_2 = match1.groups()  
            comp2_1, comp2_2 = match2.groups()
  
            return (compare_mathematical_expressions(comp1_1.strip(), comp2_1.strip()) and  
                    compare_mathematical_expressions(comp1_2.strip(), comp2_2.strip()))
  
        # 6. Try symbolic comparison using SymPy  
        try:  
            sym1 = sympify_latex_wrap(expr1_clean)  
            sym2 = sympify_latex_wrap(expr2_clean)
  
            if sym1 is not None and sym2 is not None:  
                difference = simplify(sym1 - sym2)  
                return difference == 0 or (hasattr(difference, 'is_zero') and difference.is_zero)  
        except:  
            pass
  
        # Final fallback: compare cleaned expressions  
        return clean1 == clean2
  
    except Exception:  
        return False

  
def extract_mathematical_expressions(text: str) -> List[str]:  
    """  
    Extract mathematical expressions from text using pattern matching.
      
    Args:  
        text: Text containing mathematical expressions
          
    Returns:  
        List of extracted mathematical expression strings  
    """
      
    # Patterns for mathematical expressions  
    patterns = [  
        r'=\s*([^=\n]+)',  # Expressions after equals signs  
        r'\$([^$]+)\$',    # LaTeX math mode  
        r'\\frac\{[^}]+\}\{[^}]+\}',  # Fractions  
        r'\d+\.?\d*\s*[+\-*/]\s*\d+\.?\d*',  # Basic arithmetic  
        r'\([^)]*\d[^)]*\)',  # Expressions in parentheses with numbers  
    ]
  
    expressions = []  
    for pattern in patterns:  
        matches = re.findall(pattern, text)  
        expressions.extend(matches)
  
    return expressions

  
def clean_mathematical_expression(expr: str) -> str:  
    """  
    Clean mathematical expression for comparison by removing formatting.
      
    Removes LaTeX commands, normalizes whitespace, and standardizes  
    mathematical notation without changing mathematical meaning.
      
    Args:  
        expr: Mathematical expression string (may contain LaTeX)
          
    Returns:  
        Cleaned expression string  
    """
      
    if not expr:  
        return ""
  
    # Remove outer dollar signs but preserve inner math  
    cleaned = expr.strip()  
    if cleaned.startswith('$') and cleaned.endswith('$') and cleaned.count('$') == 2:  
        cleaned = cleaned[1:-1]
  
    # Normalize whitespace  
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
  
    # Remove common LaTeX formatting that doesn't affect mathematical meaning  
    replacements = [  
        (r'\\left\(', '('),  
        (r'\\right\)', ')'),  
        (r'\\left\[', '['),  
        (r'\\right\]', ']'),  
        (r'\\left\{', '{'),  
        (r'\\right\}', '}'),  
        (r'\\left\|', '|'),  
        (r'\\right\|', '|'),  
        (r'\\cdot', '*'),  
        (r'\\times', '*'),  
        (r'\\div', '/'),  
        (r'\\text\{([^}]+)\}', r'\1'),  # Remove \text{} wrapper  
    ]
  
    for pattern, replacement in replacements:  
        cleaned = re.sub(pattern, replacement, cleaned)
  
    return cleaned

  
def sympify_latex_wrap(expr: str) -> Union[object, None]:  
    """  
    Wrapper to handle both regular expressions and LaTeX with SymPy.
      
    Tries multiple parsing strategies to convert string expressions  
    into SymPy symbolic objects.
      
    Args:  
        expr: Mathematical expression string (regular or LaTeX format)
          
    Returns:  
        SymPy expression object, or None if parsing fails  
    """
      
    if not expr:  
        return None
  
    try:  
        # First try direct sympify for regular expressions  
        return sympify(expr, evaluate=True)  
    except:  
        pass
  
    try:  
        # Try parsing as LaTeX  
        return sympy_parse_latex(expr)  
    except:  
        pass
  
    try:  
        # Clean the expression and try again  
        cleaned = clean_mathematical_expression(expr)  
        return sympify(cleaned, evaluate=True)  
    except:  
        pass
  
    return None

  
def extract_reasoning_section(text: str) -> str:  
    """  
    Extract the reasoning section from SuperGPQA response.
      
    Separates reasoning from the final answer declaration.
      
    Args:  
        text: Complete model response
          
    Returns:  
        Reasoning section text (everything before "Answer:")  
    """
      
    # Look for "Reasoning:" section  
    reasoning_match = re.search(r'Reasoning:\s*(.+?)(?=Answer:|$)', text, re.DOTALL | re.IGNORECASE)  
    if reasoning_match:  
        return reasoning_match.group(1).strip()
  
    # If no explicit reasoning section, take everything before "Answer:"  
    answer_match = re.search(r'(.+?)Answer:', text, re.DOTALL | re.IGNORECASE)  
    if answer_match:  
        return answer_match.group(1).strip()
  
    return text

  
def parse_multiple_choice_options(choices_text: str) -> List[str]:  
    """  
    Parse multiple choice options from the choices field.
      
    Extracts option letters from formatted choice text.
      
    Args:  
        choices_text: Formatted string containing multiple choice options  
                     (e.g., "A: option1 B: option2 C: option3")
          
    Returns:  
        List of choice letters (e.g., ['A', 'B', 'C'])  
    """
      
    if not choices_text:  
        return []
  
    # Pattern to match "A: option", "B: option", etc.  
    pattern = r'([A-J]):\s*([^A-J:]+?)(?=[A-J]:|$)'  
    matches = re.findall(pattern, choices_text)
  
    return [letter for letter, _ in matches]

  
def analyze_option_elimination(  
    reasoning_text: str,  
    choice_options: List[str],  
    correct_answer: str  
) -> Dict[str, Any]:  
    """  
    Analyze how well the model eliminated incorrect options in SuperGPQA.
      
    Checks whether the reasoning explicitly addresses and eliminates  
    each incorrect option with justification.
      
    Args:  
        reasoning_text: Extracted reasoning section from model output  
        choice_options: List of all available choice letters  
        correct_answer: The correct choice letter
          
    Returns:  
        Dictionary with:  
        - quality_score: Fraction of incorrect options eliminated (0.0-1.0)  
        - eliminated_options: Dict mapping each option to whether it was eliminated  
        - total_incorrect_options: Count of incorrect options  
        - eliminated_count: Count of options explicitly eliminated  
    """
      
    elimination_mentions = {}
  
    for option in choice_options:  
        if option == correct_answer:  
            continue
  
        # Look for explicit elimination of this option  
        elimination_patterns = [  
            f"option {option.lower()}.*(?:incorrect|wrong|eliminate|rule out)",  
            f"{option}.*(?:cannot be|is not|incorrect|wrong)",  
            f"(?:eliminate|rule out).*{option}"  
        ]
  
        eliminated = any(re.search(pattern, reasoning_text.lower()) for pattern in elimination_patterns)  
        elimination_mentions[option] = eliminated
  
    # Calculate quality score  
    total_incorrect = len(choice_options) - 1  # Exclude correct answer  
    eliminated_count = sum(elimination_mentions.values())
  
    quality_score = eliminated_count / total_incorrect if total_incorrect > 0 else 1.0
  
    return {  
        'quality_score': quality_score,  
        'eliminated_options': elimination_mentions,  
        'total_incorrect_options': total_incorrect,  
        'eliminated_count': eliminated_count  
    }

  
def get_failure_reason(  
    final_accuracy: bool,  
    step_errors: List[Dict],  
    reasoning_score: float  
) -> str:  
    """  
    Generate human-readable failure reason for incorrect answers.
      
    Categorizes failures by severity and type to help diagnose issues.
      
    Args:  
        final_accuracy: Whether the final answer was correct  
        step_errors: List of step error dictionaries  
        reasoning_score: Overall reasoning quality score (0.0-1.0)
          
    Returns:  
        Human-readable string describing the failure reason,  
        or "Correct" if the answer was correct  
    """
      
    if final_accuracy:  
        return "Correct"
  
    if reasoning_score < 0.3:  
        return "Poor reasoning quality with multiple step errors"  
    elif step_errors:  
        error_types = [error['error_type'] for error in step_errors]  
        if 'calculation_error' in error_types:  
            return "Mathematical calculation errors in reasoning"  
        elif 'insufficient_detail' in error_types:  
            return "Insufficient detail in reasoning steps"  
        else:  
            return f"Reasoning errors: {', '.join(set(error_types))}"  
    else:  
        return "Incorrect final answer despite reasonable reasoning"  