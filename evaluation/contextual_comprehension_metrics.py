import re  
import numpy as np  
from typing import Dict, List, Any
  
def compute_hotpotqa_boolq_accuracy(results_df, dataset_type='HOTPOTQA'):  
    """  
    Evaluate HotpotQA and BoolQ datasets focusing on:  
    - Final answer accuracy  
    - Evidence hit: whether the answer string occurs in the provided context
  
    Args:  
        results_df: DataFrame with columns 'id', 'outputs', 'gold', 'ctx', 'context'  
        dataset_type: Type of dataset being evaluated ('HOTPOTQA' or 'BOOLQ')
  
    Returns:  
        Dictionary with accuracy metrics, evidence hit rate, counts, failure reasons,  
        and detailed results per item  
    """
  
    correct_count = 0  
    evidence_hit_count = 0  
    total_count = len(results_df)  
    detailed_results = []  
    failure_reasons = []
  
    for idx, row in results_df.iterrows():  
        try:  
            # Extract basic information  
            problem_id = row.get('id', idx)  
            gold_answer = row.get('gold', '')  
            model_output = row['outputs']  
            context = row.get('ctx', '')  
            prompt = row.get('context', '')
  
            # Extract context from prompt if not in separate field  
            if not context and 'Context:' in prompt:  
                context = extract_context_from_prompt(prompt)  
            elif not context and 'Passage:' in prompt:  
                context = extract_passage_from_prompt(prompt)  
            elif (isinstance(context, list) or isinstance(context, np.ndarray)) and len(context) > 0:  
                context = context[0]
  
            # Evaluate based on dataset type  
            if dataset_type.upper() == 'HOTPOTQA':  
                gold_answer = gold_answer['answer']  
                result = evaluate_hotpotqa_response(problem_id, gold_answer, model_output, context)  
            elif dataset_type.upper() == 'BOOLQ':  
                result = evaluate_boolq_response(problem_id, gold_answer, model_output, context)  
            else:  
                raise ValueError(f"Unsupported dataset type: {dataset_type}")
  
            detailed_results.append(result)
  
            if result['final_accuracy']:  
                correct_count += 1
  
            if result['evidence_hit']:  
                evidence_hit_count += 1  
            else:  
                failure_reasons.append({  
                    'id': problem_id,  
                    'reason': result['failure_reason'],  
                    'model_answer': result['model_answer'],  
                    'context_available': result['context_available'],  
                    'evidence_hit': result['evidence_hit']  
                })
  
        except KeyError as e: #Exception as e:  
            failure_reasons.append({  
                'id': row.get('id', idx),  
                'reason': f"Processing error: {str(e)}",  
                'model_answer': row.get('outputs', ''),  
                'context_available': False,  
                'evidence_hit': False  
            })
  
    # Calculate metrics  
    accuracy = correct_count / total_count if total_count > 0 else 0.0  
    evidence_hit_rate = evidence_hit_count / total_count if total_count > 0 else 0.0
  
    return {  
        'accuracy': accuracy,  
        'evidence_hit_rate': evidence_hit_rate,  
        'correct_count': correct_count,  
        'evidence_hit_count': evidence_hit_count,  
        'total_count': total_count,  
        'failure_reasons': failure_reasons,  
        'detailed_results': detailed_results  
    }
  
def evaluate_hotpotqa_response(problem_id: str, gold_answer: str, model_output: str, context: str) -> Dict[str, Any]:  
    """  
    Evaluate HotpotQA response focusing on factual accuracy and evidence usage.
      
    Args:  
        problem_id: Unique identifier for the problem  
        gold_answer: Gold standard answer  
        model_output: Model's response text  
        context: Context text provided to the model
          
    Returns:  
        Dictionary with evaluation results including final_accuracy, evidence_hit,  
        model_answer, gold_answer, context_available, and failure_reason  
    """
  
    # Extract model's answer  
    model_answer = extract_answer_from_output(model_output)
  
    # Check final accuracy (if gold answer is available)  
    final_accuracy = False  
    if gold_answer:  
        final_accuracy = compare_answers(model_answer, gold_answer, is_boolean=False)
  
    # Check evidence hit - whether answer appears in context  
    evidence_hit = check_evidence_hit(model_answer, context)  
    context_available = bool(context and context.strip() and context.strip().lower() != 'none')
  
    # Determine failure reason  
    failure_reason = get_hotpotqa_failure_reason(final_accuracy, evidence_hit, context_available, model_answer)
  
    return {  
        'id': problem_id,  
        'final_accuracy': final_accuracy,  
        'evidence_hit': evidence_hit,  
        'model_answer': model_answer,  
        'gold_answer': gold_answer,  
        'context_available': context_available,  
        'failure_reason': failure_reason  
    }
  
def evaluate_boolq_response(problem_id: str, gold_answer: str, model_output: str, context: str) -> Dict[str, Any]:  
    """  
    Evaluate BoolQ response focusing on yes/no accuracy and evidence usage.
      
    Args:  
        problem_id: Unique identifier for the problem  
        gold_answer: Gold standard boolean answer  
        model_output: Model's response text  
        context: Context passage provided to the model
          
    Returns:  
        Dictionary with evaluation results including final_accuracy, evidence_hit,  
        model_answer, gold_answer, context_available, and failure_reason  
    """
  
    # Extract model's answer (should be Yes/No)  
    model_answer = extract_boolean_answer(model_output)
  
    # Check final accuracy (if gold answer is available)  
    final_accuracy = False  
    if gold_answer:  
        final_accuracy = compare_answers(model_answer, gold_answer, is_boolean=True)
  
    # Check evidence hit - for boolean questions, check if supporting evidence exists  
    evidence_hit = check_boolean_evidence_hit(model_answer, context)  
    context_available = bool(context and context.strip() and context.strip().lower() != 'none')
  
    # Determine failure reason  
    failure_reason = get_boolq_failure_reason(final_accuracy, evidence_hit, context_available, model_answer)
  
    return {  
        'id': problem_id,  
        'final_accuracy': final_accuracy,  
        'evidence_hit': evidence_hit,  
        'model_answer': model_answer,  
        'gold_answer': gold_answer,  
        'context_available': context_available,  
        'failure_reason': failure_reason  
    }
  
def extract_context_from_prompt(prompt: str) -> str:  
    """  
    Extract context section from HotpotQA prompt.
      
    Args:  
        prompt: Full prompt text containing context
          
    Returns:  
        Extracted context string, or empty string if not found  
    """  
    context_match = re.search(r'Context:\s*(.+?)(?=Question:|$)', prompt, re.DOTALL | re.IGNORECASE)  
    if context_match:  
        return context_match.group(1).strip()  
    return ""
  
def extract_passage_from_prompt(prompt: str) -> str:  
    """  
    Extract passage section from BoolQ prompt.
      
    Args:  
        prompt: Full prompt text containing passage
          
    Returns:  
        Extracted passage string, or empty string if not found  
    """  
    passage_match = re.search(r'Passage:\s*(.+?)(?=Question:|$)', prompt, re.DOTALL | re.IGNORECASE)  
    if passage_match:  
        return passage_match.group(1).strip()  
    return ""
  
def extract_answer_from_output(output: str) -> str:  
    """  
    Extract answer from model output using common answer patterns.
      
    Args:  
        output: Model's full output text
          
    Returns:  
        Extracted answer string (falls back to last line if no pattern matches)  
    """  
    # Look for "Answer: " pattern first  
    answer_patterns = [  
        r'Answer:\s*(.+?)(?:\n|$)',  
        r'Final answer:\s*(.+?)(?:\n|$)',  
        r'The answer is:\s*(.+?)(?:\n|$)',  
    ]
  
    for pattern in answer_patterns:  
        match = re.search(pattern, output, re.IGNORECASE | re.MULTILINE)  
        if match:  
            return match.group(1).strip()
  
    # If no explicit answer pattern, take the last line as answer  
    lines = output.strip().split('\n')  
    if lines:  
        return lines[-1].strip()
  
    return output.strip()
  
def extract_boolean_answer(output: str) -> str:  
    """  
    Extract Yes/No answer from BoolQ output.
      
    Args:  
        output: Model's full output text
          
    Returns:  
        Normalized boolean answer ("Yes" or "No"), or original answer if unclear  
    """  
    # First try standard extraction  
    answer = extract_answer_from_output(output)
  
    # Clean and normalize boolean answer  
    answer_lower = answer.lower().strip()
  
    # Look for yes/no patterns  
    if re.search(r'\byes\b', answer_lower):  
        return "Yes"  
    elif re.search(r'\bno\b', answer_lower):  
        return "No"
  
    # Check the full output for yes/no  
    output_lower = output.lower()  
    if re.search(r'\byes\b', output_lower) and not re.search(r'\bno\b', output_lower):  
        return "Yes"  
    elif re.search(r'\bno\b', output_lower) and not re.search(r'\byes\b', output_lower):  
        return "No"
  
    return answer  # Return original if can't determine
  
def check_evidence_hit(model_answer: str, context: str) -> bool:  
    """  
    Check if the model's answer can be found in the provided context.
      
    Args:  
        model_answer: Model's extracted answer  
        context: Context text provided to the model
          
    Returns:  
        True if answer is supported by context, False otherwise  
    """  
    if not model_answer or not context:  
        return False
  
    # Clean both answer and context  
    answer_clean = model_answer.strip().lower()  
    context_clean = context.strip().lower()
  
    # Direct substring match  
    if answer_clean in context_clean:  
        return True
  
    # Check for partial matches (for multi-word answers)  
    answer_words = answer_clean.split()  
    if len(answer_words) > 1:  
        # Check if significant portion of answer words appear in context  
        found_words = sum(1 for word in answer_words if word in context_clean and len(word) > 2)  
        if found_words >= len(answer_words) * 0.7:  # 70% of words found  
            return True
  
    # Check for named entity matches (proper nouns, dates, numbers)  
    if check_named_entity_match(model_answer, context):  
        return True
  
    return False
  
def check_boolean_evidence_hit(model_answer: str, context: str) -> bool:  
    """  
    Check if there's supporting evidence in context for boolean answer.
      
    Args:  
        model_answer: Model's boolean answer  
        context: Context passage provided to the model
          
    Returns:  
        True if context contains relevant evidence, False otherwise  
    """  
    if not context:  
        return False
  
    # For boolean questions, we check if there's any relevant content in context  
    # rather than looking for the exact yes/no answer  
    context_clean = context.strip().lower()
  
    # If context is substantial (more than just a few words), consider it a hit  
    if len(context_clean.split()) > 10:  
        return True
  
    # Check for common boolean indicators in context  
    boolean_indicators = [  
        'true', 'false', 'correct', 'incorrect', 'confirmed', 'denied',  
        'is', 'was', 'are', 'were', 'has', 'have', 'does', 'did',  
        'can', 'cannot', 'will', 'would', 'should', 'must'  
    ]
  
    return any(indicator in context_clean for indicator in boolean_indicators)
  
def check_named_entity_match(answer: str, context: str) -> bool:  
    """  
    Check for named entity matches between answer and context.
      
    Args:  
        answer: Model's answer text  
        context: Context text
          
    Returns:  
        True if named entities (proper nouns, numbers, dates) from answer appear in context  
    """  
  
    # Look for capitalized words (potential proper nouns)  
    answer_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', answer)
  
    if not answer_entities:  
        return False
  
    context_lower = context.lower()
  
    # Check if any named entities from answer appear in context  
    for entity in answer_entities:  
        if entity.lower() in context_lower:  
            return True
  
    # Check for number/date matches  
    answer_numbers = re.findall(r'\b\d+\b', answer)  
    for num in answer_numbers:  
        if num in context:  
            return True
  
    return False
  
def get_hotpotqa_failure_reason(final_accuracy: bool, evidence_hit: bool, context_available: bool, model_answer: str) -> str:  
    """  
    Generate failure reason for HotpotQA evaluation.
      
    Args:  
        final_accuracy: Whether the answer is correct  
        evidence_hit: Whether the answer is supported by context  
        context_available: Whether context was provided  
        model_answer: Model's extracted answer
          
    Returns:  
        String describing the evaluation outcome or failure reason  
    """  
    if final_accuracy and evidence_hit:  
        return "Correct answer with evidence support"  
    elif final_accuracy and not evidence_hit:  
        if not context_available:  
            return "Correct answer but no context provided"  
        else:  
            return "Correct answer but not supported by provided context"  
    elif not final_accuracy and evidence_hit:  
        return "Incorrect answer despite evidence being available in context"  
    elif not final_accuracy and not evidence_hit:  
        if not context_available:  
            return "Incorrect answer and no context provided"  
        elif not model_answer or model_answer.strip() == "":  
            return "No answer provided"  
        else:  
            return "Incorrect answer not supported by context"
  
def get_boolq_failure_reason(final_accuracy: bool, evidence_hit: bool, context_available: bool, model_answer: str) -> str:  
    """  
    Generate failure reason for BoolQ evaluation.
      
    Args:  
        final_accuracy: Whether the boolean answer is correct  
        evidence_hit: Whether the answer is supported by passage  
        context_available: Whether passage was provided  
        model_answer: Model's extracted boolean answer
          
    Returns:  
        String describing the evaluation outcome or failure reason  
    """  
    if final_accuracy and evidence_hit:  
        return "Correct boolean answer with context support"  
    elif final_accuracy and not evidence_hit:  
        if not context_available:  
            return "Correct boolean answer but no passage provided"  
        else:  
            return "Correct boolean answer but passage lacks clear evidence"  
    elif not final_accuracy and evidence_hit:  
        return "Incorrect boolean answer despite relevant passage content"  
    elif not final_accuracy and not evidence_hit:  
        if not context_available:  
            return "Incorrect boolean answer and no passage provided"  
        elif not model_answer or model_answer.strip() == "":  
            return "No boolean answer provided"  
        elif model_answer.lower() not in ['yes', 'no']:  
            return "Invalid boolean answer format (not Yes/No)"  
        else:  
            return "Incorrect boolean answer not supported by passage"
  
def compare_answers(model_answer: str, gold_answer: str, is_boolean: bool = False) -> bool:  
    """  
    Compare model answer with gold answer.
      
    Args:  
        model_answer: Model's answer  
        gold_answer: Gold standard answer  
        is_boolean: Whether this is a boolean (True/False, Yes/No) question
          
    Returns:  
        True if answers match (exact, numerical, or fuzzy), False otherwise  
    """  
    if not model_answer or not gold_answer:  
        return False
  
    # Handle boolean/True-False questions  
    if is_boolean:  
        return compare_boolean_answers(model_answer, gold_answer)
  
    # Clean and normalize both answers  
    model_clean = model_answer.strip().lower()  
    gold_clean = gold_answer.strip().lower()
  
    # Try exact string match first (fastest)  
    if model_clean == gold_clean:  
        return True
  
    # Try numerical comparison  
    if compare_numerical_answers(model_clean, gold_clean):  
        return True
  
    # Try fuzzy string matching for factual answers  
    if compare_fuzzy_match(model_clean, gold_clean):  
        return True
  
    return False
  
def compare_boolean_answers(model_answer: str, gold_answer: str) -> bool:  
    """  
    Compare boolean/True-False answers.
      
    Args:  
        model_answer: Model's boolean answer (can be string or other type)  
        gold_answer: Gold standard boolean answer (can be string or other type)
          
    Returns:  
        True if both answers represent the same boolean value, False otherwise  
    """  
    # Normalize boolean representations  
    true_variants = ['true', 'yes', 't', '1', 'correct', 1.0]  
    false_variants = ['false', 'no', 'f', '0', 'incorrect', None]
  
    model_bool = None  
    gold_bool = None
  
    if isinstance(model_answer, str):  
        model_answer = model_answer.lower()
  
    if isinstance(gold_answer, str):  
        gold_answer = gold_answer.lower()
  
    try:  
        if np.isnan(gold_answer):  
            gold_answer = None  
    except TypeError:
        # gold_answer is non-numeric; np.isnan is not applicable, so leave it unchanged
        pass
  
    if model_answer in true_variants:  
        model_bool = True  
    elif model_answer in false_variants:  
        model_bool = False
  
    if gold_answer in true_variants:  
        gold_bool = True  
    elif gold_answer in false_variants:  
        gold_bool = False  
    return model_bool == gold_bool and model_bool is not None
  
def compare_numerical_answers(model_answer: str, gold_answer: str) -> bool:  
    """  
    Compare numerical answers with tolerance.
      
    Args:  
        model_answer: Model's numerical answer as string  
        gold_answer: Gold standard numerical answer as string
          
    Returns:  
        True if numbers match within tolerance, False otherwise  
    """  
    try:  
        model_val = float(model_answer.replace(',', ''))  
        gold_val = float(gold_answer.replace(',', ''))
  
        # Use relative tolerance for comparison  
        if abs(gold_val) > 1e-10:  
            return abs(model_val - gold_val) / abs(gold_val) < 1e-6  
        else:  
            return abs(model_val - gold_val) < 1e-10
  
    except (ValueError, TypeError):  
        return False
  
def compare_fuzzy_match(model_answer: str, gold_answer: str, threshold: float = 0.8) -> bool:  
    """  
    Compare answers using fuzzy string matching for factual questions.
      
    Args:  
        model_answer: Model's answer  
        gold_answer: Gold standard answer  
        threshold: Similarity threshold (0.0 to 1.0) for considering a match
          
    Returns:  
        True if answers are similar enough or one contains the other, False otherwise  
    """  
    try:  
        from difflib import SequenceMatcher
  
        # Calculate similarity ratio  
        similarity = SequenceMatcher(None, model_answer, gold_answer).ratio()
  
        if similarity >= threshold:  
            return True
  
        # Also check if one answer is contained in the other (for cases like "Paris" vs "Paris, France")  
        if model_answer in gold_answer or gold_answer in model_answer:  
            return True
  
        return False
  
    except ImportError:  
        # Fallback to simple substring matching if difflib not available  
        return model_answer in gold_answer or gold_answer in model_answer  