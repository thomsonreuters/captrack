import re  
from typing import Dict, List, Tuple, Any  
from .llm_judge import llm_judge_eval  
import pandas as pd
  
def compute_ragtruth_accuracy(results_df, judge="gpt-4o-mini@openai", report_detailed=False, debug=False):  
    """  
    Evaluate RAGTruth responses focusing on epistemic faithfulness and grounding.
  
    Metrics:  
    - Final Accuracy: Whether the answer is correct  
    - Hallucination Rate: Proportion of statements unsupported by evidence  
    - Evidence Utilization: How effectively the model uses available evidence
  
    Args:  
        results_df: DataFrame with columns ['id', 'outputs', 'gold', 'ctx']  
        judge: Model ID for LLM judge evaluation  
        report_detailed: If True, include detailed results in output  
        debug: If True, print debug information during evaluation
  
    Returns:  
        Dictionary with accuracy, hallucination rate, evidence utilization metrics,  
        failure reasons, and optionally detailed results per item  
    """
  
    # First loop: collect all judge prompts  
    accuracy_prompts = []  
    claim_support_prompts = []  
    sample_info = []  # Track which prompts belong to which samples
  
    for idx, row in results_df.iterrows():  
        try:  
            # Extract basic information  
            problem_id = row.get('id', idx)  
            model_output = row['outputs']  
            gold_answer = row.get('gold', '')  
            context = row.get('ctx', '')  
            if isinstance(context, list) or isinstance(context, np.ndarray):  
                context = ' '.join(context)
  
            # Store sample info for later processing  
            sample_data = {  
                'idx': idx,  
                'problem_id': problem_id,  
                'model_output': model_output,  
                'gold_answer': gold_answer,  
                'context': context,  
                'accuracy_prompt_idx': len(accuracy_prompts),  # Index in accuracy prompts  
                'claim_support_start_idx': len(claim_support_prompts)  # Start index in claim support prompts  
            }
  
            # Create accuracy assessment prompt  
            gold_claims = extract_claims_from_response(gold_answer)  
            sample_data['num_gold_claims'] = len(gold_claims)
            for claim in gold_claims:  
                accuracy_prompts.append({  
                    'claim': claim,  
                    'context': model_output  
                })
  
            # Extract claims and create claim support prompts  
            model_claims = extract_claims_from_response(model_output)  
            sample_data['num_claims'] = len(model_claims)
  
            for claim in model_claims:  
                claim_support_prompts.append({  
                    'claim': claim,  
                    'context': context  
                })
  
            sample_info.append(sample_data)
  
        except Exception as e:  
            raise e
  
    # Call LLM judges for both tasks  
    accuracy_results = []  
    claim_support_results = []
  
    if accuracy_prompts:  
        accuracy_df = pd.DataFrame(accuracy_prompts)  
        accuracy_results = llm_judge_eval(accuracy_df, "claim_support", judge)
  
    if claim_support_prompts:  
        claim_support_df = pd.DataFrame(claim_support_prompts)  
        claim_support_results = llm_judge_eval(claim_support_df, "claim_support", judge)
  
    if debug:  
        if accuracy_prompts:  
            print(f'check accuracy:\ngold_answer: {accuracy_df["claim"]}\nmodel_answer: {accuracy_df["context"]}')  
            print(f'accuracy_results: {accuracy_results}')  
        print()  
        if claim_support_prompts:  
            print(f'check claims:\n{claim_support_df["claim"]}')  
            print(f'claim_support_results: {claim_support_results}')
  
    # Second loop: process judge responses and compute metrics  
    correct_count = 0  
    total_count = len(results_df)  
    detailed_results = []  
    failure_reasons = []
  
    total_accuracy = 0.0  
    total_hallucination_score = 0.0  
    total_evidence_utilization_score = 0.0  
    total_num_claims = 0.0
  
    for sample in sample_info:  
        try:  
            if 'error' in sample:  
                # Handle previous errors  
                failure_reasons.append({  
                    'id': sample['problem_id'],  
                    'reason': f"Processing error: {sample['error']}",  
                    'hallucination_rate': 1.0,  
                    'unsupported_claims': []  
                })  
                continue
  
            # Get accuracy result  
            final_accuracy = 0.0  
            start_idx = sample.get('accuracy_prompt_idx', -1)  
            if start_idx >= 0 and start_idx < len(accuracy_results):  
                # Determine how many gold claims this sample has; default to 1 to preserve  
                # existing behavior when per-sample gold claim counts are not tracked  
                num_gold_claims = sample.get('num_gold_claims', 1)  
                end_idx = min(start_idx + num_gold_claims, len(accuracy_results))  
                acc_slice = accuracy_results[start_idx:end_idx]  
  
                # Map raw judge outputs to binary correctness scores and aggregate.  
                # Here we use the mean correctness across all gold claims for the sample.  
                binary_scores = [1.0 if str(acc_raw) == 'True' else 0.0 for acc_raw in acc_slice]  
                if binary_scores:  
                    final_accuracy = sum(binary_scores) / len(binary_scores)  
  
            # Get claim support results  
            claim_evaluations = []  
            unsupported_claims = []
  
            if sample['num_claims'] > 0:  
                model_claims = extract_claims_from_response(sample['model_output'])
  
                for i, claim in enumerate(model_claims):  
                    claim_idx = sample['claim_support_start_idx'] + i  
                    is_supported = False
  
                    if claim_idx < len(claim_support_results):  
                        support_response = claim_support_results[claim_idx]  
                        is_supported = (support_response == "True")
  
                    claim_evaluations.append({  
                        'claim': claim,  
                        'is_supported': is_supported  
                    })
  
                    if not is_supported:  
                        unsupported_claims.append(claim)
  
            # Calculate metrics using the existing logic  
            result = process_sample_results(  
                sample['problem_id'],  
                final_accuracy,  
                claim_evaluations,  
                unsupported_claims,  
                sample['model_output'],  
                sample['context']  
            )
  
            detailed_results.append(result)
  
            # Update counters  
            if result['final_accuracy'] > 0.2:  
                correct_count += 1  
            else:  
                failure_reasons.append({  
                    'id': sample['problem_id'],  
                    'reason': result['failure_reason'],  
                    'hallucination_rate': result['hallucination_rate'],  
                    'unsupported_claims': result['unsupported_claims']  
                })
  
            # Accumulate scores  
            total_accuracy += result['final_accuracy']  
            total_hallucination_score += result['hallucination_rate']  
            total_evidence_utilization_score += result['evidence_utilization_score']  
            total_num_claims += result['total_claims']
  
        except Exception as e:  
            failure_reasons.append({  
                'id': sample.get('problem_id', 'unknown'),  
                'reason': f"Processing error: {str(e)}",  
                'hallucination_rate': 1.0,  
                'unsupported_claims': []  
            })
  
    # Calculate aggregate metrics  
    accuracy = total_accuracy / total_count if total_count > 0 else 0.0  
    avg_hallucination_rate = total_hallucination_score / total_count if total_count > 0 else 0.0  
    avg_evidence_utilization = total_evidence_utilization_score / total_count if total_count > 0 else 0.0  
  
    return {  
        'accuracy': accuracy,  
        'correct_count': correct_count,  
        'total_count': total_count,  
        'hallucination_rate': avg_hallucination_rate,  
        'evidence_utilization_score': avg_evidence_utilization,  
        'failure_reasons': failure_reasons,  
        'detailed_results': detailed_results if report_detailed else None  
    }

  
def process_sample_results(problem_id, final_accuracy, claim_evaluations, unsupported_claims, model_output, context):  
    """  
    Process the judge results for a single sample and compute metrics.
      
    Args:  
        problem_id: Unique identifier for the problem  
        final_accuracy: Whether the final answer is correct  
        claim_evaluations: List of claim evaluation dictionaries  
        unsupported_claims: List of claims not supported by context  
        model_output: Model's response text  
        context: Context text provided to the model
          
    Returns:  
        Dictionary with computed metrics including accuracy, hallucination rate,  
        evidence utilization, and failure reason  
    """
  
    # Calculate hallucination rate  
    if len(claim_evaluations) > 0:  
        supported_claims = sum(1 for eval in claim_evaluations if eval['is_supported'])  
        hallucination_rate = 1.0 - (supported_claims / len(claim_evaluations))  
    else:  
        hallucination_rate = 0.0  # No claims to evaluate
  
    # Evaluate evidence utilization  
    evidence_utilization_score = evaluate_evidence_utilization(model_output, context)
  
    # Determine failure reason  
    failure_reason = get_ragtruth_failure_reason(  
        final_accuracy, hallucination_rate, len(unsupported_claims)  
    )
  
    return {  
        'id': problem_id,  
        'final_accuracy': final_accuracy,  
        'hallucination_rate': hallucination_rate,  
        'evidence_utilization_score': evidence_utilization_score,  
        'total_claims': len(claim_evaluations),  
        'supported_claims': len(claim_evaluations) - len(unsupported_claims),  
        'unsupported_claims': unsupported_claims,  
        'failure_reason': failure_reason,  
        'claim_evaluations': claim_evaluations  
    }

  
def extract_claims_from_response(response: str) -> List[str]:  
    """  
    Extract factual claims from model response.
      
    Args:  
        response: Model response text to extract claims from
          
    Returns:  
        List of factual claim strings extracted from the response  
    """  
    if not response or not response.strip():  
        return []
  
    # Split response into sentences  
    sentences = split_into_sentences(response)
  
    claims = []  
    for sentence in sentences:  
        # Filter out non-factual sentences (questions, instructions, etc.)  
        if is_factual_claim(sentence):  
            claims.append(sentence.strip())
  
    return claims

  
def split_into_sentences(text: str) -> List[str]:  
    """  
    Split text into sentences handling both structured and unstructured text.
      
    Args:  
        text: Text to split into sentences
          
    Returns:  
        List of cleaned sentence strings  
    """  
  
    if not text or not text.strip():  
        return []
  
    # First, normalize line breaks and clean up whitespace  
    text = re.sub(r"\n+", " ", text)  # Convert newlines to spaces  
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace  
    text = text.strip()
  
    # Simple but effective sentence splitting  
    # Split on periods followed by space and capital letter, or sentence endings at end  
    sentences = re.split(r"\.(?:\s+(?=[A-Z])|$)|[!?]+(?:\s+|$)", text)
  
    cleaned_sentences = []  
    for sentence in sentences:  
        if not sentence:  
            continue
  
        sentence = sentence.strip()  
        if len(sentence) < 6:  # Skip very short fragments  
            continue
  
        # Clean up common structural markers but keep the content  
        cleaned = re.sub(r"^(?:Step \d+:|Answer:|Conclusion:|Therefore:|Finally:|However:|Moreover:|Additionally:)\s*", "", sentence, flags=re.IGNORECASE)  
        cleaned = cleaned.strip()
  
        # Remove any trailing periods that might remain  
        cleaned = re.sub(r"\.$", "", cleaned).strip()
  
        if len(cleaned) > 5:  
            cleaned_sentences.append(cleaned)
  
    return cleaned_sentences

  
def is_factual_claim(sentence: str) -> bool:  
    """  
    Determine if a sentence contains a factual claim.
      
    Args:  
        sentence: Sentence to evaluate
          
    Returns:  
        True if sentence contains a factual claim, False if it's a question, instruction, or non-factual  
    """  
    sentence_lower = sentence.lower().strip()
  
    # Skip questions  
    if sentence_lower.endswith("?"):  
        return False
  
    # Skip very short sentences or fragments  
    words = sentence.split()  
    if len(words) < 3:  
        return False
  
    # Skip instructions, commands, and subjective expressions  
    non_factual_starters = [  
        "please", "let", "try", "make sure", "remember", "note that",  
        "i think", "i believe", "in my opinion", "it seems", "perhaps",  
        "maybe", "possibly", "probably", "how", "what", "when", "where", "why"  
    ]  
    if any(sentence_lower.startswith(starter) for starter in non_factual_starters):  
        return False
  
    # Skip sentences that are primarily procedural  
    if sentence_lower.startswith(("step ", "first", "second", "third", "next", "then", "finally")):  
        # But allow if it contains factual content after the procedural marker  
        content_after_marker = re.sub(r"^(step \d+[:\.]?\s*|first[,\.]?\s*|second[,\.]?\s*|third[,\.]?\s*|next[,\.]?\s*|then[,\.]?\s*|finally[,\.]?\s*)", "", sentence_lower)  
        if len(content_after_marker.split()) < 3:  
            return False
  
    # Exclude pure expressions of uncertainty or methodology  
    uncertainty_patterns = [  
        "based on", "according to", "it appears", "this suggests",  
        "we can see", "this shows", "this indicates", "this means"  
    ]  
    if any(pattern in sentence_lower for pattern in uncertainty_patterns):  
        # These might introduce facts, so check if there's substantial content after  
        words_after_pattern = len([w for pattern in uncertainty_patterns  
                                 if pattern in sentence_lower  
                                 for w in sentence_lower.split(pattern)[-1].split()])  
        if words_after_pattern < 3:  
            return False
  
    return True

  
def extract_key_terms(text: str) -> List[str]:  
    """  
    Extract key terms (nouns, numbers, proper nouns) from text.
      
    Args:  
        text: Text to extract key terms from
          
    Returns:  
        List of key term strings (filtered to exclude stop words and short words)  
    """  
  
    # Remove common stop words  
    stop_words = {  
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',  
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',  
        'has', 'had', 'will', 'would', 'could', 'should', 'this', 'that', 'these', 'those'  
    }
  
    # Extract words (alphanumeric sequences)  
    words = re.findall(r'\b\w+\b', text.lower())
  
    # Filter out stop words and short words  
    key_terms = [word for word in words if word not in stop_words and len(word) > 2]
  
    # Also extract numbers and measurements  
    numbers = re.findall(r'\b\d+(?:\.\d+)?\s*(?:inches?|feet?|cm|mm|meters?|degrees?|%)\b', text.lower())  
    key_terms.extend(numbers)
  
    return key_terms

  
def check_contextual_relationship(claim: str, context: str, key_terms: List[str]) -> bool:  
    """  
    Check if the relationship between key terms in claim is preserved in context.
      
    Args:  
        claim: Claim text to verify  
        context: Context text to check against  
        key_terms: List of key terms extracted from claim
          
    Returns:  
        True if contextual relationship is preserved, False if contradictions detected  
    """  
    # This is a simplified heuristic - in practice, you might use NLI models
  
    # Look for negation patterns that might contradict the claim  
    negation_patterns = ['not', 'never', 'cannot', 'does not', 'is not', 'are not']
  
    claim_lower = claim.lower()  
    context_lower = context.lower()
  
    # If claim contains negation, be more strict  
    claim_has_negation = any(neg in claim_lower for neg in negation_patterns)  
    context_has_negation = any(neg in context_lower for neg in negation_patterns)
  
    # Simple heuristic: if negation patterns differ significantly, be cautious  
    if claim_has_negation != context_has_negation:  
        return False
  
    return True

  
def evaluate_evidence_utilization(response: str, context: str) -> float:  
    """  
    Evaluate how well the response utilizes available evidence from context.
      
    Args:  
        response: Model's response text  
        context: Context text available to the model
          
    Returns:  
        Float score (0.0 to 1.0) representing evidence utilization quality  
    """  
    if not context or not context.strip():  
        return 1.0  # No context available, so perfect utilization by default
  
    if not response or not response.strip():  
        return 0.0  # No response, so no utilization
  
    # Extract key information from context  
    context_terms = extract_key_terms(context)  
    response_terms = extract_key_terms(response)
  
    if len(context_terms) == 0:  
        return 1.0
  
    # Calculate overlap between response and context terms  
    overlap = set(response_terms) & set(context_terms)  
    utilization_score = len(overlap) / len(context_terms) if context_terms else 0.0
  
    # Bonus for using specific measurements, numbers, or technical terms from context  
    specific_patterns = [  
        r'\b\d+(?:\.\d+)?\s*(?:inches?|feet?|cm|mm|meters?|degrees?|%)\b',  
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns  
        r'\bstep\s+\d+\b'  # Step references  
    ]
  
    context_specific_terms = set()  
    response_specific_terms = set()
  
    for pattern in specific_patterns:  
        context_specific_terms.update(re.findall(pattern, context, re.IGNORECASE))  
        response_specific_terms.update(re.findall(pattern, response, re.IGNORECASE))
  
    if context_specific_terms:  
        specific_overlap = len(context_specific_terms & response_specific_terms)  
        specific_bonus = specific_overlap / len(context_specific_terms) * 0.2  
        utilization_score = min(1.0, utilization_score + specific_bonus)
  
    return utilization_score

  
def extract_instructional_steps(text: str) -> List[str]:  
    """  
    Extract instructional steps from text.
      
    Args:  
        text: Text containing instructions
          
    Returns:  
        List of extracted instructional step strings  
    """  
  
    # Look for numbered steps  
    numbered_steps = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', text, re.DOTALL)  
    if numbered_steps:  
        return [step.strip() for step in numbered_steps]
  
    # Look for action verbs at sentence beginnings  
    sentences = split_into_sentences(text)  
    action_verbs = ['draw', 'use', 'create', 'make', 'set', 'click', 'select', 'measure']
  
    steps = []  
    for sentence in sentences:  
        sentence_words = sentence.lower().split()  
        if sentence_words and any(sentence_words[0].startswith(verb) for verb in action_verbs):  
            steps.append(sentence)
  
    return steps

  
def calculate_step_overlap(model_steps: List[str], gold_steps: List[str]) -> float:  
    """  
    Calculate overlap between model and gold instructional steps.
      
    Args:  
        model_steps: Steps extracted from model output  
        gold_steps: Steps extracted from gold answer
          
    Returns:  
        Float score (0.0 to 1.0) representing the proportion of gold steps covered  
    """  
    if not gold_steps:  
        return 1.0
  
    if not model_steps:  
        return 0.0
  
    # Simple approach: check how many gold step concepts appear in model steps  
    overlap_count = 0
  
    for gold_step in gold_steps:  
        gold_terms = set(extract_key_terms(gold_step))
  
        for model_step in model_steps:  
            model_terms = set(extract_key_terms(model_step))
  
            # If significant overlap in key terms, consider it a match  
            if len(gold_terms & model_terms) >= len(gold_terms) * 0.5:  
                overlap_count += 1  
                break
  
    return overlap_count / len(gold_steps)

  
def calculate_semantic_similarity(text1: str, text2: str) -> float:  
    """  
    Calculate semantic similarity between two texts using sequence matching.
      
    Args:  
        text1: First text string  
        text2: Second text string
          
    Returns:  
        Float score (0.0 to 1.0) representing similarity  
    """  
    try:  
        from difflib import SequenceMatcher  
        return SequenceMatcher(None, text1, text2).ratio()  
    except ImportError:  
        # Fallback to simple token overlap  
        tokens1 = set(text1.split())  
        tokens2 = set(text2.split())
  
        if not tokens1 and not tokens2:  
            return 1.0  
        if not tokens1 or not tokens2:  
            return 0.0
  
        overlap = len(tokens1 & tokens2)  
        total = len(tokens1 | tokens2)
  
        return overlap / total if total > 0 else 0.0

  
def get_ragtruth_failure_reason(final_accuracy: float, hallucination_rate: float, unsupported_count: int) -> str:  
    """  
    Generate human-readable failure reason for RAGTruth evaluation.
      
    Args:  
        final_accuracy: Accuracy score (0.0 to 1.0)  
        hallucination_rate: Proportion of unsupported claims (0.0 to 1.0)  
        unsupported_count: Number of unsupported claims
          
    Returns:  
        String describing the evaluation outcome or failure reason  
    """  
    if final_accuracy and hallucination_rate < 0.1:  
        return "Correct and well-grounded response"  
    elif final_accuracy:  
        return f"Correct answer but contains {unsupported_count} unsupported claims (hallucination rate: {hallucination_rate:.2f})"  
    elif not final_accuracy and hallucination_rate < 0.1:  
        return "Incorrect answer despite good grounding in evidence"  
    elif not final_accuracy and hallucination_rate >= 0.5:  
        return f"Incorrect answer with high hallucination rate ({hallucination_rate:.2f}) and {unsupported_count} unsupported claims"  
    else:  
        return f"Incorrect answer with moderate hallucination issues (rate: {hallucination_rate:.2f}, unsupported claims: {unsupported_count})"  