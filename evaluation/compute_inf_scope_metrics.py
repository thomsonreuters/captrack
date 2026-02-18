import re  
from typing import Dict, List, Tuple, Any  
from .llm_judge import llm_judge_eval  
import pandas as pd
  
def compute_inf_scope_metrics(results_df, judge="gpt-4o-mini@openai", has_context=True, report_detailed=False, debug=False):  
    """  
    Evaluate model responses for coverage, hallucination, and overreach.
  
    Metrics:  
    - Coverage: Whether gold answer claims appear in model output (same as accuracy)  
    - Hallucination Rate: Proportion of model claims unsupported by context  
    - Overreach Rate: Proportion of model claims not supported by gold answer  
    - Evidence Utilization: How effectively the model uses available evidence
  
    Args:  
        results_df: DataFrame with columns ['id', 'outputs', 'gold', 'ctx']  
        judge: Model ID for LLM judge  
        has_context: If True, evaluate hallucination against context; if False, skip hallucination checks  
        report_detailed: Whether to include detailed results  
        debug: Whether to print debug information
  
    Returns:  
        Dictionary with coverage, hallucination, overreach, and evidence utilization metrics  
    """
  
    # First loop: collect all judge prompts  
    coverage_prompts = []  # Check if gold claims are in model output  
    hallucination_prompts = []  # Check if model claims are supported by context  
    overreach_prompts = []  # Check if model claims are supported by gold answer  
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
  
            # Extract claims  
            gold_claims = extract_claims_from_response(gold_answer)  
            model_claims = extract_claims_from_response(model_output)
  
            # Store sample info for later processing  
            sample_data = {  
                'idx': idx,  
                'problem_id': problem_id,  
                'model_output': model_output,  
                'gold_answer': gold_answer,  
                'context': context,  
                'coverage_prompt_start_idx': len(coverage_prompts),  
                'num_gold_claims': len(gold_claims),  
                'hallucination_prompt_start_idx': len(hallucination_prompts),  
                'overreach_prompt_start_idx': len(overreach_prompts),  
                'num_model_claims': len(model_claims)  
            }
  
            # Create coverage prompts: check if gold claims appear in model output  
            for claim in gold_claims:  
                coverage_prompts.append({  
                    'claim': claim,  
                    'context': model_output  
                })
  
            if has_context:  
                # Create hallucination prompts: check if model claims are supported by context  
                for claim in model_claims:  
                    hallucination_prompts.append({  
                        'claim': claim,  
                        'context': context  
                    })
  
            # Create overreach prompts: check if model claims are supported by gold answer  
            for claim in model_claims:  
                overreach_prompts.append({  
                    'claim': claim,  
                    'context': gold_answer  
                })
  
            sample_info.append(sample_data)
  
        except Exception as e:  
            raise e
  
    # Call LLM judges for all three tasks  
    coverage_results = []  
    hallucination_results = []  
    overreach_results = []
  
    if coverage_prompts:  
        coverage_df = pd.DataFrame(coverage_prompts)  
        coverage_results = llm_judge_eval(coverage_df, "claim_support", judge)
  
    if hallucination_prompts:  
        hallucination_df = pd.DataFrame(hallucination_prompts)  
        hallucination_results = llm_judge_eval(hallucination_df, "claim_support", judge)
  
    if overreach_prompts:  
        overreach_df = pd.DataFrame(overreach_prompts)  
        overreach_results = llm_judge_eval(overreach_df, "claim_support", judge)
  
    if debug:  
        if coverage_prompts:  
            cov_vals = "\n- ".join(coverage_df["claim"].values)    
            print(f'Check coverage:\ngold_claims:\n{cov_vals}\nmodel_output:\n{coverage_df["context"].values}')  
            print(f'coverage_results: {coverage_results}')  
        print()  
        if hallucination_prompts:  
            print(f'Check hallucination:\nmodel_claims: {hallucination_df["claim"].values}\ncontext:\n{hallucination_df["context"].values}')  
            print(f'hallucination_results: {hallucination_results}')  
        print()  
        if overreach_prompts:  
            ovr_vals = "\n- ".join(overreach_df["claim"].values)  
            print(f'Check overreach:\nmodel_claims:\n{ovr_vals}\ngold_answer:\n{overreach_df["context"].values}')  
            print(f'overreach_results: {overreach_results}')
  
    # Second loop: process judge responses and compute metrics  
    correct_count = 0  
    total_count = len(results_df)  
    detailed_results = []  
    failure_reasons = []
  
    total_coverage = 0.0  
    total_hallucination_rate = 0.0  
    total_overreach_rate = 0.0  
    total_evidence_utilization = 0.0
  
    for sample in sample_info:  
        try:  
            # Get coverage score: proportion of gold claims found in model output  
            coverage_score = 0.0  
            if sample['num_gold_claims'] > 0:  
                coverage_start = sample['coverage_prompt_start_idx']  
                coverage_end = coverage_start + sample['num_gold_claims']  
                coverage_parsed = [1 if result == 'True' else 0  
                                 for result in coverage_results[coverage_start:coverage_end]]  
                coverage_score = sum(coverage_parsed) / len(coverage_parsed) if coverage_parsed else 0.0
  
            if has_context:  
                # Get hallucination evaluations: which model claims are unsupported by context  
                hallucination_evaluations = []  
                unsupported_by_context = []
      
                if sample['num_model_claims'] > 0:  
                    model_claims = extract_claims_from_response(sample['model_output'])  
                    hallucination_start = sample['hallucination_prompt_start_idx']
      
                    for i, claim in enumerate(model_claims):  
                        claim_idx = hallucination_start + i  
                        is_supported = False
      
                        if claim_idx < len(hallucination_results):  
                            support_response = hallucination_results[claim_idx]  
                            is_supported = (support_response == "True")
      
                        hallucination_evaluations.append({  
                            'claim': claim,  
                            'is_supported_by_context': is_supported  
                        })
      
                        if not is_supported:  
                            unsupported_by_context.append(claim)
      
                # Calculate hallucination rate  
                hallucination_rate = 0.0  
                if len(hallucination_evaluations) > 0:  
                    supported_by_context = sum(1 for eval in hallucination_evaluations  
                                              if eval['is_supported_by_context'])  
                    hallucination_rate = 1.0 - (supported_by_context / len(hallucination_evaluations))  
            else:  
                hallucination_rate = 0.0  
                unsupported_by_context = []  
                supported_by_context = 0.0  
                hallucination_evaluations = None
  
            # Get overreach evaluations: which model claims are not supported by gold answer  
            overreach_evaluations = []  
            unsupported_by_gold = []
  
            if sample['num_model_claims'] > 0:  
                model_claims = extract_claims_from_response(sample['model_output'])  
                overreach_start = sample['overreach_prompt_start_idx']
  
                for i, claim in enumerate(model_claims):  
                    claim_idx = overreach_start + i  
                    is_supported = False
  
                    if claim_idx < len(overreach_results):  
                        support_response = overreach_results[claim_idx]  
                        is_supported = (support_response == "True")
  
                    overreach_evaluations.append({  
                        'claim': claim,  
                        'is_supported_by_gold': is_supported  
                    })
  
                    if not is_supported:  
                        unsupported_by_gold.append(claim)
  
            # Calculate overreach rate  
            overreach_rate = 0.0  
            if len(overreach_evaluations) > 0:  
                supported_by_gold = sum(1 for eval in overreach_evaluations  
                                       if eval['is_supported_by_gold'])  
                overreach_rate = 1.0 - (supported_by_gold / len(overreach_evaluations))
  
            # Evaluate evidence utilization  
            if has_context:  
                evidence_utilization_score = evaluate_evidence_utilization(  
                    sample['model_output'],  
                    sample['context']  
                )  
            else:  
                evidence_utilization_score = 0.0
  
            # Determine failure reason  
            failure_reason = get_inf_scope_failure_reason(  
                coverage_score,  
                hallucination_rate,  
                overreach_rate,  
                len(unsupported_by_context),  
                len(unsupported_by_gold)  
            )
  
            # Create detailed result  
            result = {  
                'id': sample['problem_id'],  
                'coverage_score': coverage_score,  
                'hallucination_rate': hallucination_rate,  
                'overreach_rate': overreach_rate,  
                'evidence_utilization_score': evidence_utilization_score,  
                'total_model_claims': len(overreach_evaluations),  
                'total_gold_claims': sample['num_gold_claims'],  
                'unsupported_by_context': unsupported_by_context,  
                'unsupported_by_gold': unsupported_by_gold,  
                'failure_reason': failure_reason,  
                'hallucination_evaluations': hallucination_evaluations,  
                'overreach_evaluations': overreach_evaluations  
            }
  
            detailed_results.append(result)
  
            # Update counters  
            if coverage_score > 0.2:  
                correct_count += 1  
            else:  
                failure_reasons.append({  
                    'id': sample['problem_id'],  
                    'reason': failure_reason,  
                    'coverage_score': coverage_score,  
                    'hallucination_rate': hallucination_rate,  
                    'overreach_rate': overreach_rate,  
                    'unsupported_by_context': unsupported_by_context,  
                    'unsupported_by_gold': unsupported_by_gold  
                })
  
            # Accumulate scores  
            total_coverage += coverage_score  
            total_hallucination_rate += hallucination_rate  
            total_overreach_rate += overreach_rate  
            total_evidence_utilization += evidence_utilization_score
  
        except KeyError as e:  
            failure_reasons.append({  
                'id': sample.get('problem_id', 'unknown'),  
                'reason': f"Processing error: {str(e)}",  
                'coverage_score': 0.0,  
                'hallucination_rate': 1.0,  
                'overreach_rate': 1.0,  
                'unsupported_by_context': [],  
                'unsupported_by_gold': []  
            })
  
    # Calculate aggregate metrics  
    avg_coverage = total_coverage / total_count if total_count > 0 else 0.0  
    avg_hallucination_rate = total_hallucination_rate / total_count if total_count > 0 else 0.0  
    avg_overreach_rate = total_overreach_rate / total_count if total_count > 0 else 0.0  
    avg_evidence_utilization = total_evidence_utilization / total_count if total_count > 0 else 0.0
  
    return {  
        'coverage_score': avg_coverage,  
        'correct_count': correct_count,  
        'total_count': total_count,  
        'hallucination_rate': avg_hallucination_rate if has_context else None,  
        'faithfulness_score': 1 - avg_hallucination_rate if has_context else None,  
        'overreach_rate': avg_overreach_rate, 
        'overreach_score': 1 - avg_overreach_rate,  
        'evidence_utilization_score': avg_evidence_utilization,  
        'failure_reasons': failure_reasons,  
        'detailed_results': detailed_results if report_detailed else None  
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
        List of sentence strings (cleaned and filtered)  
    """  
    import re
  
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
    import re
  
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

  
def evaluate_evidence_utilization(response: str, context: str) -> float:  
    """  
    Evaluate how well the response utilizes available evidence from context.
      
    Args:  
        response: Model's response text  
        context: Context/evidence text available to the model
          
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

  
def get_inf_scope_failure_reason(  
    coverage_score: float,  
    hallucination_rate: float,  
    overreach_rate: float,  
    unsupported_context_count: int,  
    unsupported_gold_count: int  
) -> str:  
    """  
    Generate human-readable failure reason for informational scope evaluation.
      
    Args:  
        coverage_score: Proportion of gold claims present in model output (0.0 to 1.0)  
        hallucination_rate: Proportion of model claims unsupported by context (0.0 to 1.0)  
        overreach_rate: Proportion of model claims not in gold answer (0.0 to 1.0)  
        unsupported_context_count: Number of claims unsupported by context  
        unsupported_gold_count: Number of claims not in gold answer
          
    Returns:  
        String describing the evaluation outcome or failure reason  
    """  
    if coverage_score > 0.8 and hallucination_rate < 0.1 and overreach_rate < 0.1:  
        return "Excellent response: high coverage, well-grounded, and aligned with gold answer"  
    elif coverage_score > 0.5 and hallucination_rate < 0.2 and overreach_rate < 0.2:  
        return "Good response with minor issues"  
    elif coverage_score < 0.3:  
        return f"Low coverage ({coverage_score:.2f}): model missed key information from gold answer"  
    elif hallucination_rate >= 0.5:  
        return f"High hallucination rate ({hallucination_rate:.2f}): {unsupported_context_count} claims unsupported by context"  
    elif overreach_rate >= 0.5:  
        return f"High overreach rate ({overreach_rate:.2f}): {unsupported_gold_count} claims not in gold answer"  
    else:  
        return f"Mixed quality: coverage={coverage_score:.2f}, hallucination={hallucination_rate:.2f}, overreach={overreach_rate:.2f}"  