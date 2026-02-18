import re  
from typing import Dict, List, Tuple, Any  
from .llm_judge import llm_judge_eval  
import pandas as pd  
from .epistemic_faithfulness_metrics import *
  
def compute_helpfulness_scope(results_df, judge="gpt-4o-mini@openai", report_detailed=False, debug=False):  
    """  
    Evaluate helpfulness and scope of model responses by measuring coverage and overreach.
  
    Metrics:  
    - Coverage Score: Proportion of gold answer claims present in model output  
    - Overreach Rate: Proportion of model claims not supported by gold answer  
    - Evidence Utilization: How effectively the model uses available evidence
  
    This metric assesses whether the model provides complete answers (coverage) without  
    adding unsupported information (overreach).
  
    Args:  
        results_df: DataFrame with columns ['id', 'outputs', 'gold', 'ctx']  
        judge: Model ID for LLM judge evaluation  
        report_detailed: If True, include detailed results in output  
        debug: If True, print debug information during evaluation
  
    Returns:  
        Dictionary with coverage_score, overreach_rate, total_count, failure_reasons,  
        and optionally detailed_results per item  
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
  
            # Create coverage assessment prompts: check if gold claims appear in model output  
            gold_claims = extract_claims_from_response(gold_answer)  
            sample_data['num_gold_claims'] = len(gold_claims)
            for claim in gold_claims:  
                accuracy_prompts.append({  
                    'claim': claim,  
                    'context': model_output  
                })
  
            # Create overreach assessment prompts: check if model claims are supported by gold answer  
            model_claims = extract_claims_from_response(model_output)  
            sample_data['num_claims'] = len(model_claims)
  
            for claim in model_claims:  
                claim_support_prompts.append({  
                    'claim': claim,  
                    'context': gold_answer  
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
            print(f'Check coverage:\ngold_claims: {accuracy_df["claim"]}\nmodel_output: {accuracy_df["context"]}')  
            print(f'coverage_results: {accuracy_results}')  
        print()  
        if claim_support_prompts:  
            print(f'Check overreach:\nmodel_claims: {claim_support_df["claim"]}\ngold_answer: {claim_support_df["context"]}')  
            print(f'overreach_results: {claim_support_results}')
  
    # Second loop: process judge responses and compute metrics  
    correct_count = 0  
    total_count = len(results_df)  
    detailed_results = []  
    failure_reasons = []
  
    total_coverage = 0.0  
    total_overreach_rate = 0.0  
    total_evidence_utilization_score = 0.0  
    total_num_claims = 0.0
  
    for sample in sample_info:  
        try:  
            if 'error' in sample:  
                # Handle previous errors  
                failure_reasons.append({  
                    'id': sample['problem_id'],  
                    'reason': f"Processing error: {sample['error']}",  
                    'overreach_rate': 1.0,  
                    'unsupported_claims': []  
                })  
                continue
  
            # Calculate coverage: proportion of gold claims present in model output  
            coverage_score = 0.0  
            if accuracy_results and 'accuracy_prompt_idx' in sample and sample['accuracy_prompt_idx'] >= 0:
                start_idx = sample['accuracy_prompt_idx']
                # Number of gold claims for this sample, if tracked; default to 0 if missing
                num_gold_claims = sample.get('num_gold_claims', 0)
                # If there are no gold claims, we leave coverage_score at 0.0 and skip slicing
                if num_gold_claims > 0:
                    start_idx = sample['accuracy_prompt_idx']
                    # Clamp indices to valid range
                    if start_idx < len(accuracy_results):
                        end_idx = min(start_idx + num_gold_claims, len(accuracy_results))
                        per_sample_results = accuracy_results[start_idx:end_idx]
                        if per_sample_results:
                            coverage_parsed = [1 if acc == 'True' else 0 for acc in per_sample_results]
                            coverage_score = sum(coverage_parsed) / len(per_sample_results)
  
            # Get claim support results for overreach calculation  
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
  
            # Calculate overreach rate: proportion of model claims not in gold answer  
            overreach_rate = 0.0  
            if len(claim_evaluations) > 0:  
                supported_by_gold = sum(1 for eval in claim_evaluations if eval['is_supported'])  
                overreach_rate = 1.0 - (supported_by_gold / len(claim_evaluations))
  
            # Evaluate evidence utilization  
            evidence_utilization_score = evaluate_evidence_utilization(  
                sample['model_output'],  
                sample['context']  
            )
  
            # Determine failure reason  
            failure_reason = get_helpfulness_failure_reason(  
                coverage_score,  
                overreach_rate,  
                len(unsupported_claims)  
            )
  
            result = {  
                'id': sample['problem_id'],  
                'coverage_score': coverage_score,  
                'overreach_rate': overreach_rate,  
                'evidence_utilization_score': evidence_utilization_score,  
                'total_claims': len(claim_evaluations),  
                'supported_claims': len(claim_evaluations) - len(unsupported_claims),  
                'unsupported_claims': unsupported_claims,  
                'failure_reason': failure_reason,  
                'claim_evaluations': claim_evaluations  
            }
  
            detailed_results.append(result)
  
            # Update counters  
            if coverage_score > 0.2:  
                correct_count += 1  
            else:  
                failure_reasons.append({  
                    'id': sample['problem_id'],  
                    'reason': failure_reason,  
                    'overreach_rate': overreach_rate,  
                    'unsupported_claims': unsupported_claims  
                })
  
            # Accumulate scores  
            total_coverage += coverage_score  
            total_overreach_rate += overreach_rate  
            total_evidence_utilization_score += evidence_utilization_score  
            total_num_claims += len(claim_evaluations)
  
        except Exception as e:  
            failure_reasons.append({  
                'id': sample.get('problem_id', 'unknown'),  
                'reason': f"Processing error: {str(e)}",  
                'overreach_rate': 1.0,  
                'unsupported_claims': []  
            })
  
    # Calculate aggregate metrics  
    avg_coverage = total_coverage / total_count if total_count > 0 else 0.0  
    avg_overreach_rate = total_overreach_rate / total_count if total_count > 0 else 0.0
  
    return {  
        'coverage_score': avg_coverage,  
        'total_count': total_count,  
        'overreach_rate': avg_overreach_rate,  
        'failure_reasons': failure_reasons,  
        'detailed_results': detailed_results if report_detailed else None  
    }

  
def get_helpfulness_failure_reason(coverage_score: float, overreach_rate: float, unsupported_count: int) -> str:  
    """  
    Generate human-readable failure reason for helpfulness evaluation.
      
    Args:  
        coverage_score: Proportion of gold claims present in model output (0.0 to 1.0)  
        overreach_rate: Proportion of model claims not supported by gold answer (0.0 to 1.0)  
        unsupported_count: Number of unsupported claims in model output
          
    Returns:  
        String describing the evaluation outcome or failure reason  
    """  
    if coverage_score >= 0.8 and overreach_rate < 0.1:  
        return "Excellent: High coverage with minimal overreach"  
    elif coverage_score >= 0.5 and overreach_rate < 0.2:  
        return "Good: Reasonable coverage and low overreach"  
    elif coverage_score < 0.3:  
        return f"Low coverage ({coverage_score:.2f}): Model missed key information from gold answer"  
    elif overreach_rate >= 0.5:  
        return f"High overreach ({overreach_rate:.2f}): {unsupported_count} claims not in gold answer"  
    elif coverage_score < 0.5 and overreach_rate >= 0.3:  
        return f"Poor balance: Low coverage ({coverage_score:.2f}) and high overreach ({overreach_rate:.2f})"  
    else:  
        return f"Mixed quality: coverage={coverage_score:.2f}, overreach={overreach_rate:.2f}"  