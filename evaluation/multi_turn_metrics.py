"""  
Multi-turn conversation quality evaluation metrics.
  
This module evaluates how well AI models handle multi-turn conversations by assessing  
the quality of follow-up responses. Uses LLM-as-a-judge to rate conversation coherence,  
context retention, and response quality across turns.  
"""
  
import re  
import numpy as np  
from typing import Dict, List, Any, Optional  
from .llm_judge import llm_judge_eval  
import pandas as pd
  
def compute_multi_turn_metrics(  
    results_df: pd.DataFrame,  
    judge: str = "gpt-4o-mini@openai",  
    report_detailed: bool = False,  
    verbose: bool = False  
) -> Dict[str, Any]:  
    """  
    Evaluate multi-turn conversation quality using LLM-as-a-judge.
  
    This function evaluates how well the model handles follow-up questions in a  
    multi-turn conversation context, focusing on the quality of the second turn response.  
    The judge evaluates context retention, coherence, and response quality.
  
    Args:  
        results_df: DataFrame with columns:  
            - 'prompt': List containing [question_1, question_2]  
            - 'turn1_response': Model's response to first question  
            - 'outputs': Model's response to second question  
            - 'id': Optional identifier for each conversation  
        judge: Model ID for LLM judge (default: Claude 3.5 Sonnet)  
        report_detailed: Whether to include detailed per-sample results  
        verbose: Whether to print progress information
  
    Returns:  
        Dictionary with multi-turn performance metrics including:  
        - rating_stats: Mean, median, std, min, max ratings  
        - rating_distribution: Count of ratings 1-10  
        - quality_categories: Breakdown by excellent/good/acceptable/poor  
        - summary: Total samples, valid ratings, success rate, pass rate  
        - detailed_results: Per-sample ratings and explanations (if report_detailed=True)  
    """
  
    if len(results_df) == 0:  
        return get_empty_metrics()
  
    # Prepare judge prompts from conversation data  
    judge_df = prepare_judge_dataframe(results_df)
  
    if judge_df is None or len(judge_df) == 0:  
        print("Failed to prepare judge prompts")  
        return get_empty_metrics()
  
    if verbose:  
        print(f"Prepared {len(judge_df)} judge prompts for evaluation")
  
    # Call LLM judge using the existing infrastructure  
    judge_responses = llm_judge_eval(  
        judge_df,  
        mode="multi_turn_rating",  
        model_id=judge,  
        max_tokens=2048,  
        temperature=0.0,  
        verbose=verbose  
    )
  
    # Process results and compute metrics  
    return process_judge_responses(  
        results_df=results_df,  
        judge_responses=judge_responses,  
        report_detailed=report_detailed,  
        verbose=verbose  
    )

  
def prepare_judge_dataframe(results_df: pd.DataFrame) -> Optional[pd.DataFrame]:  
    """  
    Prepare dataframe with judge prompts for multi-turn evaluation.
      
    Extracts conversation components (questions and answers from both turns)  
    and formats them for LLM judge evaluation.
  
    Args:  
        results_df: DataFrame with conversation data including 'prompt',   
                   'turn1_response', and 'outputs' columns
  
    Returns:  
        DataFrame with columns 'question_1', 'answer_1', 'question_2', 'answer_2', 'id'  
        formatted for llm_judge_eval, or None if no valid data could be prepared  
    """
  
    judge_data = []
  
    for idx, row in results_df.iterrows():  
        try:  
            # Extract conversation components  
            prompts = row['prompt']  
            if not isinstance(prompts, list) or len(prompts) < 2:  
                print(f"Warning: Invalid prompt format at index {idx}, skipping")  
                question_1 = ''
                question_2 = ''
            else:
                # Handle both dict and string prompt formats  
                question_1 = prompts[0]['user prompt'] if isinstance(prompts[0], dict) else prompts[0]  
                question_2 = prompts[1]['user prompt'] if isinstance(prompts[1], dict) else prompts[1]  

            answer_1 = row.get('turn1_response', '')  
            answer_2 = row.get('outputs', '')
  
            # Store data in format expected by llm_judge  
            judge_data.append({  
                'question_1': question_1,  
                'answer_1': answer_1,  
                'question_2': question_2,  
                'answer_2': answer_2,  
                'id': row.get('id', idx)  
            })
  
        except Exception as e:  
            print(f"Error preparing judge prompt for row {idx}: {e}")  
            judge_data.append({  
                'question_1': "",  
                'answer_1': "",  
                'question_2': "",  
                'answer_2': "",  
                'id': row.get('id', idx)  
            })
  
    if not judge_data:  
        return None
  
    return pd.DataFrame(judge_data)

  
def process_judge_responses(  
    results_df: pd.DataFrame,  
    judge_responses: List[Optional[int]],  
    report_detailed: bool = False,  
    verbose: bool = False  
) -> Dict[str, Any]:  
    """  
    Process judge responses and compute metrics.
      
    Validates ratings from the judge, filters invalid responses, and computes  
    aggregate statistics and quality distributions.
  
    Args:  
        results_df: Original results dataframe with conversation data  
        judge_responses: List of ratings from judge (1-10 scale, can contain None for failures)  
        report_detailed: Whether to include detailed per-sample results  
        verbose: Whether to print warnings for invalid ratings
  
    Returns:  
        Dictionary containing:  
        - rating_stats: Statistical measures of ratings  
        - rating_distribution: Histogram of ratings 1-10  
        - quality_categories: Counts and percentages for quality tiers  
        - summary: Overview including success rate and pass rate  
        - detailed_results: Per-sample data (if report_detailed=True)  
    """
  
    ratings = []  
    detailed_results = []
  
    for idx, (row_tuple, rating) in enumerate(zip(results_df.iterrows(), judge_responses)):  
        row_idx, row_data = row_tuple
  
        try:  
            prompts = row_data['prompt']  
            question_1 = prompts[0] if isinstance(prompts, list) and len(prompts) > 0 else ''  
            question_2 = prompts[1] if isinstance(prompts, list) and len(prompts) > 1 else ''
  
            # Validate rating is in valid range (1-10)  
            if rating is not None and isinstance(rating, (int, float)) and 1 <= rating <= 10:  
                ratings.append(int(rating))  
            else:  
                if verbose:  
                    print(f"Warning: Invalid rating for sample {row_data.get('id', idx)}: {rating}")  
                ratings.append(None)
  
            # Store detailed results for analysis  
            detailed_results.append({  
                'id': row_data.get('id', idx),  
                'question_1': question_1,  
                'question_2': question_2,  
                'turn1_response': row_data.get('turn1_response', ''),  
                'turn2_response': row_data.get('outputs', ''),  
                'rating': rating if rating is not None else None  
            })
  
        except Exception as e:  
            if verbose:  
                print(f"Error processing sample {row_data.get('id', idx)}: {e}")  
            ratings.append(None)  
            detailed_results.append({  
                'id': row_data.get('id', idx),  
                'error': str(e)  
            })
  
    # Filter out None ratings for metric computation  
    valid_ratings = [r for r in ratings if r is not None]
  
    if len(valid_ratings) == 0:  
        print("Warning: No valid ratings extracted from judge responses")  
        return get_empty_metrics()
  
    # Compute aggregate metrics  
    metrics = compute_rating_metrics(valid_ratings, len(results_df))
  
    # Add detailed results if requested  
    if report_detailed:  
        metrics['detailed_results'] = detailed_results
  
    return metrics

  
def compute_rating_metrics(ratings: List[int], total_samples: int) -> Dict[str, Any]:  
    """  
    Compute aggregate metrics from ratings.
      
    Calculates statistical measures, rating distributions, and quality category  
    breakdowns following MT-Bench evaluation conventions.
  
    Args:  
        ratings: List of valid ratings on 1-10 scale  
        total_samples: Total number of samples attempted (including failed ones)
  
    Returns:  
        Dictionary containing:  
        - rating_stats: Dict with mean, median, std, min, max ratings  
        - rating_distribution: Dict mapping each rating (1-10) to its count  
        - quality_categories: Dict with excellent (9-10), good (7-8),   
          acceptable (5-6), and poor (1-4) breakdowns  
        - summary: Dict with total_samples, valid_ratings, failed_ratings,  
          success_rate, and pass_rate (percentage rated 7+)  
    """
  
    from collections import Counter
  
    if not ratings:  
        return get_empty_metrics()
  
    # Basic statistics  
    mean_rating = np.mean(ratings)  
    median_rating = np.median(ratings)  
    std_rating = np.std(ratings)  
    min_rating = int(min(ratings))  
    max_rating = int(max(ratings))
  
    # Rating distribution (count for each rating 1-10)  
    rating_counter = Counter(ratings)  
    rating_distribution = {i: rating_counter.get(i, 0) for i in range(1, 11)}
  
    # Quality categories (following MT-Bench conventions)  
    excellent_count = sum(1 for r in ratings if r >= 9)  # 9-10: Outstanding  
    good_count = sum(1 for r in ratings if 7 <= r < 9)  # 7-8: Good  
    acceptable_count = sum(1 for r in ratings if 5 <= r < 7)  # 5-6: Acceptable  
    poor_count = sum(1 for r in ratings if r < 5)  # 1-4: Poor
  
    valid_count = len(ratings)  
    failed_count = total_samples - valid_count
  
    return {  
        # Basic statistics  
        'rating_stats': {  
            'mean_rating': float(mean_rating),  
            'median_rating': float(median_rating),  
            'std_rating': float(std_rating),  
            'min_rating': min_rating,  
            'max_rating': max_rating  
        },
  
        # Distribution  
        'rating_distribution': rating_distribution,
  
        # Quality categories  
        'quality_categories': {  
            'excellent': {  
                'count': excellent_count,  
                'percentage': excellent_count / valid_count if valid_count > 0 else 0.0,  
                'description': 'Ratings 9-10: Outstanding multi-turn handling'  
            },  
            'good': {  
                'count': good_count,  
                'percentage': good_count / valid_count if valid_count > 0 else 0.0,  
                'description': 'Ratings 7-8: Good multi-turn handling'  
            },  
            'acceptable': {  
                'count': acceptable_count,  
                'percentage': acceptable_count / valid_count if valid_count > 0 else 0.0,  
                'description': 'Ratings 5-6: Acceptable multi-turn handling'  
            },  
            'poor': {  
                'count': poor_count,  
                'percentage': poor_count / valid_count if valid_count > 0 else 0.0,  
                'description': 'Ratings 1-4: Poor multi-turn handling'  
            }  
        },
  
        # Summary  
        'summary': {  
            'total_samples': total_samples,  
            'valid_ratings': valid_count,  
            'failed_ratings': failed_count,  
            'success_rate': valid_count / total_samples if total_samples > 0 else 0.0,  
            'pass_rate': (excellent_count + good_count) / valid_count if valid_count > 0 else 0.0  # 7+ is passing  
        },
  
        'detailed_results': None  # Will be populated if report_detailed=True  
    }

  
def get_empty_metrics() -> Dict[str, Any]:  
    """  
    Return empty metrics structure for edge cases.
      
    Provides a consistent metrics structure when no valid data is available,  
    such as when the input dataframe is empty or all evaluations fail.
      
    Returns:  
        Dictionary with same structure as compute_rating_metrics() but with  
        all counts set to 0 and all rates/percentages set to 0.0  
    """  
    return {  
        'rating_stats': {  
            'mean_rating': 0.0,  
            'median_rating': 0.0,  
            'std_rating': 0.0,  
            'min_rating': 0,  
            'max_rating': 0  
        },
  
        'rating_distribution': {i: 0 for i in range(1, 11)},
  
        'quality_categories': {  
            'excellent': {'count': 0, 'percentage': 0.0, 'description': 'Ratings 9-10'},  
            'good': {'count': 0, 'percentage': 0.0, 'description': 'Ratings 7-8'},  
            'acceptable': {'count': 0, 'percentage': 0.0, 'description': 'Ratings 5-6'},  
            'poor': {'count': 0, 'percentage': 0.0, 'description': 'Ratings 1-4'}  
        },
  
        'summary': {  
            'total_samples': 0,  
            'valid_ratings': 0,  
            'failed_ratings': 0,  
            'success_rate': 0.0,  
            'pass_rate': 0.0  
        },
  
        'detailed_results': []  
    }  