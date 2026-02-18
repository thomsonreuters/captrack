import re  
from typing import Dict, List, Any, Optional, Tuple  
from difflib import SequenceMatcher
  
def compute_ruler_accuracy(results_df):  
    """  
    Evaluate RULER dataset performance: extracting hidden numbers from long context.
  
    Args:  
        results_df: DataFrame with RULER results containing 'outputs', 'gold', 'id' columns
  
    Returns:  
        Dictionary with RULER extraction metrics including exact/partial accuracy,  
        extraction patterns, response analysis, and quality indicators  
    """
  
    if len(results_df) == 0:  
        return get_empty_ruler_metrics()
  
    # Initialize counters  
    exact_matches = 0  
    partial_matches = 0  
    total_items = 0
  
    detailed_results = []  
    extraction_patterns = {  
        'exact_match': 0,  
        'embedded_correct': 0,  # Correct number found within response  
        'partial_match': 0,     # Similar but not exact  
        'no_number_found': 0,  
        'wrong_number': 0,  
        'multiple_numbers': 0  
    }
  
    for idx, row in results_df.iterrows():  
        try:  
            problem_id = row.get('id', idx)  
            model_output = row.get('outputs', '')  
            gold_number = extract_gold_number(row.get('gold', ''))
  
            if gold_number is None:  
                # Skip if no gold number available  
                continue
  
            # Extract numbers from model output  
            if isinstance(model_output, str):  
                extracted_numbers = extract_numbers_from_response(model_output)  
            else:  
                extracted_numbers = [model_output]

  
            # Evaluate extraction  
            result = evaluate_number_extraction(extracted_numbers, gold_number, model_output)
  
            # Update counters  
            if result['exact_match']:  
                exact_matches += 1  
                extraction_patterns['exact_match'] += 1  
            elif result['partial_match']:  
                partial_matches += 1  
                extraction_patterns['partial_match'] += 1  
            elif result['embedded_correct']:  
                exact_matches += 1  # Count as correct  
                extraction_patterns['embedded_correct'] += 1  
            elif result['no_numbers']:  
                extraction_patterns['no_number_found'] += 1  
            elif result['multiple_numbers']:  
                extraction_patterns['multiple_numbers'] += 1  
            else:  
                extraction_patterns['wrong_number'] += 1
  
            detailed_results.append({  
                'id': problem_id,  
                'gold_number': gold_number,  
                'extracted_numbers': extracted_numbers,  
                'model_output': str(model_output),  
                'exact_match': result['exact_match'],  
                'partial_match': result['partial_match'],  
                'embedded_correct': result['embedded_correct'],  
                'evaluation_details': result  
            })
  
            total_items += 1
  
        except KeyError as e:  
            detailed_results.append({  
                'id': row.get('id', idx),  
                'error': str(e),  
                'exact_match': False,  
                'partial_match': False  
            })  
            total_items += 1
  
    # Calculate metrics  
    exact_accuracy = exact_matches / total_items if total_items > 0 else 0.0  
    partial_accuracy = (exact_matches + partial_matches) / total_items if total_items > 0 else 0.0
  
    # Analyze response lengths and patterns  
    response_analysis = analyze_response_patterns(detailed_results)
  
    return {  
        # Primary metrics  
        'exact_accuracy': exact_accuracy,  
        'partial_accuracy': partial_accuracy,
  
        # Counts  
        'exact_matches': exact_matches,  
        'partial_matches': partial_matches,  
        'total_items': total_items,  
        'failed_extractions': total_items - exact_matches - partial_matches,
  
        # Pattern analysis  
        'extraction_patterns': extraction_patterns,  
        'pattern_rates': {k: v / total_items if total_items > 0 else 0.0  
                        for k, v in extraction_patterns.items()},
  
        # Response analysis  
        'response_analysis': response_analysis,
  
        # Quality indicators  
        'clean_extraction_rate': extraction_patterns['exact_match'] / total_items if total_items > 0 else 0.0,  
        'confusion_rate': extraction_patterns['multiple_numbers'] / total_items if total_items > 0 else 0.0,
  
        # Detailed results  
        #'detailed_results': detailed_results  
    }
  
def extract_gold_number(gold_data) -> Optional[str]:  
    """  
    Extract the gold number from various gold data formats.
  
    Args:  
        gold_data: Gold standard data (could be string, list, or dict)
  
    Returns:  
        Gold number as string, or None if not found  
    """  
    if not gold_data:  
        return None
  
    # Handle different gold data formats  
    if isinstance(gold_data, str):  
        # Direct string  
        return gold_data.strip()  
    elif isinstance(gold_data, list) and len(gold_data) > 0:  
        # List format - take first element  
        return str(gold_data[0]).strip()  
    elif isinstance(gold_data, dict):  
        # Dict format - look for common keys  
        for key in ['answer', 'number', 'gold', 'target']:  
            if key in gold_data:  
                return str(gold_data[key]).strip()
  
    # Try to convert to string  
    return str(gold_data).strip()
  
def extract_numbers_from_response(response: str) -> List[str]:  
    """  
    Extract all potential numbers from model response.
  
    Args:  
        response: Model's response text
  
    Returns:  
        List of extracted numbers as strings, sorted by length (longest first)  
    """  
    if not response:  
        return []
  
    # Pattern to match numbers (integers and decimals)  
    # This matches: integers, decimals, numbers with commas, scientific notation  
    number_patterns = [  
        r'\b\d{7,}\b',  # Long integers (7+ digits, likely the target)  
        #r'\b\d+\.\d+\b',  # Decimals  
        #r'\b\d{1,3}(?:,\d{3})*\b',  # Numbers with commas  
        #r'\b\d+(?:[eE][+-]?\d+)?\b',  # Scientific notation  
        #r'\b\d+\b'  # Any integers  
    ]
  
    all_numbers = []
  
    for pattern in number_patterns:  
        matches = re.findall(pattern, response)  
        for match in matches:  
            # Clean the number (remove commas, etc.)  
            cleaned = re.sub(r'[,\s]', '', match)  
            if cleaned and cleaned not in all_numbers:  
                all_numbers.append(cleaned)
  
    # Sort by length (longer numbers first, as they're more likely to be the target)  
    all_numbers.sort(key=len, reverse=True)
  
    return all_numbers
  
def evaluate_number_extraction(extracted_numbers: List[str], gold_number: str, full_response: str) -> Dict[str, Any]:  
    """  
    Evaluate the quality of number extraction.
  
    Args:  
        extracted_numbers: Numbers extracted from response  
        gold_number: Expected gold standard number  
        full_response: Full model response text
  
    Returns:  
        Dictionary with evaluation results including exact_match, partial_match,  
        embedded_correct flags, similarity scores, and closest match  
    """  
    result = {  
        'exact_match': False,  
        'partial_match': False,  
        'embedded_correct': False,  
        'no_numbers': len(extracted_numbers) == 0,  
        'multiple_numbers': len(extracted_numbers) > 1,  
        'closest_match': None,  
        'similarity_score': 0.0  
    }
  
    if not extracted_numbers:  
        return result
  
    gold_cleaned = clean_number(gold_number)
  
    # Check for exact matches  
    for num in extracted_numbers:  
        num_cleaned = clean_number(num)  
        if num_cleaned == gold_cleaned:  
            result['exact_match'] = True  
            result['closest_match'] = num  
            result['similarity_score'] = 1.0  
            return result
  
    # Check if the gold number appears anywhere in the response (embedded)  
    if gold_cleaned in str(full_response):  
        result['embedded_correct'] = True  
        result['closest_match'] = gold_number  
        result['similarity_score'] = 1.0  
        return result
  
    # Check for partial matches (similar numbers)  
    best_seq_match_ratio = 0.0  
    best_similarity = 0.0  
    best_match = None
  
    for num in extracted_numbers:  
        num_cleaned = clean_number(num)  
        similarity, seq_match_ratio = calculate_number_similarity(num_cleaned, gold_cleaned)
  
        if seq_match_ratio > best_seq_match_ratio:  
            best_seq_match_ratio = seq_match_ratio  
            best_similarity = similarity  
            best_match = num
  
    result['seq_match_score'] = best_seq_match_ratio  
    result['similarity_score'] = best_similarity  
    result['closest_match'] = best_match
  
    # Consider it a partial match if similarity is high enough  
    if best_similarity >= 0.7:  # 70% similarity threshold  
        result['partial_match'] = True
  
    return result
  
def clean_number(number_str: str) -> str:  
    """  
    Clean and normalize a number string.
      
    Args:  
        number_str: Number string to clean
          
    Returns:  
        Cleaned number string with formatting removed and leading zeros stripped  
    """  
    if not number_str:  
        return ""
  
    # Remove whitespace, commas, and other formatting  
    cleaned = re.sub(r'[,\s\-_]', '', str(number_str))
  
    # Remove leading zeros (except for single zero)  
    if len(cleaned) > 1:  
        cleaned = cleaned.lstrip('0') or '0'
  
    return cleaned
  
def calculate_number_similarity(num1: str, num2: str) -> Tuple[float, float]:  
    """  
    Calculate similarity between two number strings.
  
    Args:  
        num1: First number string to compare  
        num2: Second number string to compare
  
    Returns:  
        Tuple of (combined_similarity, sequence_matching_ratio), both between 0.0 and 1.0  
    """  
    if not num1 or not num2:  
        return 0.0, 0.0
  
    if num1 == num2:  
        return 1.0, 1.0
  
    # Character-level similarity (Jaccard similarity)  
    set1 = set(num1)  
    set2 = set(num2)
  
    intersection = len(set1.intersection(set2))  
    union = len(set1.union(set2))
  
    jaccard_sim = intersection / union if union > 0 else 0.0
  
    # Length similarity  
    len_sim = 1.0 - abs(len(num1) - len(num2)) / max(len(num1), len(num2))
  
    # Longest common subsequence similarity  
    lcs_length = longest_common_subsequence_length(num1, num2)  
    lcs_sim = lcs_length / max(len(num1), len(num2))
  
    # Combined similarity (weighted average)  
    combined_sim = 0.4 * jaccard_sim + 0.3 * len_sim + 0.3 * lcs_sim  
    sequence_matching_ratio = SequenceMatcher(None, num1, num2).ratio()  
    return combined_sim, sequence_matching_ratio
  
def longest_common_subsequence_length(s1: str, s2: str) -> int:  
    """  
    Calculate the length of the longest common subsequence between two strings.
      
    Args:  
        s1: First string  
        s2: Second string
          
    Returns:  
        Length of the longest common subsequence  
    """  
    m, n = len(s1), len(s2)
  
    # Create DP table  
    dp = [[0] * (n + 1) for _ in range(m + 1)]
  
    # Fill the DP table  
    for i in range(1, m + 1):  
        for j in range(1, n + 1):  
            if s1[i-1] == s2[j-1]:  
                dp[i][j] = dp[i-1][j-1] + 1  
            else:  
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
  
    return dp[m][n]
  
def analyze_response_patterns(detailed_results: List[Dict]) -> Dict[str, Any]:  
    """  
    Analyze patterns in model responses.
  
    Args:  
        detailed_results: List of detailed evaluation results from compute_ruler_accuracy
  
    Returns:  
        Dictionary with response pattern analysis including average response length,  
        numbers extracted, success/failure counts, and failure pattern breakdown  
    """  
    if not detailed_results:  
        return {}
  
    response_lengths = []  
    number_counts = []  
    successful_responses = []  
    failed_responses = []
  
    for result in detailed_results:  
        if 'error' in result:  
            continue
  
        response_text = result.get('model_output', '')  
        response_lengths.append(len(response_text.split()))
  
        extracted_nums = result.get('extracted_numbers', [])  
        number_counts.append(len(extracted_nums))
  
        if result.get('exact_match') or result.get('embedded_correct'):  
            successful_responses.append(result)  
        else:  
            failed_responses.append(result)
  
    analysis = {  
        'avg_response_length': sum(response_lengths) / len(response_lengths) if response_lengths else 0,  
        'avg_numbers_extracted': sum(number_counts) / len(number_counts) if number_counts else 0,  
        'successful_response_count': len(successful_responses),  
        'failed_response_count': len(failed_responses)  
    }
  
    # Analyze common failure patterns  
    if failed_responses:  
        failure_patterns = {}  
        for result in failed_responses:  
            extracted = result.get('extracted_numbers', [])  
            if not extracted:  
                failure_patterns['no_extraction'] = failure_patterns.get('no_extraction', 0) + 1  
            elif len(extracted) > 3:  
                failure_patterns['too_many_numbers'] = failure_patterns.get('too_many_numbers', 0) + 1  
            else:  
                failure_patterns['wrong_extraction'] = failure_patterns.get('wrong_extraction', 0) + 1
  
        analysis['failure_patterns'] = failure_patterns
  
    return analysis
  
def get_empty_ruler_metrics() -> Dict[str, Any]:  
    """  
    Return empty metrics for edge cases (e.g., empty DataFrame).
      
    Returns:  
        Dictionary with all RULER metrics set to zero/empty values  
    """  
    return {  
        'exact_accuracy': 0.0,  
        'partial_accuracy': 0.0,  
        'exact_matches': 0,  
        'partial_matches': 0,  
        'total_items': 0,  
        'failed_extractions': 0,  
        'extraction_patterns': {  
            'exact_match': 0,  
            'embedded_correct': 0,  
            'partial_match': 0,  
            'no_number_found': 0,  
            'wrong_number': 0,  
            'multiple_numbers': 0  
        },  
        'pattern_rates': {},  
        'response_analysis': {},  
        'clean_extraction_rate': 0.0,  
        'confusion_rate': 0.0,  
        'detailed_results': []  
    }  