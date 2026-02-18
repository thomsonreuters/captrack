"""  
Style and elaboration quality metrics for conversational AI responses.
  
This module evaluates stylistic aspects of AI model responses including verbosity,  
hedging behavior, directness, and formatting choices. Primarily used for MT-Bench  
and similar conversational evaluation benchmarks to assess response quality beyond  
just correctness.  
"""
  
import re  
import numpy as np  
from typing import Dict, List, Any, Tuple  
from collections import Counter  
import pandas as pd

  
def compute_style_elaboration_metrics(  
    results_df: pd.DataFrame,  
    report_detailed: bool = False  
) -> Dict[str, Any]:  
    """  
    Compute style and elaboration metrics for MT-Bench responses.
  
    Evaluates multiple dimensions of response style:  
    1. Verbosity: Length metrics including mean, percentiles, and sentence-level analysis  
    2. Hedging Rate: Frequency of uncertainty expressions and AI disclaimers  
    3. Directness: Whether responses provide immediate, clear answers  
    4. Formatting Style: Use of structured formatting (bullets, tables, emojis)
  
    Args:  
        results_df: DataFrame with columns:  
            - 'outputs': Model responses (strings)  
            - 'id': Optional problem identifier  
            - 'prompt': Optional prompt text for directness assessment  
        report_detailed: If True, include per-sample detailed results
  
    Returns:  
        Dictionary containing:  
        - verbosity: Dict with mean_answer_length, percentile_90_length,   
          avg_sentence_length, median_sentence_length, min/max/std metrics  
        - hedging: Dict with hedging_rate, hedging_count, most_common_hedges  
        - directness: Dict with directness_rate, direct_count, indirect_count  
        - formatting: Dict with bullet/table/emoji usage rates and counts  
        - summary: Dict with total_responses and quality indicators  
        - detailed_results: Per-sample metrics (if report_detailed=True)  
    """
  
    if len(results_df) == 0:  
        return get_empty_metrics()
  
    # Initialize counters  
    answer_lengths = []  
    sentence_lengths = []  
    hedging_count = 0  
    direct_count = 0  
    bullet_count = 0  
    table_count = 0  
    emoji_count = 0
  
    detailed_results = []
  
    for idx, row in results_df.iterrows():  
        try:  
            output = row.get("outputs", "").strip()  
            problem_id = row.get("id", idx)  
            prompt = row.get("prompt", "")
  
            if not output:  
                continue
  
            # 1. Verbosity metrics  
            token_count = count_tokens(output)  
            answer_lengths.append(token_count)
  
            # Extract sentence-level metrics  
            sentences = extract_sentences(output)  
            current_sentence_lengths = [count_tokens(sentence) for sentence in sentences if sentence.strip()]  
            sentence_lengths.extend(current_sentence_lengths)
  
            # 2. Hedging rate  
            has_hedging = detect_hedging(output)  
            if has_hedging:  
                hedging_count += 1
  
            # 3. Directness  
            is_direct = assess_directness(output, prompt)  
            if is_direct:  
                direct_count += 1
  
            # 4. Formatting style  
            has_bullets = detect_bullets(output)  
            has_table = detect_table(output)  
            has_emoji = detect_emoji(output)
  
            if has_bullets:  
                bullet_count += 1  
            if has_table:  
                table_count += 1  
            if has_emoji:  
                emoji_count += 1
  
            # Store detailed results  
            detailed_results.append({  
                "id": problem_id,  
                "token_count": token_count,  
                "sentence_count": len(current_sentence_lengths),  
                "avg_sentence_length": np.mean(current_sentence_lengths) if current_sentence_lengths else 0.0,  
                "has_hedging": has_hedging,  
                "is_direct": is_direct,  
                "has_bullets": has_bullets,  
                "has_table": has_table,  
                "has_emoji": has_emoji,  
                "hedging_phrases": extract_hedging_phrases(output) if has_hedging else []  
            })
  
        except Exception as e:  
            # Handle errors gracefully  
            detailed_results.append({  
                "id": row.get("id", idx),  
                "error": str(e)  
            })
  
    # Calculate metrics  
    total_responses = len([r for r in detailed_results if "error" not in r])
  
    if total_responses == 0:  
        return get_empty_metrics()
  
    # Verbosity metrics  
    mean_length = np.mean(answer_lengths) if answer_lengths else 0.0  
    percentile_90_length = np.percentile(answer_lengths, 90) if answer_lengths else 0.0
  
    # Average sentence length metrics  
    avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0.0  
    median_sentence_length = np.median(sentence_lengths) if sentence_lengths else 0.0
  
    # Rate metrics  
    hedging_rate = hedging_count / total_responses  
    directness_rate = direct_count / total_responses  
    bullet_usage_rate = bullet_count / total_responses  
    table_usage_rate = table_count / total_responses  
    emoji_usage_rate = emoji_count / total_responses
  
    return {  
        # Verbosity metrics  
        "verbosity": {  
            "mean_answer_length": mean_length,  
            "percentile_90_length": percentile_90_length,  
            "min_length": min(answer_lengths) if answer_lengths else 0,  
            "max_length": max(answer_lengths) if answer_lengths else 0,  
            "std_length": np.std(answer_lengths) if answer_lengths else 0.0,  
            # Sentence length metrics  
            "avg_sentence_length": avg_sentence_length,  
            "median_sentence_length": median_sentence_length,  
            "total_sentences": len(sentence_lengths),  
            "min_sentence_length": min(sentence_lengths) if sentence_lengths else 0,  
            "max_sentence_length": max(sentence_lengths) if sentence_lengths else 0  
        },
  
        # Hedging metrics  
        "hedging": {  
            "hedging_rate": hedging_rate,  
            "hedging_count": hedging_count,  
            "most_common_hedges": get_most_common_hedges(detailed_results)  
        },
  
        # Directness metrics  
        "directness": {  
            "directness_rate": directness_rate,  
            "direct_count": direct_count,  
            "indirect_count": total_responses - direct_count  
        },
  
        # Formatting metrics  
        "formatting": {  
            "bullet_usage_rate": bullet_usage_rate,  
            "table_usage_rate": table_usage_rate,  
            "emoji_usage_rate": emoji_usage_rate,  
            "bullet_count": bullet_count,  
            "table_count": table_count,  
            "emoji_count": emoji_count  
        },
  
        # Summary statistics  
        "summary": {  
            "total_responses": total_responses,  
            "avg_response_quality_indicators": {  
                "concise_and_direct": sum(1 for r in detailed_results  
                                        if r.get("token_count", 0) < mean_length  
                                        and r.get("is_direct", False)),  
                "verbose_with_hedging": sum(1 for r in detailed_results  
                                          if r.get("token_count", 0) > percentile_90_length  
                                          and r.get("has_hedging", False)),  
                "well_formatted": sum(1 for r in detailed_results  
                                    if r.get("has_bullets", False) or r.get("has_table", False)),  
                "high_sentence_density": sum(1 for r in detailed_results  
                                           if r.get("sentence_count", 0) > np.mean([r.get("sentence_count", 0) for r in detailed_results if "error" not in r]))  
            }  
        },
  
        # Detailed results for further analysis  
        "detailed_results": detailed_results if report_detailed else None  
    }

  
def extract_sentences(text: str) -> List[str]:  
    """  
    Extract individual sentences from text with robust handling of edge cases.
      
    Handles various sentence endings while avoiding false splits on:  
    - Common abbreviations (Dr., Mr., etc.)  
    - Decimal numbers (3.14)  
    - URLs and file paths
      
    Uses temporary placeholder replacement to protect abbreviations during splitting.
      
    Args:  
        text: Input text containing multiple sentences
          
    Returns:  
        List of sentence strings, cleaned and normalized  
    """  
    if not text:  
        return []
  
    # Clean text first - remove extra whitespace and normalize  
    text = re.sub(r"\s+", " ", text.strip())
  
    # Handle common abbreviations that shouldn't trigger sentence breaks  
    abbreviations = ["Dr", "Mr", "Mrs", "Ms", "Prof", "Inc", "Ltd", "Co", "Corp", "vs", "etc", "i.e", "e.g"]  
    temp_text = text
  
    # Temporarily replace abbreviations to avoid false sentence breaks  
    for i, abbr in enumerate(abbreviations):  
        temp_text = re.sub(rf"\b{abbr}\.", f"{abbr}__TEMP_DOT_{i}__", temp_text, flags=re.IGNORECASE)
  
    # Split on sentence endings, but be careful with decimal numbers and URLs  
    # This regex looks for sentence endings not preceded by digits  
    sentence_pattern = r"(?<!\d)(?<!\w\.\w)(?<=\.|\!|\?)\s+(?=[A-Z])"  
    sentences = re.split(sentence_pattern, temp_text)
  
    # Restore abbreviations  
    for i, abbr in enumerate(abbreviations):  
        sentences = [s.replace(f"{abbr}__TEMP_DOT_{i}__", f"{abbr}.") for s in sentences]
  
    # Clean up sentences - remove empty ones and strip whitespace  
    cleaned_sentences = []  
    for sentence in sentences:  
        sentence = sentence.strip()  
        if sentence and len(sentence) > 1:  # Filter out very short fragments  
            cleaned_sentences.append(sentence)
  
    return cleaned_sentences

  
def count_tokens(text: str) -> int:  
    """  
    Count approximate tokens in text using simple word-based tokenization.
      
    Uses regex to extract word tokens (alphanumeric sequences). This is a  
    simplified approximation; actual tokenization may differ by model.
      
    Args:  
        text: Input text to tokenize
          
    Returns:  
        Approximate token count (number of words)  
    """  
    if not text:  
        return 0
  
    # Simple tokenization - split on whitespace and extract words  
    tokens = re.findall(r'\b\w+\b', text)  
    return len(tokens)

  
def detect_hedging(text: str) -> bool:  
    """  
    Detect hedging phrases indicating uncertainty or AI disclaimers.
      
    Checks for multiple categories of hedging:  
    - Uncertainty markers (might, may, possibly)  
    - AI disclaimers (as an AI, I'm not sure)  
    - Qualification phrases (in my opinion, it seems)  
    - Caution phrases (please note, however)
      
    Args:  
        text: Input text to analyze for hedging
          
    Returns:  
        True if any hedging patterns are found, False otherwise  
    """  
    hedging_patterns = [  
        # Uncertainty markers  
        r'\bmight\b', r'\bmay\b', r'\bcould\b', r'\bwould\b', r'\bshould\b',  
        r'\bpossibly\b', r'\bprobably\b', r'\blikely\b', r'\bunlikely\b',  
        r'\bperhaps\b', r'\bmaybe\b', r'\bpotentially\b',
  
        # AI disclaimers  
        r'\bas an ai\b', r'\bi\'m an ai\b', r'\bi am an ai\b',  
        r'\bas a language model\b', r'\bas an assistant\b',  
        r'\bi cannot guarantee\b', r'\bi can\'t guarantee\b',  
        r'\bi\'m not sure\b', r'\bi am not sure\b',  
        r'\bi don\'t know\b', r'\bi do not know\b',
  
        # Qualification phrases  
        r'\bit seems\b', r'\bit appears\b', r'\bit looks like\b',  
        r'\bin my opinion\b', r'\bi think\b', r'\bi believe\b',  
        r'\bto my knowledge\b', r'\bas far as i know\b',  
        r'\bif i understand correctly\b', r'\bif i\'m not mistaken\b',
  
        # Caution phrases  
        r'\bplease note\b', r'\bkeep in mind\b', r'\bbear in mind\b',  
        r'\bit\'s worth noting\b', r'\bit should be noted\b',  
        r'\bhowever\b', r'\balthough\b', r'\bnevertheless\b'  
    ]
  
    text_lower = text.lower()
  
    for pattern in hedging_patterns:  
        if re.search(pattern, text_lower):  
            return True
  
    return False

  
def extract_hedging_phrases(text: str) -> List[str]:  
    """  
    Extract actual hedging phrases found in text for detailed analysis.
      
    Identifies and returns specific hedging expressions used in the text,  
    useful for understanding common hedging patterns.
      
    Args:  
        text: Input text to analyze
          
    Returns:  
        List of unique hedging phrases found (deduplicated)  
    """  
    hedging_patterns = [  
        r'\bmight\b', r'\bmay\b', r'\bcould\b', r'\bwould\b', r'\bshould\b',  
        r'\bpossibly\b', r'\bprobably\b', r'\blikely\b', r'\bunlikely\b',  
        r'\bperhaps\b', r'\bmaybe\b', r'\bpotentially\b',  
        r'\bas an ai\b', r'\bi\'m an ai\b', r'\bi am an ai\b',  
        r'\bas a language model\b', r'\bas an assistant\b',  
        r'\bi cannot guarantee\b', r'\bi can\'t guarantee\b',  
        r'\bi\'m not sure\b', r'\bi am not sure\b',  
        r'\bit seems\b', r'\bit appears\b', r'\bit looks like\b',  
        r'\bin my opinion\b', r'\bi think\b', r'\bi believe\b'  
    ]
  
    found_phrases = []  
    text_lower = text.lower()
  
    for pattern in hedging_patterns:  
        matches = re.findall(pattern, text_lower)  
        found_phrases.extend(matches)
  
    return list(set(found_phrases))  # Remove duplicates

  
def assess_directness(text: str, prompt: str = "") -> bool:  
    """  
    Assess whether the response provides a direct answer in the first sentence.
      
    Evaluates directness by checking if the opening sentence contains:  
    - Direct answer indicators (is, are, yes, no)  
    - Concrete information (numbers, proper nouns)  
    - Definitive statements without hedging
      
    Penalizes responses that start with questions or heavy hedging.
      
    Args:  
        text: Model response text to evaluate  
        prompt: Optional prompt text (currently unused but reserved for context-aware assessment)
          
    Returns:  
        True if first sentence contains a direct response, False otherwise  
    """  
    if not text:  
        return False
  
    # Extract first sentence  
    first_sentence = extract_first_sentence(text)
  
    if not first_sentence or len(first_sentence.split()) < 3:  
        return False
  
    # Check for direct answer indicators  
    directness_indicators = [  
        # Definitive statements  
        r'\bis\b', r'\bare\b', r'\bwas\b', r'\bwere\b',  
        r'\byes\b', r'\bno\b', r'\btrue\b', r'\bfalse\b',
  
        # Numbers and quantities  
        r'\b\d+\b', r'\bfirst\b', r'\bsecond\b', r'\bthird\b',  
        r'\bmain\b', r'\bprimary\b', r'\bkey\b',
  
        # Direct answer starters  
        r'^the answer is\b', r'^the result is\b', r'^the solution is\b',  
        r'^to answer\b', r'^simply put\b', r'^in short\b'  
    ]
  
    first_sentence_lower = first_sentence.lower()
  
    # Check for direct indicators  
    direct_indicator_count = sum(1 for pattern in directness_indicators  
                               if re.search(pattern, first_sentence_lower))
  
    # Check for hedging in first sentence (reduces directness)  
    has_first_sentence_hedging = detect_hedging(first_sentence)
  
    # Check for question format (reduces directness)  
    is_question = first_sentence.strip().endswith('?')
  
    # Scoring logic  
    if direct_indicator_count >= 2 and not has_first_sentence_hedging and not is_question:  
        return True  
    elif direct_indicator_count >= 1 and not has_first_sentence_hedging and not is_question:  
        # Additional check: does first sentence contain concrete information?  
        return contains_concrete_information(first_sentence)
  
    return False

  
def extract_first_sentence(text: str) -> str:  
    """  
    Extract the first sentence from text by splitting on sentence endings.
      
    Args:  
        text: Input text containing one or more sentences
          
    Returns:  
        First sentence string, or entire text if no sentence ending found  
    """  
    # Split on sentence endings  
    sentences = re.split(r'[.!?]+', text)
  
    if sentences:  
        first_sentence = sentences[0].strip()  
        return first_sentence
  
    return text.strip()

  
def contains_concrete_information(sentence: str) -> bool:  
    """  
    Check if sentence contains concrete, specific information.
      
    Looks for indicators of specificity including:  
    - Proper nouns (capitalized words)  
    - Numbers and quantities  
    - Dates (months, years)  
    - Currency amounts  
    - Absolute terms (always, never, all)
      
    Args:  
        sentence: Sentence text to analyze
          
    Returns:  
        True if sentence contains concrete information indicators, False otherwise  
    """  
    # Look for specific entities, numbers, or definitive statements  
    concrete_patterns = [  
        r'\b[A-Z][a-z]+\b',  # Proper nouns  
        r'\b\d+(?:\.\d+)?\b',  # Numbers  
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b',  # Months  
        r'\b\d{4}\b',  # Years  
        r'\$\d+',  # Money  
        r'\b(?:always|never|all|none|every|each)\b'  # Absolute terms  
    ]
  
    return any(re.search(pattern, sentence, re.IGNORECASE) for pattern in concrete_patterns)

  
def detect_bullets(text: str) -> bool:  
    """  
    Detect bullet points or numbered lists in text.
      
    Recognizes multiple list formats:  
    - Bullet points (-, *, •)  
    - Numbered lists (1., 2., etc.)  
    - Numbered with parentheses (1), 2), etc.)  
    - Lettered lists (a., b., etc.)
      
    Args:  
        text: Input text to check for list formatting
          
    Returns:  
        True if any bullet or list pattern is found, False otherwise  
    """  
    bullet_patterns = [  
        r'^\s*[-*•]\s',  # Bullet points  
        r'^\s*\d+\.\s',  # Numbered lists  
        r'^\s*\d+\)\s',  # Numbered lists with parentheses  
        r'^\s*[a-zA-Z]\.\s',  # Lettered lists  
        r'^\s*[a-zA-Z]\)\s'  # Lettered lists with parentheses  
    ]
  
    lines = text.split('\n')
  
    for line in lines:  
        for pattern in bullet_patterns:  
            if re.search(pattern, line, re.MULTILINE):  
                return True
  
    return False

  
def detect_table(text: str) -> bool:  
    """  
    Detect markdown-style tables in text.
      
    Looks for markdown table indicators including:  
    - Pipe separators (|)  
    - Table alignment markers (|:--|)  
    - Multiple consecutive lines with table formatting
      
    Requires at least 2 lines with table formatting to confirm presence.
      
    Args:  
        text: Input text to check for table formatting
          
    Returns:  
        True if markdown table detected (2+ table lines), False otherwise  
    """  
    # Look for markdown table separators  
    table_patterns = [  
        r'\|.*\|',  # Lines with pipe separators  
        r'^\s*\|.*\|\s*$',  # Full table rows  
        r'\|.*:.*\|',  # Table headers with alignment  
    ]
  
    lines = text.split('\n')  
    table_line_count = 0
  
    for line in lines:  
        for pattern in table_patterns:  
            if re.search(pattern, line):  
                table_line_count += 1  
                break
  
    # Require at least 2 lines that look like table rows  
    return table_line_count >= 2

  
def detect_emoji(text: str) -> bool:  
    """  
    Detect unicode emojis in text.
      
    Checks for emoji characters across multiple unicode ranges:  
    - Emoticons (😀-😿)  
    - Symbols & pictographs (🌀-🗿)  
    - Transport & map symbols (🚀-🛿)  
    - Flags (🇦-🇿)  
    - Dingbats (✂-➰)
      
    Args:  
        text: Input text to check for emojis
          
    Returns:  
        True if any emoji characters found, False otherwise  
    """  
    # Unicode ranges for emojis  
    emoji_pattern = re.compile(  
        "["  
        "\U0001F600-\U0001F64F"  # emoticons  
        "\U0001F300-\U0001F5FF"  # symbols & pictographs  
        "\U0001F680-\U0001F6FF"  # transport & map symbols  
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)  
        "\U00002702-\U000027B0"  # dingbats  
        "\U000024C2-\U0001F251"  
        "]+",  
        flags=re.UNICODE  
    )
  
    return bool(emoji_pattern.search(text))

  
def get_most_common_hedges(detailed_results: List[Dict]) -> List[Tuple[str, int]]:  
    """  
    Identify the most frequently used hedging phrases across all responses.
      
    Aggregates hedging phrases from all samples and returns the top 10  
    most common ones with their frequencies.
      
    Args:  
        detailed_results: List of per-sample result dictionaries containing  
                         'hedging_phrases' keys
          
    Returns:  
        List of (phrase, count) tuples for the 10 most common hedges,  
        sorted by frequency in descending order  
    """  
    all_hedges = []  
    for result in detailed_results:  
        if 'hedging_phrases' in result:  
            all_hedges.extend(result['hedging_phrases'])
  
    hedge_counter = Counter(all_hedges)  
    return hedge_counter.most_common(10)  # Top 10 most common hedges

  
def get_empty_metrics() -> Dict[str, Any]:  
    """  
    Return empty metrics structure for edge cases with no valid data.
      
    Provides a consistent metrics dictionary structure when the input  
    dataframe is empty or contains no valid responses.
      
    Returns:  
        Dictionary with same structure as compute_style_elaboration_metrics()  
        but with all counts set to 0 and all rates set to 0.0  
    """  
    return {  
        'verbosity': {  
            'mean_answer_length': 0.0,  
            'percentile_90_length': 0.0,  
            'min_length': 0,  
            'max_length': 0,  
            'std_length': 0.0,  
            'avg_sentence_length': 0.0,  
            'median_sentence_length': 0.0,  
            'total_sentences': 0,  
            'min_sentence_length': 0,  
            'max_sentence_length': 0  
        },  
        'hedging': {  
            'hedging_rate': 0.0,  
            'hedging_count': 0,  
            'most_common_hedges': []  
        },  
        'directness': {  
            'directness_rate': 0.0,  
            'direct_count': 0,  
            'indirect_count': 0  
        },  
        'formatting': {  
            'bullet_usage_rate': 0.0,  
            'table_usage_rate': 0.0,  
            'emoji_usage_rate': 0.0,  
            'bullet_count': 0,  
            'table_count': 0,  
            'emoji_count': 0  
        },  
        'summary': {  
            'total_responses': 0,  
            'avg_response_quality_indicators': {  
                'concise_and_direct': 0,  
                'verbose_with_hedging': 0,  
                'well_formatted': 0,  
                'high_sentence_density': 0  
            }  
        },  
        'detailed_results': []  
    }  