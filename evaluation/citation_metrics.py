import re  
from typing import Dict, List, Any, Optional, Tuple
  
def compute_citation_accuracy(results_df, debug=False):  
    """  
    Evaluate citation accuracy for source attribution tasks with support for multiple citations.
  
    Args:  
        results_df: DataFrame with citation results containing "outputs", "gold" columns  
        debug: If True, print debug information during evaluation
  
    Returns:  
        Dictionary with simplified citation evaluation metrics including accuracy scores,  
        error breakdown, and detailed results per item  
    """
  
    if len(results_df) == 0:  
        return get_empty_citation_metrics()
  
    # Initialize counters  
    format_correct = 0  
    response_correct = 0  
    source_correct = 0  
    complete_correct = 0  
    total_items = 0
  
    detailed_results = []  
    error_patterns = {  
        "no_citation": 0,  
        "wrong_format": 0,  
        "incorrect_source": 0,  
        "wrong_answer": 0,  
        "missing_sources": 0  
    }
  
    for idx, row in results_df.iterrows():  
        try:  
            problem_id = row.get("id", idx)  
            model_output = row.get("outputs", "").strip()  
            gold = row.get("gold", {})  
            gold_answer = gold.get("answer", "")  
            supporting_facts = gold.get("supporting_facts", {})  
            context_titles = gold.get("context_titles", None)
  
            # Deduplicate and filter supporting facts  
            filtered_supporting_facts = deduplicate_and_filter_supporting_facts(  
                supporting_facts, context_titles  
            )
  
            # Parse model output to extract multiple citations  
            parsed_response = parse_multiple_citation_response(model_output)  
            if debug:  
                print(f'parsed_response: {parsed_response}')
  
            # Evaluate components  
            format_acc = evaluate_citation_format(parsed_response)  
            response_acc = evaluate_response_correctness(  
                parsed_response.get("answer", ""), gold_answer  
            )  
            source_acc = evaluate_source_attribution(  
                parsed_response.get("citations", []), filtered_supporting_facts  
            )
  
            # Update counters  
            if format_acc:  
                format_correct += 1  
            if response_acc:  
                response_correct += 1  
            source_correct += source_acc
  
            complete_acc = format_acc and response_acc and source_acc == 1  
            if complete_acc:  
                complete_correct += 1
  
            # Error analysis  
            error_type = analyze_citation_error(  
                parsed_response, format_acc, response_acc, source_acc, filtered_supporting_facts  
            )  
            if error_type in error_patterns:  
                error_patterns[error_type] += 1
  
            detailed_results.append({  
                "id": problem_id,  
                "complete_correct": complete_acc,  
                "error_type": error_type,  
                "required_sources": len(filtered_supporting_facts.get("title", [])),  
                "provided_sources": len(parsed_response.get("citations", []))  
            })
  
            total_items += 1
  
        except Exception as e:  
            error_patterns["wrong_format"] += 1  
            detailed_results.append({  
                "id": row.get("id", idx),  
                "complete_correct": False,  
                "error_type": "wrong_format",  
                "error": str(e)  
            })  
            total_items += 1
  
    # Calculate simplified metrics  
    return {  
        "accuracy": complete_correct / total_items if total_items > 0 else 0.0,  
        "format_accuracy": format_correct / total_items if total_items > 0 else 0.0,  
        "answer_accuracy": response_correct / total_items if total_items > 0 else 0.0,  
        "source_accuracy": source_correct / total_items if total_items > 0 else 0.0,  
        "citation_usage_rate": (total_items - error_patterns["no_citation"]) / total_items if total_items > 0 else 0.0,  
        "total_items": total_items,  
        "error_breakdown": error_patterns,  
        "multi_source_items": len([r for r in detailed_results if r.get("required_sources", 0) > 1]),  
        "avg_sources_required": sum(r.get("required_sources", 0) for r in detailed_results) / total_items if total_items > 0 else 0.0,  
        "detailed_results": detailed_results  
    }
  
def deduplicate_and_filter_supporting_facts(supporting_facts: Dict[str, Any],  
                                          context_titles: Optional[List[str]] = None) -> Dict[str, Any]:  
    """  
    Deduplicate repeated references and filter by available context.
  
    Args:  
        supporting_facts: Original supporting facts from gold data  
        context_titles: List of titles available in context (None means no filtering)
  
    Returns:  
        Deduplicated and filtered supporting facts dictionary with "title" and "sent_id" keys  
    """  
    if not supporting_facts or "title" not in supporting_facts:  
        return supporting_facts
  
    original_titles = supporting_facts.get("title", [])  
    original_sent_ids = supporting_facts.get("sent_id", [])
  
    # Deduplicate titles while preserving order  
    seen_titles = set()  
    unique_titles = []  
    unique_sent_ids = []
  
    for i, title in enumerate(original_titles):  
        if title not in seen_titles:  
            seen_titles.add(title)  
            unique_titles.append(title)  
            if i < len(original_sent_ids):  
                unique_sent_ids.append(original_sent_ids[i])
  
    # Filter by context if provided  
    if context_titles is not None:  
        filtered_titles = []  
        filtered_sent_ids = []
  
        for i, title in enumerate(unique_titles):  
            if title in context_titles:  
                filtered_titles.append(title)  
                if i < len(unique_sent_ids):  
                    filtered_sent_ids.append(unique_sent_ids[i])
  
        return {  
            "title": filtered_titles,  
            "sent_id": filtered_sent_ids  
        }
  
    return {  
        "title": unique_titles,  
        "sent_id": unique_sent_ids  
    }
  
def parse_multiple_citation_response(model_output: str) -> Dict[str, Any]:    
    """    
    Parse model output to extract answer and multiple citation components.
    
    Args:    
        model_output: Raw model response text
    
    Returns:    
        Dictionary with parsed components including answer text, list of citations,  
        has_citation flag, and citation_count  
    """    
    parsed = {    
        "answer": "",    
        "citations": [],    
        "has_citation": False,    
        "citation_count": 0    
    }
    
    if not model_output:    
        return parsed
    
    # Pattern 1: Comma-separated citations in brackets - [Citation1, Citation2, Citation3]    
    #comma_pattern = r"\[([^\]]+(?:,[^\]]+)+)\]"    
    comma_pattern = r"\[([^\],]+(?:,[^\],]+)+)\]"   
    comma_match = re.search(comma_pattern, model_output)
    
    if comma_match:    
        citation_text = comma_match.group(1)    
        citations = [citation.strip() for citation in citation_text.split(",")]    
        answer_text = re.sub(comma_pattern, "", model_output).strip()    
        parsed["answer"] = answer_text    
        parsed["citations"] = citations    
        parsed["has_citation"] = True    
        parsed["citation_count"] = len(citations)    
        return parsed
    
    # Pattern 2: Semicolon-separated citations in brackets - [Citation1; Citation2]    
    semicolon_pattern = r"\[([^\]]+(?:;[^\]]+)+)\]"    
    semicolon_match = re.search(semicolon_pattern, model_output)
    
    if semicolon_match:    
        citation_text = semicolon_match.group(1)    
        citations = [citation.strip() for citation in citation_text.split(";")]    
        answer_text = re.sub(semicolon_pattern, "", model_output).strip()    
        parsed["answer"] = answer_text    
        parsed["citations"] = citations    
        parsed["has_citation"] = True    
        parsed["citation_count"] = len(citations)    
        return parsed
    
    # Pattern 3: Multiple bracket citations - [Citation1] [Citation2]    
    bracket_citations = re.findall(r"\[([^\]]+)\]", model_output)
    
    if bracket_citations:    
        answer_text = re.sub(r"\s*\[([^\]]+)\]\s*", " ", model_output).strip()    
        parsed["answer"] = answer_text    
        parsed["citations"] = [citation.strip() for citation in bracket_citations]    
        parsed["has_citation"] = True    
        parsed["citation_count"] = len(bracket_citations)    
        return parsed
    
    # Pattern 4: Multiple parenthetical citations - (Citation1) (Citation2)    
    paren_citations = re.findall(r"\(([^)]+)\)", model_output)    
    citation_candidates = [c for c in paren_citations if is_likely_citation(c)]
    
    if citation_candidates:    
        answer_text = model_output    
        for citation in citation_candidates:    
            answer_text = answer_text.replace(f"({citation})", "").strip()    
        parsed["answer"] = answer_text    
        parsed["citations"] = citation_candidates    
        parsed["has_citation"] = True    
        parsed["citation_count"] = len(citation_candidates)    
        return parsed
    
    # No citations found    
    parsed["answer"] = model_output.strip()    
    return parsed  
  
def evaluate_citation_format(parsed_response: Dict[str, Any]) -> bool:  
    """  
    Evaluate whether the citation format is correct.
      
    Args:  
        parsed_response: Parsed response dictionary from parse_multiple_citation_response
          
    Returns:  
        True if citation format is valid (has citations), False otherwise  
    """  
    return parsed_response.get("has_citation", False) and len(parsed_response.get("citations", [])) > 0
  
def evaluate_response_correctness(model_answer: str, gold_answer: str) -> bool:  
    """  
    Evaluate whether the response is correct and complete.
      
    Args:  
        model_answer: The model's answer text  
        gold_answer: The gold standard answer text
          
    Returns:  
        True if the model answer matches or is semantically similar to gold answer  
    """  
    if not model_answer or not gold_answer:  
        return False
  
    model_clean = clean_text(model_answer)  
    gold_clean = clean_text(gold_answer)
  
    # Exact match or containment  
    if model_clean == gold_clean or gold_clean in model_clean:  
        return True
  
    # Semantic similarity for key entities  
    model_entities = extract_key_entities(model_clean)  
    gold_entities = extract_key_entities(gold_clean)  
    if model_entities and gold_entities:  
        return len(model_entities.intersection(gold_entities)) > 0
  
    # Text similarity  
    sim = calculate_text_similarity(model_clean, gold_clean)  
    return calculate_text_similarity(model_clean, gold_clean) >= 0.8
  
def evaluate_source_attribution(model_citations: List[str], supporting_facts: Dict[str, Any]) -> float:  
    """  
    Evaluate whether the cited sources are correct for multiple citation scenario.
  
    Args:  
        model_citations: List of model's citations  
        supporting_facts: Deduplicated gold standard supporting facts
  
    Returns:  
        Float representing the proportion of required sources that are correctly cited (0.0 to 1.0)  
    """  
    if not model_citations:  
        return 0.0
  
    gold_titles = supporting_facts.get("title", [])  
    if not gold_titles:  
        return 0.0
  
    # Clean citations and titles  
    model_citations_clean = [clean_text(citation) for citation in model_citations]  
    gold_titles_clean = [clean_text(title) for title in gold_titles]
  
    # Check if all required sources are covered  
    matched_gold_titles = set()
  
    for model_citation in model_citations_clean:  
        for gold_title in gold_titles_clean:  
            if is_citation_match(model_citation, gold_title):  
                matched_gold_titles.add(gold_title)  
                break
  
    # Success if all required sources are matched  
    return len(matched_gold_titles)/len(gold_titles_clean)
  
def is_citation_match(model_citation: str, gold_title: str) -> bool:  
    """  
    Check if a model citation matches a gold title.
      
    Args:  
        model_citation: Citation text from model output  
        gold_title: Gold standard title
          
    Returns:  
        True if the citation matches the title (exact, partial, or similarity-based)  
    """  
    # Exact match  
    if model_citation == gold_title:  
        return True
  
    # Partial match (containment)  
    if gold_title in model_citation or model_citation in gold_title:  
        return True
  
    # Similarity-based match  
    return calculate_text_similarity(model_citation, gold_title) >= 0.7
  
def analyze_citation_error(parsed_response: Dict, format_acc: bool, response_acc: bool,  
                         source_acc: float, supporting_facts: Dict) -> str:  
    """  
    Analyze the type of citation error.
      
    Args:  
        parsed_response: Parsed response dictionary  
        format_acc: Whether citation format is correct  
        response_acc: Whether the answer is correct  
        source_acc: Source attribution accuracy score  
        supporting_facts: Gold standard supporting facts
          
    Returns:  
        String describing the error type: "no_citation", "wrong_format", "wrong_answer",  
        "missing_sources", "incorrect_source", or "correct"  
    """
  
    if not parsed_response.get("has_citation", False):  
        return "no_citation"
  
    if not format_acc:  
        return "wrong_format"
  
    if not response_acc:  
        return "wrong_answer"
  
    if not source_acc:  
        model_citations = parsed_response.get("citations", [])  
        gold_titles = supporting_facts.get("title", [])
  
        if len(model_citations) < len(gold_titles):  
            return "missing_sources"  
        else:  
            return "incorrect_source"
  
    return "correct"
  
# Helper functions  
def is_likely_citation(text: str) -> bool:  
    """  
    Check if text looks like a citation rather than clarification.
      
    Args:  
        text: Text to evaluate
          
    Returns:  
        True if text appears to be a citation based on heuristics  
    """  
    citation_indicators = [  
        len(text) > 3,  
        any(char.isupper() for char in text),  
        not text.lower().startswith(("i.e.", "e.g.", "see", "note")),  
        not re.match(r"^\d+$", text.strip())  
    ]  
    return sum(citation_indicators) >= 2
  
def clean_text(text: str) -> str:  
    """  
    Clean and normalize text for comparison.
      
    Args:  
        text: Text to clean
          
    Returns:  
        Cleaned and normalized text (lowercase, normalized whitespace, no trailing punctuation)  
    """  
    if not text:  
        return ""  
    cleaned = re.sub(r"\s+", " ", text.strip())  
    cleaned = re.sub(r"[.!?]+$", "", cleaned)  
    return cleaned.lower()
  
def extract_key_entities(text: str) -> set:  
    """  
    Extract key entities from text (proper nouns and quoted terms).
      
    Args:  
        text: Text to extract entities from
          
    Returns:  
        Set of extracted entity strings (lowercase)  
    """  
    if not text:  
        return set()
  
    entities = set()  
    # Find proper nouns  
    proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)  
    entities.update([entity.lower() for entity in proper_nouns])
  
    # Find quoted terms (content inside double quotes only)  
    quoted_terms = re.findall(r'"([^"]+)"', text)  
    entities.update([term.lower() for term in quoted_terms])
  
    return entities
  
def calculate_text_similarity(text1: str, text2: str) -> float:  
    """  
    Calculate Jaccard similarity between two text strings.
      
    Args:  
        text1: First text string  
        text2: Second text string
          
    Returns:  
        Jaccard similarity score (0.0 to 1.0)  
    """  
    if not text1 or not text2:  
        return 0.0
  
    words1 = set(text1.lower().split())  
    words2 = set(text2.lower().split())
  
    intersection = len(words1.intersection(words2))  
    union = len(words1.union(words2))
  
    return intersection / union if union > 0 else 0.0
  
def get_empty_citation_metrics() -> Dict[str, Any]:  
    """  
    Return empty metrics for edge cases (e.g., empty DataFrame).
      
    Returns:  
        Dictionary with all metrics set to zero/empty values  
    """  
    return {  
        "accuracy": 0.0,  
        "format_accuracy": 0.0,  
        "answer_accuracy": 0.0,  
        "source_accuracy": 0.0,  
        "citation_usage_rate": 0.0,  
        "total_items": 0,  
        "error_breakdown": {  
            "no_citation": 0,  
            "wrong_format": 0,  
            "incorrect_source": 0,  
            "wrong_answer": 0,  
            "missing_sources": 0  
        },  
        "multi_source_items": 0,  
        "avg_sources_required": 0.0,  
        "detailed_results": []  
    }  