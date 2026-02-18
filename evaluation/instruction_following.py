import json  
import tempfile  
import os  
from typing import Dict, List, Any, Optional  
import pandas as pd  
from datasets import load_dataset  
import nltk
  
# Import the official IFEval evaluation  
from .ifeval import evaluation_lib  
from .llm_judge import llm_judge_eval

# Global variable to cache the dataset  
_IFEVAL_DATASET = None
# Global flag to track if NLTK data has been downloaded
_NLTK_DATA_DOWNLOADED = False

def _ensure_nltk_data():
    """
    Ensure required NLTK data is downloaded.
    This is called lazily on first use to avoid import-time side effects.
    """
    global _NLTK_DATA_DOWNLOADED
    if not _NLTK_DATA_DOWNLOADED:
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        _NLTK_DATA_DOWNLOADED = True
  
def load_ifeval_dataset():  
    """  
    Load and cache the IFEval dataset from Hugging Face.
      
    Returns:  
        Dataset object containing IFEval train split (cached after first load)  
    """  
    global _IFEVAL_DATASET  
    if _IFEVAL_DATASET is None:
        _ensure_nltk_data()  # Ensure NLTK data is available before loading dataset
        dataset = load_dataset("google/IFEval")  
        _IFEVAL_DATASET = dataset["train"]  # IFEval only has a train split  
    return _IFEVAL_DATASET
  
def compute_ifeval_accuracy(results_df, report_detailed=False):  
    """  
    Evaluate instruction following using official IFEval implementation.
  
    Maps DataFrame format to official IFEval format and returns metrics.
  
    Args:  
        results_df: DataFrame with columns ["id", "prompt", "outputs", "gold"]  
        report_detailed: If True, include detailed per-item results in output
  
    Returns:  
        Dictionary with instruction-following metrics from official implementation,  
        including strict and loose mode results with prompt-level and instruction-level  
        accuracy, tier-based metrics, and constraint analysis  
    """
  
    if len(results_df) == 0:  
        return get_empty_ifeval_metrics()
  
    try:  
        # Load the original IFEval dataset  
        ifeval_dataset = load_ifeval_dataset()
    
        # Convert DataFrame to official IFEval format  
        inputs, prompt_to_response = prepare_ifeval_data(results_df, ifeval_dataset)
    
        # Run official evaluation for both strict and loose modes  
        results = {}
    
        for mode in ["strict", "loose"]:  
            if mode == "strict":  
                eval_func = evaluation_lib.test_instruction_following_strict  
            else:  
                eval_func = evaluation_lib.test_instruction_following_loose
    
            # Evaluate each input  
            outputs = []  
            for inp in inputs:
                result = eval_func(inp, prompt_to_response)  
                outputs.append(result)
    
            # Parse detailed results  
            mode_results = parse_evaluation_results(outputs, mode, report_detailed)
    
            results[mode] = mode_results
  
    except Exception as e:  
        # Fallback to empty metrics on error  
        return {  
            "error": str(e),  
            "strict": get_empty_ifeval_metrics(),  
            "loose": get_empty_ifeval_metrics()  
        }
  
    return results
  
def prepare_ifeval_data(results_df, ifeval_dataset):  
    """  
    Convert DataFrame format to official IFEval input format by matching prompts  
    to the original dataset.
  
    Args:  
        results_df: DataFrame with IFEval data containing 'id', 'prompt', 'outputs', 'gold' columns  
        ifeval_dataset: The original IFEval dataset from Hugging Face
  
    Returns:  
        Tuple of (inputs, prompt_to_response) where inputs is a list of InputExample objects  
        and prompt_to_response is a dict mapping prompts to model responses  
    """  
    inputs = []  
    prompt_to_response = {}
  
    # Create a lookup dictionary from the original dataset for fast matching  
    ifeval_lookup = create_ifeval_lookup(ifeval_dataset)
  
    unmatched_prompts = []
  
    for idx, row in results_df.iterrows():  
        # Extract the core instruction from the prompt  
        full_prompt = row.get("prompt", "")  
        core_instruction = extract_core_instruction(full_prompt)
  
        # Try to match the core instruction to the original dataset  
        matched_data = match_prompt_to_ifeval(core_instruction, ifeval_lookup)
  
        if matched_data:  
            # Use the matched instruction_id_list and kwargs from original dataset  
            instruction_id_list = matched_data["instruction_id_list"]  
            original_kwargs = matched_data["kwargs"]
  
            # Filter kwargs for each instruction based on what it expects  
            filtered_kwargs = []  
            for i, instruction_id in enumerate(instruction_id_list):  
                if i < len(original_kwargs):  
                    instruction_kwargs = filter_kwargs_for_instruction(  
                        instruction_id, original_kwargs[i]  
                    )  
                    filtered_kwargs.append(instruction_kwargs)  
                else:  
                    filtered_kwargs.append({})
  
            # Create InputExample format expected by official evaluation  
            input_example = evaluation_lib.InputExample(  
                key=idx,  # Using index as key since it expects int  
                instruction_id_list=instruction_id_list,  
                prompt=core_instruction,  
                kwargs=filtered_kwargs  
            )
  
            inputs.append(input_example)
  
            # Map prompt to response for official evaluation  
            prompt_to_response[core_instruction] = row.get("outputs", "")  
        else:  
            # Track unmatched prompts for debugging  
            unmatched_prompts.append({  
                "idx": idx,  
                "prompt": core_instruction[:100] + "..." if len(core_instruction) > 100 else core_instruction  
            })
  
            # For unmatched prompts, try to use the gold column as fallback  
            instruction_id_list = row.get("gold", [])  
            if instruction_id_list:  
                # Create empty kwargs as fallback  
                kwargs = [{}] * len(instruction_id_list)
  
                input_example = evaluation_lib.InputExample(  
                    key=idx,  
                    instruction_id_list=instruction_id_list,  
                    prompt=core_instruction,  
                    kwargs=kwargs  
                )
  
                inputs.append(input_example)  
                prompt_to_response[core_instruction] = row.get("outputs", "")
  
    if unmatched_prompts:  
        print(f"Warning: {len(unmatched_prompts)} prompts could not be matched to the original IFEval dataset:")  
        for unmatched in unmatched_prompts[:5]:  # Show first 5  
            print(f'  - Index {unmatched["idx"]}: {unmatched["prompt"]}')  
        if len(unmatched_prompts) > 5:  
            print(f"  ... and {len(unmatched_prompts) - 5} more")
  
    return inputs, prompt_to_response
  
def filter_kwargs_for_instruction(instruction_id, original_kwargs):  
    """  
    Filter kwargs based on what each instruction expects.
  
    Args:  
        instruction_id: The instruction ID (e.g., "language:response_language")  
        original_kwargs: The original kwargs dictionary from dataset
  
    Returns:  
        Filtered kwargs dictionary containing only expected parameters for the instruction  
    """  
    if not original_kwargs:  
        return {}
  
    # Import the instructions registry to get the instruction class  
    from .ifeval import instructions_registry
  
    # Get the instruction class  
    if instruction_id not in instructions_registry.INSTRUCTION_DICT:  
        return {}
  
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
  
    # Create a temporary instance to get expected args  
    temp_instruction = instruction_cls(instruction_id)
  
    try:  
        expected_args = temp_instruction.get_instruction_args_keys()  
    except (NotImplementedError, AttributeError):  
        # If the method doesn't exist or isn't implemented, return empty dict  
        return {}
  
    # Filter original_kwargs to only include expected arguments  
    filtered_kwargs = {}  
    for key in expected_args:  
        if key in original_kwargs:  
            filtered_kwargs[key] = original_kwargs[key]
  
    return filtered_kwargs
  
def create_ifeval_lookup(ifeval_dataset):  
    """  
    Create a lookup dictionary from the IFEval dataset for efficient prompt matching.
  
    Args:  
        ifeval_dataset: The original IFEval dataset from Hugging Face
  
    Returns:  
        Dictionary mapping normalized prompts to their instruction data  
        (instruction_id_list, kwargs, original_prompt)  
    """  
    lookup = {}
  
    for item in ifeval_dataset:  
        prompt = item["prompt"]  
        normalized_prompt = normalize_prompt_for_matching(prompt)
  
        lookup[normalized_prompt] = {  
            "instruction_id_list": item["instruction_id_list"],  
            "kwargs": item["kwargs"],  
            "original_prompt": prompt  
        }
  
    return lookup
  
def normalize_prompt_for_matching(prompt):  
    """  
    Normalize a prompt for matching by removing extra whitespace and standardizing format.
  
    Args:  
        prompt: The prompt string to normalize
  
    Returns:  
        Normalized prompt string (lowercase, whitespace normalized, quotes removed)  
    """  
    # Remove extra whitespace  
    normalized = " ".join(prompt.split())
  
    # Convert to lowercase for case-insensitive matching  
    normalized = normalized.lower()
  
    # Remove common punctuation that might vary  
    normalized = normalized.replace(""", "").replace(""", "").replace("`", "")
  
    return normalized
  
def match_prompt_to_ifeval(core_instruction, ifeval_lookup):  
    """  
    Match a core instruction to the IFEval dataset using exact and fuzzy matching.
  
    Args:  
        core_instruction: The extracted core instruction text  
        ifeval_lookup: Lookup dictionary from IFEval dataset
  
    Returns:  
        Matched data dictionary with instruction_id_list and kwargs, or None if no match found  
    """  
    normalized_instruction = normalize_prompt_for_matching(core_instruction)
  
    # Try exact match first  
    if normalized_instruction in ifeval_lookup:  
        return ifeval_lookup[normalized_instruction]
  
    # Try fuzzy matching for partial matches  
    for lookup_prompt, data in ifeval_lookup.items():  
        # Check if core instruction is contained in the lookup prompt  
        if normalized_instruction in lookup_prompt or lookup_prompt in normalized_instruction:  
            # Additional check: ensure they have similar length to avoid false matches  
            len_ratio = min(len(normalized_instruction), len(lookup_prompt)) / max(len(normalized_instruction), len(lookup_prompt))  
            if len_ratio > 0.7:  # At least 70% length similarity  
                return data
  
    # Try matching by key phrases (for cases where formatting differs significantly)  
    key_phrases = extract_key_phrases(normalized_instruction)  
    if key_phrases:  
        for lookup_prompt, data in ifeval_lookup.items():  
            lookup_phrases = extract_key_phrases(lookup_prompt)  
            if key_phrases & lookup_phrases:  # Set intersection  
                return data
  
    return None
  
def extract_key_phrases(text):  
    """  
    Extract key phrases from text for fuzzy matching.
  
    Args:  
        text: Input text to extract phrases from
  
    Returns:  
        Set of key phrases including numbers, instruction keywords, and quoted strings  
    """  
    import re
  
    # Extract numbers, specific formatting requirements, and key nouns  
    phrases = set()
  
    # Extract numbers  
    numbers = re.findall(r"\d+", text)  
    phrases.update(numbers)
  
    # Extract specific instruction keywords  
    instruction_keywords = [  
        "sections", "words", "sentences", "bullet", "points", "paragraphs",  
        "language", "format", "json", "markdown", "title", "quotation",  
        "capital", "lowercase", "comma", "frequency", "keyword", "forbidden"  
    ]
  
    for keyword in instruction_keywords:  
        if keyword in text:  
            phrases.add(keyword)
  
    # Extract quoted strings (often contain specific requirements)  
    quotes = re.findall(r'([^"]*)', text)  
    phrases.update(quotes)
  
    return phrases
  
def extract_core_instruction(full_prompt):  
    """  
    Extract the core instruction from the formatted prompt.
  
    The full prompt contains system messages and formatting, but we need  
    just the core instruction for matching against the original dataset.
      
    Args:  
        full_prompt: Full formatted prompt including system messages
          
    Returns:  
        Core instruction text extracted from the prompt  
    """  
    # Look for the instruction after "Instruction:" marker  
    if "Instruction:" in full_prompt:  
        instruction_part = full_prompt.split("Instruction:")[1]  
        # Take everything up to "Rules:" if it exists  
        if "Rules:" in instruction_part:  
            core_instruction = instruction_part.split("Rules:")[0].strip()  
        else:  
            # Take everything up to "Response:" if it exists  
            if "Response:" in instruction_part:  
                core_instruction = instruction_part.split("Response:")[0].strip()  
            else:  
                core_instruction = instruction_part.strip()  
    else:  
        # Fallback: try to extract from user message  
        if "<|start_header_id|>user<|end_header_id|>" in full_prompt:  
            user_part = full_prompt.split("<|start_header_id|>user<|end_header_id|>")[1]  
            if "<|eot_id|>" in user_part:  
                user_part = user_part.split("<|eot_id|>")[0]  
            # Extract instruction after constraint-following assistant text  
            if "Instruction:" in user_part:  
                core_instruction = user_part.split("Instruction:")[1].split("Rules:")[0].strip()  
            else:  
                # Take everything after the system prompt  
                lines = user_part.strip().split("\n")  
                # Skip system instruction lines and take the actual task  
                for i, line in enumerate(lines):  
                    if line.strip() and not line.startswith("You are") and not line.startswith("Your task"):  
                        core_instruction = "\n".join(lines[i:]).strip()  
                        break  
                else:  
                    core_instruction = user_part.strip()  
        else:  
            # Ultimate fallback  
            core_instruction = full_prompt
  
    return core_instruction
  
def parse_evaluation_results(outputs, mode, report_detailed=False):  
    """  
    Parse official evaluation results into standardized dictionary format.
  
    Args:  
        outputs: List of OutputExample objects from official IFEval implementation  
        mode: Evaluation mode - "strict" or "loose"  
        report_detailed: If True, include detailed per-item results
  
    Returns:  
        Dictionary with parsed metrics including prompt-level accuracy, instruction-level  
        accuracy, tier-based pass rates, constraint analysis, and optionally detailed results  
    """  
    if not outputs:  
        return get_empty_ifeval_metrics()
  
    # Calculate overall metrics  
    follow_all_instructions = [o.follow_all_instructions for o in outputs]  
    total_items = len(outputs)  
    overall_passes = sum(follow_all_instructions)  
    overall_pass_rate = overall_passes / total_items if total_items > 0 else 0.0
  
    # Calculate instruction-level metrics  
    instruction_total = 0  
    instruction_correct = 0
  
    # Analyze per-constraint performance  
    constraint_results = {}  
    constraint_counts = {}  
    detailed_results = []
  
    # Tier 0 and Tier 1 metrics (matching the official implementation)  
    tier0_total = {}  
    tier0_correct = {}  
    tier1_total = {}  
    tier1_correct = {}
  
    for output in outputs:  
        follow_instruction_list = output.follow_instruction_list  
        instruction_id_list = output.instruction_id_list
  
        # Overall instruction counting  
        instruction_total += len(instruction_id_list)  
        instruction_correct += sum(follow_instruction_list)
  
        # Process each instruction  
        for instruction_id, followed_or_not in zip(instruction_id_list, follow_instruction_list):  
            # Tier 0: instruction type level (before colon)  
            tier0_id = instruction_id.split(":")[0]  
            tier0_total[tier0_id] = tier0_total.get(tier0_id, 0) + 1  
            if followed_or_not:  
                tier0_correct[tier0_id] = tier0_correct.get(tier0_id, 0) + 1
  
            # Tier 1: full instruction level  
            tier1_total[instruction_id] = tier1_total.get(instruction_id, 0) + 1  
            if followed_or_not:  
                tier1_correct[instruction_id] = tier1_correct.get(instruction_id, 0) + 1
  
            # For constraint results (backward compatibility)  
            if instruction_id not in constraint_results:  
                constraint_results[instruction_id] = []  
            constraint_results[instruction_id].append(followed_or_not)
  
        # Detailed per-item results  
        if report_detailed:  
            item_detail = {  
                "prompt": output.prompt,  
                "response": output.response,  
                "overall_pass": output.follow_all_instructions,  
                "instruction_ids": output.instruction_id_list,  
                "instruction_results": output.follow_instruction_list,  
                "constraints_satisfied": sum(output.follow_instruction_list),  
                "total_constraints": len(output.instruction_id_list)  
            }  
            detailed_results.append(item_detail)
  
    # Calculate per-constraint pass rates  
    constraint_pass_rates = {}  
    for constraint_id, results in constraint_results.items():  
        if results:  
            pass_rate = sum(results) / len(results)  
            constraint_pass_rates[constraint_id] = pass_rate  
            constraint_counts[constraint_id] = {  
                "total": len(results),  
                "passed": sum(results),  
                "failed": len(results) - sum(results)  
            }
  
    # Calculate tier-level pass rates  
    tier0_pass_rates = {}  
    for tier0_id in tier0_total:  
        tier0_pass_rates[tier0_id] = tier0_correct.get(tier0_id, 0) / tier0_total[tier0_id]
  
    tier1_pass_rates = {}  
    for tier1_id in tier1_total:  
        tier1_pass_rates[tier1_id] = tier1_correct.get(tier1_id, 0) / tier1_total[tier1_id]
  
    # Calculate instruction-level accuracy  
    instruction_level_accuracy = instruction_correct / instruction_total if instruction_total > 0 else 0.0
  
    return {  
        # Primary metrics (matching official implementation)  
        "prompt_level_accuracy": overall_pass_rate,  
        "instruction_level_accuracy": instruction_level_accuracy,  
        "overall_passes": overall_passes,  
        "total_items": total_items,  
        "total_instructions": instruction_total,  
        "correct_instructions": instruction_correct,
  
        # Tier-level metrics (matching official implementation)  
        "tier0_pass_rates": tier0_pass_rates,  
        "tier1_pass_rates": tier1_pass_rates,  
        "tier0_counts": {id: {"total": tier0_total[id], "correct": tier0_correct.get(id, 0)}  
                        for id in tier0_total},  
        "tier1_counts": {id: {"total": tier1_total[id], "correct": tier1_correct.get(id, 0)}  
                        for id in tier1_total},
  
        # Backward compatibility metrics  
        "constraint_pass_rates": constraint_pass_rates,  
        "constraint_counts": constraint_counts,
  
        # Analysis  
        "most_difficult_constraints": get_most_difficult_constraints(tier1_pass_rates),  
        "easiest_constraints": get_easiest_constraints(tier1_pass_rates),  
        "constraint_frequency": {k: len(v) for k, v in constraint_results.items()},
  
        # Summary statistics  
        "avg_constraints_per_item": instruction_total / total_items if total_items > 0 else 0,  
        "perfect_compliance_rate": overall_pass_rate,  
        "partial_compliance_items": sum(1 for o in outputs  
                                      if not o.follow_all_instructions  
                                      and sum(o.follow_instruction_list) > 0),
  
        # Mode-specific info  
        "evaluation_mode": mode,
  
        # Detailed results  
        "detailed_results": detailed_results if report_detailed else None  
    }
  
def get_most_difficult_constraints(constraint_pass_rates: Dict[str, float]) -> List[tuple]:  
    """  
    Get constraints with lowest pass rates.
      
    Args:  
        constraint_pass_rates: Dictionary mapping constraint IDs to pass rates
          
    Returns:  
        List of (constraint_id, pass_rate) tuples for the 5 most difficult constraints  
    """  
    if not constraint_pass_rates:  
        return []  
    sorted_constraints = sorted(constraint_pass_rates.items(), key=lambda x: x[1])  
    return sorted_constraints[:5]  # Bottom 5
  
def get_easiest_constraints(constraint_pass_rates: Dict[str, float]) -> List[tuple]:  
    """  
    Get constraints with highest pass rates.
      
    Args:  
        constraint_pass_rates: Dictionary mapping constraint IDs to pass rates
          
    Returns:  
        List of (constraint_id, pass_rate) tuples for the 5 easiest constraints  
    """  
    if not constraint_pass_rates:  
        return []  
    sorted_constraints = sorted(constraint_pass_rates.items(), key=lambda x: x[1], reverse=True)  
    return sorted_constraints[:5]  # Top 5
  
def get_empty_ifeval_metrics() -> Dict[str, Any]:  
    """  
    Return empty metrics for edge cases (e.g., empty DataFrame).
      
    Returns:  
        Dictionary with all IFEval metrics set to zero/empty values  
    """  
    return {  
        "prompt_level_accuracy": 0.0,  
        "instruction_level_accuracy": 0.0,  
        "overall_passes": 0,  
        "total_items": 0,  
        "total_instructions": 0,  
        "correct_instructions": 0,  
        "tier0_pass_rates": {},  
        "tier1_pass_rates": {},  
        "tier0_counts": {},  
        "tier1_counts": {},  
        "constraint_pass_rates": {},  
        "constraint_counts": {},  
        "most_difficult_constraints": [],  
        "easiest_constraints": [],  
        "constraint_frequency": {},  
        "avg_constraints_per_item": 0,  
        "perfect_compliance_rate": 0.0,  
        "partial_compliance_items": 0,  
        "evaluation_mode": "unknown",  
        "detailed_results": []  
    }

    
def compute_followbench_accuracy(    
    results_df: pd.DataFrame,    
    judge: str = "gpt-4o-mini@openai",    
    max_tokens: int = 2048,    
    temperature: float = 0.0,    
    debug: bool = False    
) -> Dict[str, Any]:    
    """    
    Compute FollowBench instruction-following accuracy using LLM judge.
        
    This evaluation assesses how well models follow complex instructions with    
    multiple constraints (formatting, word counts, content requirements, etc.).
        
    Args:    
        results_df: DataFrame with 'outputs' (model responses) and 'prompt' (instructions)    
        judge: Judge model ID with backend (e.g., "model@bedrock", "model@gemini", "model@gpt")    
        max_tokens: Maximum tokens for judge response    
        temperature: Temperature for judge (0.0 for deterministic)    
        debug: If True, print debug information during evaluation
        
    Returns:    
        Dictionary containing accuracy metrics (mean/median scores), perfect/good following rates,  
        score distribution by category, and detailed individual scores  
    """    
    outputs = results_df['outputs']    
    prompts = results_df['prompt']
        
    if debug:    
        print(f"Evaluating {len(outputs)} FollowBench responses...")
        
    # Prepare data for LLM judge    
    judge_df = pd.DataFrame({    
        'instruction': prompts,    
        'response': outputs    
    })
        
    # Run LLM judge evaluation    
    judge_scores = llm_judge_eval(    
        judge_df,    
        mode="instruction_following",    
        model_id=judge,    
        max_tokens=max_tokens,    
        temperature=temperature,    
        verbose=debug    
    )
        
    # Calculate metrics    
    valid_scores = [s for s in judge_scores if s is not None]
        
    if len(valid_scores) == 0:    
        return {    
            'accuracy': 0.0,    
            'mean_score': 0.0,    
            'median_score': 0.0,    
            'perfect_following_rate': 0.0,    
            'good_following_rate': 0.0,    
            'total_count': len(outputs),    
            'valid_count': 0,    
            'detailed_scores': judge_scores    
        }
        
    mean_score = sum(valid_scores) / len(valid_scores)    
    sorted_scores = sorted(valid_scores)    
    median_score = sorted_scores[len(sorted_scores) // 2]
        
    # Calculate threshold-based metrics    
    perfect_following = sum(1 for s in valid_scores if s >= 0.9)    
    good_following = sum(1 for s in valid_scores if s >= 0.7)
        
    perfect_following_rate = perfect_following / len(valid_scores)    
    good_following_rate = good_following / len(valid_scores)
        
    # Categorize responses    
    categories = {    
        'perfect': sum(1 for s in valid_scores if s >= 0.9),    
        'very_good': sum(1 for s in valid_scores if 0.8 <= s < 0.9),    
        'good': sum(1 for s in valid_scores if 0.6 <= s < 0.8),    
        'fair': sum(1 for s in valid_scores if 0.4 <= s < 0.6),    
        'poor': sum(1 for s in valid_scores if 0.2 <= s < 0.4),    
        'failed': sum(1 for s in valid_scores if s < 0.2)    
    }
        
    if debug:    
        print(f"\n=== FollowBench Results ===")    
        print(f"Mean Score: {mean_score:.3f}")    
        print(f"Median Score: {median_score:.3f}")    
        print(f"Perfect Following Rate (≥0.9): {perfect_following_rate:.1%}")    
        print(f"Good Following Rate (≥0.7): {good_following_rate:.1%}")    
        print(f"\nScore Distribution:")    
        for category, count in categories.items():    
            print(f"  {category}: {count} ({count/len(valid_scores):.1%})")
        
    return {    
        'accuracy': mean_score,  # Use mean score as overall accuracy    
        'mean_score': mean_score,    
        'median_score': median_score,    
        'perfect_following_rate': perfect_following_rate,    
        'good_following_rate': good_following_rate,    
        'total_count': len(outputs),    
        'valid_count': len(valid_scores),    
        'score_distribution': categories,    
        'detailed_scores': judge_scores    
    }

    
def compute_followbench_accuracy_with_constraints(    
    results_df: pd.DataFrame,    
    judge: str = "gpt-4o-mini@openai",    
    constraint_types: Optional[list] = None,    
    max_tokens: int = 2048,    
    temperature: float = 0.0,    
    debug: bool = False    
) -> Dict[str, Any]:    
    """    
    Enhanced FollowBench evaluation with constraint-specific analysis.
        
    Args:    
        results_df: DataFrame with 'outputs', 'prompt', and optionally 'constraint_type' columns  
        judge: Judge model ID with backend specification  
        constraint_types: List of constraint types to analyze separately    
            (e.g., ['format', 'content', 'length', 'style'])    
        max_tokens: Maximum tokens for judge response  
        temperature: Temperature for judge (0.0 for deterministic)  
        debug: If True, print debug information including per-constraint-type breakdown
        
    Returns:    
        Dictionary with overall metrics plus per-constraint-type breakdown if constraint_types  
        are provided and 'constraint_type' column exists in DataFrame  
    """    
    # Get overall results    
    overall_results = compute_followbench_accuracy(    
        results_df,    
        judge=judge,    
        max_tokens=max_tokens,    
        temperature=temperature,    
        debug=debug    
    )
        
    # If constraint types are provided, analyze by type    
    if constraint_types and 'constraint_type' in results_df.columns:    
        constraint_results = {}
            
        for constraint_type in constraint_types:    
            mask = results_df['constraint_type'] == constraint_type    
            if mask.sum() > 0:    
                subset_df = results_df[mask].reset_index(drop=True)    
                constraint_results[constraint_type] = compute_followbench_accuracy(    
                    subset_df,    
                    judge=judge,    
                    max_tokens=max_tokens,    
                    temperature=temperature,    
                    debug=False    
                )
            
        overall_results['by_constraint_type'] = constraint_results
            
        if debug:    
            print(f"\n=== Results by Constraint Type ===")    
            for ctype, results in constraint_results.items():    
                print(f"{ctype}: {results['mean_score']:.3f} "    
                      f"(n={results['total_count']})")
        
    return overall_results  