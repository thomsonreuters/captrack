"""  
Captrack Benchmark Metrics Computation

A standalone evaluation suite for computing metrics across multiple AI benchmark tasks.  
This script processes model outputs from various inference pipelines and computes  
standardized metrics for assessing language model capabilities.

Supports multiple input formats (JSON, JSONL, CSV, Parquet) and flexible column mapping.  
"""

import json  
import argparse  
import pandas as pd  
from pathlib import Path  
from typing import Dict, Any, List, Optional, Union  
from datetime import datetime  
import logging

from evaluation import *  
from evaluation.taxonomy_cfg import TAXONOMY_CONFIG

# Configure logging  
logging.basicConfig(  
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s'  
)  
logger = logging.getLogger(__name__)


# Mapping of task identifiers to their corresponding metric computation functions  
TASK_TO_METRIC = {  
    "mmlu_pro.subset": compute_mc_accuracy,  
    "mmlu_pro.rephrased": compute_mc_accuracy,  
    "mmlu_pro.schema": json_schema_score_detailed,  
    "mmlu_pro.table_schema": json_schema_score_detailed,  
    "gsm8k.subset": compute_gsm8k_accuracy,  
    "gsm8k.rephrased": compute_gsm8k_accuracy,  
    "gsm8k.schema": json_schema_score_detailed,  
    "gsm8k.table_schema": json_schema_score_detailed,  
    "mgsm.subset": compute_gsm8k_accuracy,  
    "humaneval.full": compute_humaneval_accuracy,  
    "math.subset": compute_math_supergpqa_accuracy,  
    "supergpqa.subset": compute_math_supergpqa_accuracy,  
    "hotpotqa.subset": compute_hotpotqa_boolq_accuracy,  
    "hotpotqa.citation": compute_citation_accuracy,  
    "boolq.subset": compute_hotpotqa_boolq_accuracy,  
    "ragtruth.subset": compute_inf_scope_metrics,  
    "winogrande.subset": compute_mc_accuracy,  
    "hellaswag.subset": compute_mc_accuracy,  
    "ifeval.subset": compute_ifeval_accuracy,  
    "bfcl.subset": compute_bfcl_accuracy,  
    "ruler.4k": compute_uncertainty_calibration_accuracy,  
    "ruler.32k": compute_ruler_accuracy,  
    "mtbench.turn1": compute_style_elaboration_metrics,  
    "mtbench.turn2": compute_multi_turn_metrics,  
    "harmbench.subset": compute_risk_adjusted_utility,  
    "popqa.subset": compute_open_accuracy,  
    "livemathbench.full": compute_livemathbench_accuracy,  
    "mbpp.full": compute_humaneval_accuracy,  
    "truthfulqa.full": compute_mc_accuracy,  
    "xtreme.subset": compute_open_accuracy,  
    "eli5.subset": compute_inf_scope_metrics,  
    "oasst1.full": compute_style_elaboration_metrics,  
    "followbench.subset": compute_followbench_accuracy,  
    "mnms.full": compute_bfcl_accuracy,  
    "structflowbench.turn2": compute_multi_turn_metrics,  
    "longbenchv2.full": compute_mc_accuracy,  
    "qasper.citation": compute_citation_accuracy,  
}


# Task-specific configurations  
TASK_CONFIGS = {  
    "mmlu_pro.subset": {"judge": "gpt-4o-mini@openai"},  
    "mmlu_pro.rephrased": {"judge": "gpt-4o-mini@openai"},  
    "mmlu_pro.schema": {},  
    "mmlu_pro.table_schema": {},  
    "gsm8k.subset": {},  
    "gsm8k.rephrased": {},  
    "gsm8k.schema": {},  
    "gsm8k.table_schema": {},  
    "humaneval.full": {},  
    "math.subset": {"dataset_type": "MATH", "judge": "gpt-4o-mini@openai"},  
    "supergpqa.subset": {"dataset_type": "SuperGPQA", "judge": "gpt-4o-mini@openai"},  
    "hotpotqa.subset": {"dataset_type": "HOTPOTQA"},  
    "boolq.subset": {"dataset_type": "BOOLQ"},  
    "ragtruth.subset": {"judge": "gpt-4o-mini@openai"},  
    "winogrande.subset": {"judge": "gpt-4o-mini@openai"},  
    "hellaswag.subset": {"judge": "gpt-4o-mini@openai"},  
    "mgsm.subset": {},  
    "ifeval.subset": {},  
    "bfcl.subset": {},  
    "ruler.4k": {},  
    "ruler.32k": {},  
    "mtbench.turn1": {},  
    "mtbench.turn2": {"judge": "gpt-4o-mini@openai"},  
    "harmbench.subset": {"judge": "gpt-4.1-mini@openai"},  
    "hotpotqa.citation": {},  
    "popqa.subset": {},  
    "livemathbench.full": {"judge": "gpt-4o-mini@openai"},  
    "mbpp.full": {},  
    "truthfulqa.full": {"judge": "gpt-4o-mini@openai"},  
    "xtreme.subset": {},  
    "eli5.subset": {"judge": "gpt-4o-mini@openai", "has_context": False},  
    "oasst1.full": {},  
    "followbench.subset": {"judge": "gpt-4o-mini@openai"},  
    "mnms.full": {"dataset_format": 'mnms'},  
    "structflowbench.turn2": {"judge": "gpt-4o-mini@openai"},  
    "longbenchv2.full": {"judge": "gpt-4o-mini@openai"},  
    "qasper.citation": {},  
}


def load_data_file(  
    file_path: Union[str, Path],  
    file_format: Optional[str] = None  
) -> pd.DataFrame:  
    """  
    Load data from various file formats into a pandas DataFrame.  
      
    Args:  
        file_path: Path to the data file  
        file_format: Optional format specification ('json', 'jsonl', 'csv', 'parquet').  
                    If None, infers from file extension.  
      
    Returns:  
        DataFrame containing the loaded data  
          
    Raises:  
        ValueError: If file format is unsupported or cannot be inferred  
        FileNotFoundError: If file does not exist  
    """  
    file_path = Path(file_path)  
      
    if not file_path.exists():  
        raise FileNotFoundError(f"File not found: {file_path}")  
      
    # Infer format from extension if not provided  
    if file_format is None:  
        extension = file_path.suffix.lower()  
        format_map = {  
            '.json': 'json',  
            '.jsonl': 'jsonl',  
            '.csv': 'csv',  
            '.parquet': 'parquet',  
            '.pq': 'parquet'  
        }  
        file_format = format_map.get(extension)  
          
        if file_format is None:  
            raise ValueError(  
                f"Cannot infer format from extension '{extension}'. "  
                f"Please specify format explicitly using --file_format"  
            )  
      
    logger.info(f"Loading {file_format.upper()} file: {file_path}")  
      
    try:  
        if file_format == 'jsonl':  
            df = pd.read_json(file_path, orient='records', lines=True)  
        elif file_format == 'json':  
            df = pd.read_json(file_path, orient='records')  
        elif file_format == 'csv':  
            df = pd.read_csv(file_path)  
        elif file_format == 'parquet':  
            df = pd.read_parquet(file_path)  
        else:  
            raise ValueError(f"Unsupported file format: {file_format}")  
              
        logger.info(f"Successfully loaded {len(df)} records")  
        return df  
          
    except Exception as e:  
        raise ValueError(f"Error loading file {file_path}: {str(e)}")


def normalize_answer(text: Any) -> str:  
    """  
    Normalize text for comparison (imported from metrics_common_functions).  
      
    This is a placeholder - the actual implementation should be imported  
    from evaluation.metrics.util.metrics_common_functions  
    """  
    if text is None:  
        return ""  
    return str(text).strip()


def transform_dataframe(  
    df: pd.DataFrame,  
    output_column: str = "outputs",  
    gold_column: str = "gold",  
    task_column: Optional[str] = None,  
    task_name: Optional[str] = None  
) -> pd.DataFrame:  
    """  
    Transform input dataframe to standardized format expected by metric functions.  
      
    Args:  
        df: Input dataframe with model outputs  
        output_column: Name of column containing model responses  
        gold_column: Name of column containing ground truth answers  
        task_column: Optional column name containing task identifiers  
        task_name: Optional task name to filter by (if task_column is provided)  
      
    Returns:  
        Transformed dataframe with standardized columns:  
        - 'outputs': model responses  
        - 'gold': ground truth answers  
        - 'pp-outputs': preprocessed/normalized outputs  
        - 'pp-gold': preprocessed/normalized gold answers  
          
    Raises:  
        ValueError: If required columns are missing  
    """  
    # Validate required columns exist  
    if output_column not in df.columns:  
        raise ValueError(  
            f"Output column '{output_column}' not found in dataframe. "  
            f"Available columns: {list(df.columns)}"  
        )  
      
    if gold_column not in df.columns:  
        raise ValueError(  
            f"Gold column '{gold_column}' not found in dataframe. "  
            f"Available columns: {list(df.columns)}"  
        )  
      
    # Filter by task if specified  
    if task_column and task_name:  
        if task_column not in df.columns:  
            raise ValueError(f"Task column '{task_column}' not found in dataframe")  
        df = df[df[task_column] == task_name].copy()  
        logger.info(f"Filtered to {len(df)} records for task '{task_name}'")  
      
    # Create standardized dataframe  
    result_df = pd.DataFrame()  
    result_df['outputs'] = df[output_column]  
    result_df['gold'] = df[gold_column]  
      
    # Add normalized versions for comparison  
    result_df['pp-outputs'] = result_df['outputs'].apply(normalize_answer)  
    result_df['pp-gold'] = result_df['gold'].apply(normalize_answer)  
      
    # Preserve any additional columns that might be needed  
    for col in df.columns:  
        if col not in [output_column, gold_column] and col not in result_df.columns:  
            result_df[col] = df[col].values  
      
    return result_df


def load_task_data(  
    data_dir: Union[str, Path],  
    task: str,  
    model_name: str,
    output_column: str = "outputs",  
    gold_column: str = "gold",  
    file_format: Optional[str] = None,  
    task_column: Optional[str] = None  
) -> Optional[pd.DataFrame]:  
    """  
    Load and transform data for a specific task.  
      
    Args:  
        data_dir: Directory containing task data files  
        task: Task identifier (e.g., "gsm8k.core")  
        output_column: Column name for model outputs  
        gold_column: Column name for ground truth  
        file_format: File format override  
        task_column: Column containing task identifiers (for multi-task files)  
      
    Returns:  
        Transformed dataframe ready for metric computation, or None if file not found  
    """  
    data_dir = Path(data_dir) / model_name 
      
    # Try to find file matching task name  
    task_base = task.replace('.', '_')  
    possible_extensions = ['.jsonl', '.json', '.csv', '.parquet', '.pq']  
      
    file_path = None  
    for ext in possible_extensions:  
        candidate = data_dir / f"{task_base}{ext}"  
        if candidate.exists():  
            file_path = candidate  
            break  
      
    # If no exact match, try without variant suffix (e.g., "gsm8k" instead of "gsm8k_subset")  
    if file_path is None:  
        task_name_only = task.split('.')[0]  
        for ext in possible_extensions:  
            candidate = data_dir / f"{task_name_only}{ext}"  
            if candidate.exists():  
                file_path = candidate  
                break  
      
    if file_path is None:  
        logger.warning(f"No data file found for task '{task}' in {data_dir}")  
        return None  
      
    try:  
        df = load_data_file(file_path, file_format)  
          
        # Transform to standardized format  
        transformed_df = transform_dataframe(  
            df,  
            output_column=output_column,  
            gold_column=gold_column,  
            task_column=task_column,  
            task_name=task if task_column else None  
        )  
          
        return transformed_df  
          
    except Exception as e:  
        logger.error(f"Error loading task '{task}': {str(e)}")  
        return None


def compute_captrack_metrics(  
    data_dir: Union[str, Path],  
    output_column: str = "outputs",  
    gold_column: str = "gold",  
    file_format: Optional[str] = None,  
    task_column: Optional[str] = None,  
    tasks: Optional[List[str]] = None,  
    model_name: str = "model"  
) -> Dict[str, Any]:  
    """  
    Compute Captrack metrics for all configured benchmark tasks.  
      
    Args:  
        data_dir: Directory containing task data files  
        output_column: Column name containing model responses  
        gold_column: Column name containing ground truth answers  
        file_format: Optional file format override ('json', 'jsonl', 'csv', 'parquet')  
        task_column: Optional column name containing task identifiers  
        tasks: Optional list of specific tasks to evaluate. If None, evaluates all tasks.  
        model_name: Name of the model being evaluated  
      
    Returns:  
        Dictionary mapping task names to their computed metrics  
    """  
    data_dir = Path(data_dir)  
      
    if not data_dir.exists():  
        raise FileNotFoundError(f"Data directory not found: {data_dir}")  
      
    # Determine which tasks to evaluate  
    tasks_to_eval = tasks if tasks else list(TASK_CONFIGS.keys())  
      
    logger.info(f"Computing metrics for {len(tasks_to_eval)} tasks")  
    logger.info(f"Data directory: {data_dir}")  
    logger.info(f"Output column: '{output_column}', Gold column: '{gold_column}'")  
      
    captrack_metrics = {}  
    successful_tasks = 0  
    failed_tasks = 0  
      
    for task in tasks_to_eval:  
        if task not in TASK_TO_METRIC:  
            logger.warning(f"Unknown task '{task}' - skipping")  
            continue  
          
        try:  
            logger.info(f"\n{'='*60}")  
            logger.info(f"Processing task: {task}")  
            logger.info(f"{'='*60}")  
              
            # Load task data  
            results = load_task_data(  
                data_dir,  
                task,  
                model_name,
                output_column=output_column,  
                gold_column=gold_column,  
                file_format=file_format,  
                task_column=task_column  
            )  
              
            if results is None or len(results) == 0:  
                logger.warning(f"No data available for task '{task}' - skipping")  
                failed_tasks += 1  
                continue  
              
            logger.info(f"Loaded {len(results)} examples for task '{task}'")  
              
            # Get metric function and configuration
            metric_func = TASK_TO_METRIC[task]
            kwargs = TASK_CONFIGS.get(task, {})
              
            # Special handling for harmbench which requires benign baseline
            if task == "harmbench.subset":
                results_benign = load_task_data(
                    data_dir,
                    "gsm8k.subset",
                    model_name,
                    output_column=output_column,
                    gold_column=gold_column,
                    file_format=file_format,
                    task_column=task_column
                )  
                if results_benign is None:  
                    logger.error("harmbench.subset requires gsm8k.subset data as benign baseline")  
                    failed_tasks += 1  
                    continue  
                captrack_metrics[task] = metric_func(results_benign, results, **kwargs)  
            else:  
                captrack_metrics[task] = metric_func(results, **kwargs)  
              
            logger.info(f"✓ Successfully computed metrics for '{task}'")  
            successful_tasks += 1  
              
        except Exception as e:  
            logger.error(f"✗ Error processing task '{task}': {str(e)}", exc_info=True)  
            failed_tasks += 1  
      
    logger.info(f"\n{'='*60}")  
    logger.info(f"Metrics computation complete")  
    logger.info(f"Successful: {successful_tasks}, Failed: {failed_tasks}")  
    logger.info(f"{'='*60}\n")  
      
    return captrack_metrics


def extract_metrics_to_dataframe(      
    data: Dict[str, Any],      
    model_name: str = "model"  
) -> pd.DataFrame:      
    """      
    Extract metrics from a nested dictionary based on TAXONOMY_CONFIG.      
    Returns non-aggregated metrics.  
          
    Args:      
        data: Dictionary with structure {task_name: {metric_name: value, ...}, ...}      
        model_name: Name/identifier for the model (used as index)      
          
    Returns:      
        DataFrame with one row per model and columns for each metric  
    """      
      
    metrics_dict = {}
    none_cols = []
    taxonomy_config = TAXONOMY_CONFIG    
          
    # Iterate through each subcategory in the taxonomy      
    for subcategory, config in taxonomy_config.items():      
        benchmarks = config["benchmarks"]      
              
        # Iterate through each benchmark configuration      
        for benchmark in benchmarks:      
            task_name = benchmark.task_name      
            metric_name = benchmark.metric_name      
            display_name = benchmark.display_name      
                  
            # Create column name: "{subcategory} {display_name}"      
            column_name = f"{subcategory} {display_name}"      
                  
            # Extract the metric value from the data dictionary      
            value = None      
            if task_name in data:      
                task_data = data[task_name]      
                      
                # Handle nested metric paths  
                if isinstance(metric_name, list):      
                    value = task_data      
                    for key in metric_name:      
                        if isinstance(value, dict) and key in value:      
                            value = value[key]      
                        else:      
                            value = None      
                            break      
                # Handle simple metric names  
                else:      
                    value = task_data.get(metric_name, None)      
                  
            # Store the value (will be NaN if None)  
            if value is None:
                none_cols.append(f'{task_name} - {metric_name}')
            metrics_dict[column_name] = value if value is not None else 0.0    

    logger.info(f"{'='*60}")
    logger.info(f"Extracted metrics for model '{model_name}':")
    if none_cols:
        logger.warning(f"Metrics with None values (set to 0.0):")
        for col in none_cols:
            logger.warning(f" - {col}")

    # Create DataFrame with one row      
    df = pd.DataFrame([metrics_dict], index=[model_name])      
        
    return df  


def save_metrics(  
    metrics: Dict[str, Any],  
    output_dir: Union[str, Path],  
    model_name: str,  
    save_json: bool = True,  
    save_csv: bool = True
):  
    """  
    Save computed metrics to disk in various formats.  
      
    Args:  
        metrics: Dictionary of computed metrics from compute_captrack_metrics()  
        output_dir: Directory to save output files  
        model_name: Name of the model (used in filenames)  
        save_json: Whether to save raw metrics as JSON  
        save_csv: Whether to save metrics as CSV  
        aggregate_metrics: Whether to compute and save aggregated metrics  
    """  
    output_dir = Path(output_dir)  
    output_dir.mkdir(parents=True, exist_ok=True)  
      
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
    base_filename = f"{model_name}_{timestamp}"  
      
    # Save raw metrics as JSON  
    if save_json:  
        json_path = output_dir / f"{base_filename}_raw_metrics.json"  
        with open(json_path, 'w') as f:  
            json.dump(metrics, f, indent=2)  
        logger.info(f"✓ Saved raw metrics to: {json_path}")  
      
    # Save as CSV using existing dataframe extraction function  
    if save_csv:           
        # Save detailed metrics  
        df_detailed = extract_metrics_to_dataframe(  
            metrics,  
            model_name=model_name
        )  
        csv_path = output_dir / f"{base_filename}_metrics.csv"  
        df_detailed.to_csv(csv_path)  
        logger.info(f"✓ Saved metrics to: {csv_path}") 


def main():  
    """  
    Main entry point for standalone Captrack metrics computation.  
    """  
    parser = argparse.ArgumentParser(  
        description="Compute Captrack benchmark metrics from model outputs",  
        formatter_class=argparse.RawDescriptionHelpFormatter,  
        epilog="""  
            Examples:  
              # Basic usage with JSONL files  
              python main.py --model_name my_model
                
              # Specify custom column names  
              python main.py --model_name my_model --output_column response --gold_column answer
                
              # Process CSV files with specific tasks  
              python main.py --model_name my_model --file_format csv --tasks gsm8k.core math.subset
                
              # Multi-task file with task identifier column  
              python main.py --model_name my_model --task_column task_name --output_column model_output
                
              # Save aggregated metrics  
              python main.py --model_name gpt4
        """  
    )  
      
    # Required arguments  
    parser.add_argument(  
        "--model_name",  
        type=str,  
        required=True,  
        help="Name of the model being evaluated (used in output filenames)"  
    ) 
    
    # Optional arguments  
    parser.add_argument(  
        "--data_dir",  
        type=str,  
        default="./model_responses",
        help="Directory containing model output files for each task (default: ./model_responses)"
    )  
      
    parser.add_argument(  
        "--output_dir",  
        type=str,  
        default="./captrack_results",  
        help="Directory to save computed metrics (default: ./captrack_results)"  
    )      
      
    parser.add_argument(  
        "--output_column",  
        type=str,  
        default="outputs",  
        help="Column name containing model responses (default: outputs)"  
    )  
      
    parser.add_argument(  
        "--gold_column",  
        type=str,  
        default="gold",  
        help="Column name containing ground truth answers (default: gold)"  
    )  
      
    parser.add_argument(  
        "--file_format",  
        type=str,  
        choices=['json', 'jsonl', 'csv', 'parquet'],  
        default=None,  
        help="File format override (auto-detected from extension if not specified)"  
    )  
      
    parser.add_argument(  
        "--task_column",  
        type=str,  
        default=None,  
        help="Column name containing task identifiers (for multi-task files)"  
    )  
      
    parser.add_argument(  
        "--tasks",  
        nargs='+',  
        default=None,  
        help="Specific tasks to evaluate (default: all tasks). Example: gsm8k.core math.subset"  
    )  
      
    parser.add_argument(  
        "--save_json",  
        action="store_true",  
        default=True,  
        help="Save raw metrics as JSON (default: True)"  
    )  
      
    parser.add_argument(  
        "--save_csv",  
        action="store_true",  
        default=True,  
        help="Save metrics as CSV (default: True)"  
    )
      
    parser.add_argument(  
        "--log_level",  
        type=str,  
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],  
        default='INFO',  
        help="Logging level (default: INFO)"  
    )  
      
    args = parser.parse_args()  
      
    # Configure logging level  
    logging.getLogger().setLevel(getattr(logging, args.log_level))  
      
    # Print configuration  
    logger.info("="*60)  
    logger.info("Captrack Metrics Computation")  
    logger.info("="*60)  
    logger.info(f"Model: {args.model_name}")  
    logger.info(f"Data directory: {args.data_dir}")  
    logger.info(f"Output directory: {args.output_dir}")  
    logger.info(f"Output column: '{args.output_column}'")  
    logger.info(f"Gold column: '{args.gold_column}'")  
    if args.file_format:  
        logger.info(f"File format: {args.file_format}")  
    if args.task_column:  
        logger.info(f"Task column: '{args.task_column}'")  
    if args.tasks:  
        logger.info(f"Tasks to evaluate: {', '.join(args.tasks)}")  
    logger.info("="*60 + "\n")  
      
    try:  
        # Compute metrics  
        metrics = compute_captrack_metrics(  
            data_dir=args.data_dir,  
            output_column=args.output_column,  
            gold_column=args.gold_column,  
            file_format=args.file_format,  
            task_column=args.task_column,  
            tasks=args.tasks,  
            model_name=args.model_name  
        )  
          
        # Save results  
        save_metrics(  
            metrics=metrics,  
            output_dir=args.output_dir,  
            model_name=args.model_name,  
            save_json=args.save_json,  
            save_csv=args.save_csv
        )  
          
        logger.info("\n" + "="*60)  
        logger.info("✓ Metrics computation completed successfully!")  
        logger.info(f"Results saved to: {args.output_dir}")  
        logger.info("="*60)  
          
    except Exception as e:  
        logger.error(f"\n✗ Metrics computation failed: {str(e)}", exc_info=True)  
        return 1  
      
    return 0


if __name__ == "__main__":  
    exit(main())  