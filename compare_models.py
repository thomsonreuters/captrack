"""  
Relative Comparison and Visualization Script for CapTrack

This script computes relative differences between base and adapted models  
and creates heatmap visualizations of capability-level changes.

Usage:  
    python compare_models.py --config config.yaml

Or modify the configuration section below and run:  
    python compare_models.py  
"""

import pandas as pd  
import numpy as np  
import argparse  
import yaml  
from pathlib import Path  
from typing import List, Tuple, Optional, Dict  
import sys

# Import plotting utilities  
from evaluation.plotting_utils import (  
    compute_relative_difference,  
    filter_metrics_by_categories,  
    aggregate_metrics,  
    create_heatmap  
)


# ============================================================================  
# CONFIGURATION  
# ============================================================================

DEFAULT_CONFIG = {  
    # Model pairs to compare: [(base_metrics_path, adapted_metrics_path, comparison_name), ...]  
    "model_pairs": [  
        ("./captrack_results/qwen3-235b-in-oob.csv",   
         "./captrack_results/qwen3-235b-gspo-legal.csv",   
         "Base vs Adapted"),  
    ],  
      
    # Output directory for figures and results  
    "output_dir": "./comparison_results",  
      
    # Categories to include (None = all categories)  
    # Examples: ['C1', 'C2', 'W1'], ['C1', 'C2_DD'], None  
    "categories_to_plot": [
        "C1", "C2", "C3", "C4", "C5a", "C5b", "C5c",
        "W1", "W2", "W3a", "W3b",
        "H1", "H2", "H3", "H4", "H5", "H6"
    ],  
      
    # Whether to aggregate metrics before plotting  
    "aggregate_metrics": True,  
      
    # Plot settings  
    "plot_settings": {  
        "color_cap": 15.0,        # Maximum value for color scale  
        "show_average": True,      # Show average columns for each category  
        "use_green_red": False,    # Use symmetric red-green colormap  
        "figsize": [16, 6],        # Figure size [width, height]  
    },  
      
    # Save options  
    "save_options": {  
        "save_figure": True,  
        "figure_format": "pdf",    # pdf, png, svg  
        "save_csv": True,          # Save processed data as CSV  
        "save_excel": False,       # Save processed data as Excel  
    }  
}


# ============================================================================  
# HELPER FUNCTIONS  
# ============================================================================

def load_metrics_from_file(file_path: str) -> pd.DataFrame:  
    """  
    Load metrics from a CSV file.  
      
    Args:  
        file_path: Path to the metrics CSV file  
          
    Returns:  
        DataFrame with metrics  
    """  
    file_path = Path(file_path)  
      
    if not file_path.exists():  
        raise FileNotFoundError(f"Metrics file not found: {file_path}")  
      
    # Try loading with different index configurations  
    try:  
        # First try with index_col=0 (if saved with index)  
        df = pd.read_csv(file_path, index_col=0)  
    except:  
        # If that fails, try without index  
        df = pd.read_csv(file_path)  
      
    print(f"  Loaded metrics from: {file_path}")  
    print(f"    Shape: {df.shape}")  
      
    return df


def validate_model_pairs(model_pairs: List[Tuple[str, str, str]]) -> None:  
    """  
    Validate that all model pair files exist.  
      
    Args:  
        model_pairs: List of (base_path, adapted_path, name) tuples  
    """  
    missing_files = []  
      
    for base_path, adapted_path, name in model_pairs:  
        if not Path(base_path).exists():  
            missing_files.append(base_path)  
        if not Path(adapted_path).exists():  
            missing_files.append(adapted_path)  
      
    if missing_files:  
        raise FileNotFoundError(  
            f"The following metrics files were not found:\n" +   
            "\n".join(f"  - {f}" for f in missing_files)  
        )


def load_config_from_yaml(config_path: str) -> Dict:  
    """  
    Load configuration from YAML file.  
      
    Args:  
        config_path: Path to YAML configuration file  
          
    Returns:  
        Configuration dictionary  
    """  
    with open(config_path, 'r') as f:  
        config = yaml.safe_load(f)  
      
    # Merge with defaults for any missing keys  
    for key, value in DEFAULT_CONFIG.items():  
        if key not in config:  
            config[key] = value  
        elif isinstance(value, dict):  
            for subkey, subvalue in value.items():  
                if subkey not in config[key]:  
                    config[key][subkey] = subvalue  
      
    return config


def create_example_config(output_path: str = "comparison_config.yaml") -> None:  
    """  
    Create an example configuration file.  
      
    Args:  
        output_path: Path where to save the example config  
    """  
    with open(output_path, 'w') as f:  
        yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)  
      
    print(f"Example configuration saved to: {output_path}")  
    print("Edit this file and run: python compare_models.py --config comparison_config.yaml")


# ============================================================================  
# MAIN COMPARISON PIPELINE  
# ============================================================================

def run_comparison(config: Dict) -> None:  
    """  
    Run the full comparison pipeline.  
      
    Args:  
        config: Configuration dictionary  
    """  
      
    # Extract configuration  
    model_pairs = config["model_pairs"]  
    output_dir = Path(config["output_dir"])  
    categories_to_plot = config["categories_to_plot"]  
    aggregate_metrics_flag = config["aggregate_metrics"]  
    plot_settings = config["plot_settings"]  
    save_options = config["save_options"]  
      
    # Create output directory  
    output_dir.mkdir(parents=True, exist_ok=True)  
      
    print("=" * 80)  
    print("CapTrack Model Comparison Pipeline")  
    print("=" * 80)  
      
    # ========================================================================  
    # 1. VALIDATE FILES  
    # ========================================================================  
      
    print("\n[1/5] Validating input files...")  
    validate_model_pairs(model_pairs)  
    print(f"✓ All {len(model_pairs)} model pair(s) validated")  
      
    # ========================================================================  
    # 2. LOAD DATA AND COMPUTE RELATIVE DIFFERENCES  
    # ========================================================================  
      
    print("\n[2/5] Loading metrics and computing relative differences...")  
      
    results = {}  
    for base_path, adapted_path, comparison_name in model_pairs:  
        print(f"\n  Processing: {comparison_name}")  
          
        # Load metrics  
        base_df = load_metrics_from_file(base_path)  
        adapted_df = load_metrics_from_file(adapted_path)  
          
        # Compute relative differences  
        rel_diff = compute_relative_difference(base_df, adapted_df)  
        results[comparison_name] = rel_diff  
          
        print(f"    ✓ Computed {len(rel_diff)} relative differences")  
      
    # Combine into a DataFrame (rows = model pairs, columns = metrics)  
    results_df = pd.DataFrame(results).T  
      
    print(f"\n✓ Loaded {len(results_df)} model comparison(s) with {len(results_df.columns)} metrics")  
      
    # ========================================================================  
    # 3. FILTER BY CATEGORIES (OPTIONAL)  
    # ========================================================================  
    print(results_df.columns)   
    if categories_to_plot is not None:  
        print(f"\n[3/5] Filtering to categories: {categories_to_plot}")  
        results_df = filter_metrics_by_categories(results_df, categories_to_plot)  
        print(f"✓ Filtered to {len(results_df.columns)} metrics")  
    else:  
        print("\n[3/5] No category filtering applied (using all metrics)")  
      
    # ========================================================================  
    # 4. AGGREGATE METRICS (OPTIONAL)  
    # ========================================================================  
    results_std_df = None  
      
    if aggregate_metrics_flag:  
        print("\n[4/5] Aggregating metrics...")  
        results_df, results_std_df = aggregate_metrics(results_df)  
        print(f"✓ Aggregated to {len(results_df.columns)} metrics")  
    else:  
        print("\n[4/5] No aggregation applied (using raw metrics)")  
      
    # ========================================================================  
    # 5. CREATE VISUALIZATIONS  
    # ========================================================================  
      
    print("\n[5/5] Creating heatmap visualization...")  
      
    # Determine output filename  
    figure_format = save_options.get("figure_format", "pdf")  
    output_filename = f"relative_differences_heatmap.{figure_format}"  
    output_path = output_dir / output_filename  
      
    # Create heatmap  
    fig = create_heatmap(  
        results_df,  
        data_std=results_std_df,  
        output_path=output_path if save_options["save_figure"] else None,  
        color_cap=plot_settings["color_cap"],  
        figsize=tuple(plot_settings["figsize"]),  
        show_average=plot_settings["show_average"],  
        use_green_red=plot_settings["use_green_red"],  
        relative=True  
    )  
      
    if save_options["save_figure"]:  
        print(f"✓ Heatmap saved to: {output_path}")  
      
    # ========================================================================  
    # 6. SAVE PROCESSED DATA  
    # ========================================================================  
      
    print("\n[6/6] Saving processed data...")  
      
    if save_options["save_csv"]:  
        csv_path = output_dir / "relative_differences.csv"  
        results_df.to_csv(csv_path)  
        print(f"✓ Saved relative differences to: {csv_path}")  
          
        if results_std_df is not None:  
            std_csv_path = output_dir / "relative_differences_std.csv"  
            results_std_df.to_csv(std_csv_path)  
            print(f"✓ Saved standard deviations to: {std_csv_path}")  
      
    if save_options.get("save_excel", False):  
        excel_path = output_dir / "relative_differences.xlsx"  
        with pd.ExcelWriter(excel_path) as writer:  
            results_df.to_excel(writer, sheet_name="Relative Differences")  
            if results_std_df is not None:  
                results_std_df.to_excel(writer, sheet_name="Standard Deviations")  
        print(f"✓ Saved Excel file to: {excel_path}")  
      
    # ========================================================================  
    # SUMMARY  
    # ========================================================================  
      
    print("\n" + "=" * 80)  
    print("COMPARISON SUMMARY")  
    print("=" * 80)  
    print(f"Model comparisons: {len(results_df)}")  
    print(f"Metrics analyzed: {len(results_df.columns)}")  
    print(f"Output directory: {output_dir.absolute()}")  
    print("\nKey findings:")  
      
    # Show average changes per comparison  
    for comparison_name in results_df.index:  
        avg_change = results_df.loc[comparison_name].mean()  
        print(f"  {comparison_name}: {avg_change:+.2f}% average change")  
      
    # Show most affected metrics  
    print("\nTop 5 most degraded metrics (averaged across comparisons):")  
    avg_by_metric = results_df.mean(axis=0).sort_values()  
    for metric, value in avg_by_metric.head(5).items():  
        print(f"  {metric}: {value:.2f}%")  
      
    print("\nTop 5 most improved metrics (averaged across comparisons):")  
    for metric, value in avg_by_metric.tail(5).items():  
        print(f"  {metric}: {value:+.2f}%")  
      
    print("\n" + "=" * 80)  
    print("✓ Comparison pipeline completed successfully!")  
    print("=" * 80)  
      
    # Display the plot  
    import matplotlib.pyplot as plt  
    plt.show()


# ============================================================================  
# COMMAND-LINE INTERFACE  
# ============================================================================

def main():  
    """Main entry point for the script."""  
      
    parser = argparse.ArgumentParser(  
        description="Compare CapTrack metrics between base and adapted models",  
        formatter_class=argparse.RawDescriptionHelpFormatter,  
        epilog="""  
Examples:  
  # Run with default configuration (edit the script first)  
  python compare_models.py  
    
  # Run with a YAML configuration file  
  python compare_models.py --config my_comparison.yaml  
    
  # Generate an example configuration file  
  python compare_models.py --create-config  
    
  # Specify model pairs directly  
  python compare_models.py \\  
    --base ./results/base_metrics.csv \\  
    --adapted ./results/adapted_metrics.csv \\  
    --name "My Comparison" \\  
    --output ./figures  
        """  
    )  
      
    parser.add_argument(  
        "--config",  
        type=str,  
        help="Path to YAML configuration file"  
    )  
      
    parser.add_argument(  
        "--create-config",  
        action="store_true",  
        help="Create an example configuration file and exit"  
    )  
      
    parser.add_argument(  
        "--base",  
        type=str,  
        help="Path to base model metrics CSV"  
    )  
      
    parser.add_argument(  
        "--adapted",  
        type=str,  
        help="Path to adapted model metrics CSV"  
    )  
      
    parser.add_argument(  
        "--name",  
        type=str,  
        default="Comparison",  
        help="Name for this comparison"  
    )  
      
    parser.add_argument(  
        "--output",  
        type=str,  
        default="./comparison_results",  
        help="Output directory for results"  
    )  
      
    parser.add_argument(  
        "--no-aggregate",  
        action="store_true",  
        help="Disable metric aggregation"  
    )  
      
    parser.add_argument(  
        "--categories",  
        type=str,  
        nargs="+",  
        help="Categories to include (e.g., C1 C2 W1)"  
    )  
      
    parser.add_argument(  
        "--color-cap",  
        type=float,  
        default=10.0,  
        help="Maximum value for color scale"  
    )  
      
    args = parser.parse_args()  
      
    # Handle --create-config  
    if args.create_config:  
        create_example_config()  
        return  
      
    # Load or create configuration  
    if args.config:  
        print(f"Loading configuration from: {args.config}")  
        config = load_config_from_yaml(args.config)  
    elif args.base and args.adapted:  
        # Create config from command-line arguments  
        config = DEFAULT_CONFIG.copy()  
        config["model_pairs"] = [(args.base, args.adapted, args.name)]  
        config["output_dir"] = args.output  
        config["aggregate_metrics"] = not args.no_aggregate  
        if args.categories:  
            config["categories_to_plot"] = args.categories  
        config["plot_settings"]["color_cap"] = args.color_cap  
    else:  
        # Use default configuration from script  
        print("Using default configuration from script")  
        print("(Edit the DEFAULT_CONFIG in the script or use --config)")  
        config = DEFAULT_CONFIG  
      
    # Run the comparison pipeline  
    try:  
        run_comparison(config)  
    except Exception as e:  
        print(f"\n❌ Error: {e}", file=sys.stderr)  
        import traceback  
        traceback.print_exc()  
        sys.exit(1)


if __name__ == "__main__":  
    main()  