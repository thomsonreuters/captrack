"""  
Utility functions for plotting Captrack benchmark results.

This module provides functions for loading metrics, computing relative differences,  
aggregating results, and creating heatmaps for model comparisons.  
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union


BASE_MODEL_LOOKUP = {
    "qwen-3-4b": ["e52677747d844a12b9e23656b8a973cb"],
    "qwen-3-30b": ["46a872bba99641f494332690271d2be5", "30c36111827145cfa0faf492e9d8c1f5", "d652fc6595964907b77055386fed2c26"],
    "qwen-3-80b": ["5dda65fa1d984a1daeb19aac6bbb839d", "63ade8f6147f4c9ca7a444c0f5727570", "cc63c487fad94ab6b0b7f6c25566d83b"],
    "qwen-3-235b": ["c6cf8b9edf984f1eb742be01ee626367", "795a0a272f87434f8a05acdbfedbddf9", "195c40eb79924741b3526a79da636232"]
}


# Aggregation rules mapping (subcategory, metric) -> list of source metrics  
AGGREGATION_RULES = {  
    ("C1", "Accuracy (MMLU-Pro)"): [  
        "Accuracy (MMLU-Pro)", "Accuracy (GSM8K)", "Accuracy (HumanEval)",   
        "Accuracy (PopQA)", "Accuracy (livemathbench)", "Accuracy (mbpp)"  
    ],  
    ("C2", "Accuracy (MATH)"): [  
        "Accuracy (MATH)", "Accuracy (SuperGPQA)",   
        "Reasoning (MATH)", "Reasoning (SuperGPQA)"  
    ],  
    ("C2_DD", "Accuracy (MATH)"): [  
        "Accuracy (MATH)", "Accuracy (SuperGPQA)"  
    ],  
    ("C2_DD", "Reasoning (MATH)"): [  
        "Reasoning (MATH)", "Reasoning (SuperGPQA)"  
    ],  
    ("C2_DD", "Step Validity (MATH)"): [  
        "Step Validity (MATH)", "Step Validity (SuperGPQA)"  
    ],  
    ("C2_DD", "Logical Coherence (MATH)"): [  
        "Logical Coherence (MATH)", "Logical Coherence (SuperGPQA)"  
    ],  
    ("C2_DD", "Step Consistency (MATH)"): [  
        "Step Consistency (MATH)", "Step Consistency (SuperGPQA)"  
    ],  
    ("C2_DD", "N Steps (MATH)"): [  
        "N Steps (MATH)", "N Steps (SuperGPQA)"  
    ],  
    ("C3", "Accuracy (HotpotQA)"): [  
        "Accuracy (HotpotQA)", "Accuracy (BoolQ)",   
        "Evidence Hit (HotpotQA)", "Evidence Hit (BoolQ)"  
    ],  
    ("C4", "Faithfulness (RAGTruth)"): [  
        "Faithfulness (RAGTruth)", "Accuracy (TruthfulQA)"  
    ],  
    ("C5a", "Rephrased (MMLU-Pro)"): [  
        "Rephrased (MMLU-Pro)", "Rephrased (GSM8K)"  
    ],  
    ("C5b", "Domain Shift (WinoGrande)"): [  
        "Domain Shift (WinoGrande)", "Domain Shift (HellaSwag)"  
    ],  
    ("C5c", "Multilingual (MGSM)"): [  
        "Multilingual (MGSM)", "Multilingual (XTREME)"  
    ], 
    ("W1", "Unsafe Refusal"): [  
        "Unsafe Refusal"  
    ],  
    ("W1", "Compliance Rate"): [  
        "Compliance Rate"  
    ], 
    ("W2", "Coverage (RAGTruth)"): [  
        "Coverage (RAGTruth)", "Coverage (ELI5)"  
    ],  
    ("W2", "Overreach (RAGTruth)"): [  
        "Overreach (RAGTruth)", "Overreach (ELI5)"  
    ],  
    ("W3a", "Verbosity (MTBenchT1)"): [  
        "Verbosity (MTBenchT1)", "Verbosity (OASST1)"  
    ],  
    ("W3a_DD", "Verbosity (MTBenchT1)"): [  
        "Verbosity (MTBenchT1)", "Verbosity (OASST1)"  
    ],  
    ("W3a_DD", "Verbosity (std response len)"): [  
        "Verbosity (std response len)", "Verbosity (std response len) (OASST1)"  
    ],  
    ("W3a_DD", "Verbosity (avg sentence len)"): [  
        "Verbosity (avg sentence len)", "Verbosity (avg sentence len) (OASST1)"  
    ],  
    ("W3a_DD", "Verbosity (total_sentences)"): [  
        "Verbosity (total_sentences)", "Verbosity (total_sentences) (OASST1)"  
    ],  
    ("W3a_DD", "Hedging"): [  
        "Hedging", "Hedging (OASST1)"  
    ],  
    ("W3a_DD", "Directness"): [  
        "Directness", "Directness (OASST1)"  
    ],  
    ("W3b", "Formatting (Bullet Usage)"): [  
        "Formatting (Bullet Usage)", "Formatting (Table Usage)",   
        "Formatting (Emoji Usage)", "Formatting (Bullet Usage) (OASST1)",   
        "Formatting (Table Usage) (OASST1)", "Formatting (Emoji Usage) (OASST1)"  
    ],  
    ("W3b_DD", "Formatting (Bullet Usage)"): [  
        "Formatting (Bullet Usage)", "Formatting (Bullet Usage) (OASST1)"  
    ],  
    ("W3b_DD", "Formatting (Table Usage)"): [  
        "Formatting (Table Usage)", "Formatting (Table Usage) (OASST1)"  
    ],  
    ("W3b_DD", "Formatting (Emoji Usage)"): [  
        "Formatting (Emoji Usage)", "Formatting (Emoji Usage) (OASST1)"  
    ],  
    ("H1", "Pass Rate (IFEval)"): [  
        "Pass Rate (IFEval)", "Accuracy (FollowBench)"  
    ],  
    ("H2", "Format (MMLU-Pro)"): [  
        "Format (MMLU-Pro)", "Format (GSM8K)",   
        "Format Table (MMLU-Pro)", "Format Table (GSM8K)"  
    ],  
    ("H3", "Selection (BFCL)"): [  
        "Selection (BFCL)", "Arguments (BFCL)",   
        "Selection (MNMS)", "Arguments (MNMS)"  
    ],  
    ("H4", "Turn 2 Rating (MT-Bench)"): [  
        "Turn 2 Rating (MT-Bench)", "Turn 2 Rating (StructFlowBench)"  
    ],  
    ("H5", "Accuracy (RULER 32k)"): [  
        "Accuracy (RULER 32k)", "Accuracy (LonvbenchV2)"  
    ],  
    ("H6", "Format Accuracy (HotpotQA)"): [  
        "Format Accuracy (HotpotQA)", "Source Accuracy (HotpotQA)",   
        "Format Accuracy (QASPER)", "Source Accuracy (QASPER)"  
    ],  
    ("H6_DD", "Format Accuracy (HotpotQA)"): [  
        "Format Accuracy (HotpotQA)", "Format Accuracy (QASPER)"  
    ],  
    ("H6_DD", "Source Accuracy (HotpotQA)"): [  
        "Source Accuracy (HotpotQA)", "Source Accuracy (QASPER)"  
    ],  
    ("H6_DD", "Citation Usage (HotpotQA)"): [  
        "Citation Usage (HotpotQA)", "Citation Usage (QASPER)"  
    ],  
}

# Display names for aggregated metrics  
DISPLAY_NAMES = {  
    "C1_Accuracy (MMLU-Pro)": "C1 Knowledge",  
    "C2_Accuracy (MATH)": "C2 Reasoning",  
    "C2_Reasoning (MATH)": "C2_DD Reasoning Quality",  
    "C2_N Steps (MATH)": "C2_DD No. Reasoning Steps",  
    "C2_DD_Accuracy (MATH)": "C2_DD Reasoning Task Acc.",  
    "C2_DD_Reasoning (MATH)": "C2_DD Reasoning Score",  
    "C2_DD_Step Validity (MATH)": "C2_DD Step Validity",  
    "C2_DD_Logical Coherence (MATH)": "C2_DD Logical Coherence",  
    "C2_DD_Step Consistency (MATH)": "C2_DD Step Consistency",  
    "C2_DD_N Steps (MATH)": "C2_DD No. Reasoning Steps",  
    "C3_Accuracy (HotpotQA)": "C3 ICL Score",  
    "C4_Faithfulness (RAGTruth)": "C4 Faithfulness",  
    "C5a_Rephrased (MMLU-Pro)": "C5a Prompt Robust.",  
    "C5b_Domain Shift (WinoGrande)": "C5b Domain Robust.",  
    "C5c_Multilingual (MGSM)": "C5c Multilingual",  
    "W1_Benign Compliance": "W1 Benign Compliance",  
    "W1_Unsafe Refusal": "W1 Unsafe Refusal",  
    "W1_Compliance Rate": "W1 Underspec. Comp.",  
    "W2_Coverage (RAGTruth)": "W2 Coverage",  
    "W2_Overreach (RAGTruth)": "W2 Overreach",  
    "W3a_Verbosity (MTBenchT1)": "W3a Verbosity",  
    "W3a_Hedging": "W3a Hedging",  
    "W3a_Directness": "W3a Directness",  
    "W3a_DD_Verbosity (MTBenchT1)": "W3a_DD Avg. Response Length",  
    "W3a_DD_Verbosity (std response len)": "W3a_DD Std. Response Length",  
    "W3a_DD_Verbosity (avg sentence len)": "W3a_DD Avg. Sentence Length",  
    "W3a_DD_Verbosity (total_sentences)": "W3a_DD Avg. No. Sentences",  
    "W3a_DD_Hedging": "W3a_DD Hedging",  
    "W3a_DD_Directness": "W3a_DD Directness",  
    "W3b_Formatting (Bullet Usage)": "W3b Formatting Use",  
    "W3b_DD_Formatting (Bullet Usage)": "W3b_DD Bullet Use",  
    "W3b_DD_Formatting (Table Usage)": "W3b_DD Table Use",  
    "W3b_DD_Formatting (Emoji Usage)": "W3b_DD Emoji Use",  
    "H1_Pass Rate (IFEval)": "H1 Instructions",  
    "H2_Format (MMLU-Pro)": "H2 Format Fidelity",  
    "H3_Selection (BFCL)": "H3 Tool Use",  
    "H4_Turn 2 Rating (MT-Bench)": "H4 Multi-turn",  
    "H5_Accuracy (RULER 32k)": "H5 Long-Context",  
    "H6_Format Accuracy (HotpotQA)": "H6 Citation Score",  
    "H6_Source Accuracy (HotpotQA)": "H6_DD Citation Source Acc.",  
    "H6_Citation Usage (HotpotQA)": "H6_DD Citation Usage",  
    "H6_DD_Format Accuracy (HotpotQA)": "H6_DD Citation Score",  
    "H6_DD_Source Accuracy (HotpotQA)": "H6_DD Citation Source Acc.",  
    "H6_DD_Citation Usage (HotpotQA)": "H6_DD Citation Usage",  
}


def compute_relative_difference(  
    base_df: pd.DataFrame,   
    adapted_df: pd.DataFrame  
) -> pd.Series:  
    """  
    Compute relative percentage difference: ((adapted - base) / base) * 100  
      
    Args:  
        base_df: DataFrame with base model metrics  
        adapted_df: DataFrame with adapted model metrics  
          
    Returns:  
        Series with relative differences for each metric  
    """  
    # Get numeric columns only  
    numeric_cols = base_df.select_dtypes(include=[np.number]).columns  
      
    base_values = base_df[numeric_cols].iloc[0]  
    adapted_values = adapted_df[numeric_cols].iloc[0]  
      
    # Compute relative difference  
    rel_diff = ((adapted_values - base_values) / base_values) * 100  
      
    return rel_diff


def filter_metrics_by_categories(  
    df: pd.DataFrame,   
    categories: Optional[List[str]] = None  
) -> pd.DataFrame:  
    """  
    Filter dataframe to only include metrics from specified categories.  
      
    Args:  
        df: DataFrame with metrics as columns  
        categories: List of category prefixes (e.g., ['C1', 'C2', 'W1'])  
                   If None, returns all metrics  
          
    Returns:  
        Filtered DataFrame  
    """  
    if categories is None:  
        return df  
      
    # Extract category prefix from column names  
    def get_category_prefix(col_name: str) -> str:  
        parts = col_name.split()  
        if parts:  
            # Extract category like "C1", "C2_DD", "W3a", etc.  
            first_part = parts[0]  
            # Handle cases like "C2_DD" vs "C2"  
            if '_DD' in first_part:  
                return first_part.split('_')[0] + '_DD'  
            else:  
                # Return just the letter+number part (C1, W3a, etc.)  
                import re  
                match = re.match(r'^([A-Z]\d+[a-z]*)', first_part)  
                if match:  
                    return match.group(1)  
        return ""  
      
    # Filter columns  
    selected_cols = [  
        col for col in df.columns   
        if get_category_prefix(col) in categories  
    ]  
    print(selected_cols)
    return df[selected_cols]


def aggregate_metrics(  
    df: pd.DataFrame,   
    aggregation_rules: Optional[Dict] = None  
) -> Tuple[pd.DataFrame, pd.DataFrame]:  
    """  
    Aggregate metrics according to aggregation rules.  
      
    Args:  
        df: DataFrame with rows as model pairs and columns as metrics  
        aggregation_rules: Dictionary mapping (subcategory, metric) -> list of source metrics  
                          If None, uses default AGGREGATION_RULES  
          
    Returns:  
        Tuple of (mean_df, std_df) with aggregated metrics  
    """  
    if aggregation_rules is None:  
        aggregation_rules = AGGREGATION_RULES  
      
    aggregated_mean = {}  
    aggregated_std = {}  
      
    for (subcategory, metric_display), source_metrics in aggregation_rules.items():  
        if f"{subcategory} {metric_display}" not in df.columns:
            continue
        
        # Build the aggregated column name  
        agg_key = f"{subcategory}_{metric_display}"  
          
        # Get display name if available  
        final_column_name = DISPLAY_NAMES.get(agg_key, agg_key)  
          
        # Collect values for the metrics to aggregate  
        values_per_row = []  
          
        for idx in df.index:  
            row_values = []  
            for source_metric in source_metrics:  
                # Find matching columns in the dataframe  
                matching_cols = [col for col in df.columns if " ".join(col.split(" ")[1:]) == source_metric]    
                  
                for col in matching_cols:  
                    val = df.loc[idx, col]  
                    # Only include non-zero, non-NaN values  
                    if not pd.isna(val):  
                        row_values.append(val)  
              
            values_per_row.append(row_values)  
          
        # Calculate mean and std for each row  
        means = []  
        stds = []  
          
        for row_values in values_per_row:  
            if len(row_values) > 0:  
                means.append(np.mean(row_values))  
                stds.append(np.std(row_values) if len(row_values) > 1 else 0.0)  
            else:  
                means.append(0.0)  
                stds.append(0.0)  
          
        aggregated_mean[final_column_name] = means  
        aggregated_std[final_column_name] = stds  
      
    # Create dataframes  
    mean_df = pd.DataFrame(aggregated_mean, index=df.index)  
    std_df = pd.DataFrame(aggregated_std, index=df.index)  
      
    return mean_df, std_df


def _infer_cat(label: str) -> str:  
    """Infer category (CAN/WILL/HOW) from metric label."""  
    s = label.strip()  
    if s.startswith("C"):  
        return "CAN"  
    if s.startswith("W"):  
        return "WILL"  
    if s.startswith("H"):  
        return "HOW"  
    return "UNK"


def _luminance(r: float, g: float, b: float) -> float:  
    """Calculate luminance for determining text color."""  
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _cat_cmap(cat: str, green_red: bool) -> mcolors.Colormap:  
    """Create colormap for a specific category."""  
    if green_red:  
        colors = ['#C2327A', '#F0434F', '#FFFFFF', '#9BC750', '#479A5F']  
    elif cat == "CAN":  
        colors = ["#DA3F0E", "#D64000", "#FFFFFF", "#FFFFFF", "#FFFFFF"]  
    elif cat == "WILL":  
        colors = ["#3A8673", "#4DB299", "#FFFFFF", "#FFFFFF", "#FFFFFF"]  
    elif cat == "HOW":  
        colors = ["#BC8E39", "#E9B045", "#FFFFFF", "#FFFFFF", "#FFFFFF"]  
    else:  
        colors = ["#777777", "#BBBBBB", "#FFFFFF", "#FFFFFF", "#FFFFFF"]  
    return mcolors.LinearSegmentedColormap.from_list(f"Custom_{cat}_White", colors, N=256)


def create_heatmap(    
    data: pd.DataFrame,    
    data_std: Optional[pd.DataFrame] = None,    
    output_path: Optional[Union[str, Path]] = None,    
    color_cap: float = 10.0,    
    figsize: Tuple[int, int] = (16, 6),    
    show_average: bool = False,    
    use_green_red: bool = False,    
    relative: bool = True,  
):    
    """    
    Create a heatmap showing relative differences across metrics.    
        
    Args:    
        data: DataFrame with model pairs as rows and metrics as columns    
        data_std: DataFrame with standard deviations (optional)    
        output_path: Path to save figure (optional)    
        color_cap: Maximum absolute value for color scale    
        figsize: Figure size (width, height)    
        show_average: If True, adds an average column for each category    
        use_green_red: If True, uses symmetric red-green colormap    
        relative: If True, shows "Relative Deviation (%)", else "Absolute Deviation (pp)"  
            
    Returns:    
        Matplotlib figure object    
    """    
    # Infer categories from column labels    
    col_cats = [_infer_cat(lbl) for lbl in data.columns]    
      
    # Extract subcategory prefixes (e.g., "C1", "C2", "W1", "W2", "H1", "H2")  
    def extract_subcategory(label: str) -> str:  
        """Extract subcategory prefix from label (e.g., 'C1 Knowledge' -> 'C1')"""  
        if "Average" in label:  
            return label  # Keep average labels as-is  
        # Split by space and take the first part  
        parts = label.split()  
        if parts:  
            return parts[0]  
        return label  
      
    col_subcats = [extract_subcategory(lbl) for lbl in data.columns]  
        
    # If show_average, insert average columns    
    if show_average:    
        target_order = ["CAN", "WILL", "HOW"]    
            
        new_data_cols = []    
        new_std_cols = [] if data_std is not None else None    
        new_col_names = []    
        new_col_cats = []    
        new_col_subcats = []  
            
        for cat in target_order:    
            # Get columns for this category    
            cat_cols = [col for col, c in zip(data.columns, col_cats) if c == cat]    
                
            if len(cat_cols) == 0:    
                continue    
                
            # Add original columns    
            for i, col in enumerate(cat_cols):    
                new_data_cols.append(data[col])    
                if data_std is not None:    
                    new_std_cols.append(data_std[col])    
                new_col_names.append(col)    
                new_col_cats.append(cat)  
                new_col_subcats.append(col_subcats[list(data.columns).index(col)])  
                
            # Compute and add average column    
            cat_data = data[cat_cols]    
            avg_data = cat_data.mean(axis=1)    
                
            avg_label = f"{cat} Average"  
            new_data_cols.append(avg_data)    
            new_col_names.append(avg_label)    
            new_col_cats.append(cat)    
            new_col_subcats.append(avg_label)  
                
            if data_std is not None:    
                cat_std = data_std[cat_cols]    
                # Pooled standard error    
                avg_std = np.sqrt((cat_std**2).mean(axis=1)) / np.sqrt(len(cat_cols))    
                new_std_cols.append(avg_std)    
            
        # Rebuild dataframes    
        data = pd.DataFrame(dict(zip(new_col_names, new_data_cols)), index=data.index)    
        if data_std is not None:    
            data_std = pd.DataFrame(dict(zip(new_col_names, new_std_cols)), index=data_std.index)    
        col_cats = new_col_cats  
        col_subcats = new_col_subcats  
        
    # Clip data for visualization    
    data_clipped = data.clip(-color_cap, color_cap)    
        
    # Create figure    
    fig, ax = plt.subplots(figsize=figsize)    
        
    # Setup colormaps    
    vmax = color_cap    
    vmin = -color_cap    
    norm_sym = Normalize(vmin=vmin, vmax=vmax, clip=True)    
        
    col_cmaps = {c: _cat_cmap(c, use_green_red) for c in set(col_cats)}    
        
    # Compute category spans and transitions    
    target_order = ["CAN", "WILL", "HOW"]    
    spans = {}    
    for cat in target_order:    
        idxs = [j for j, c in enumerate(col_cats) if c == cat]    
        if len(idxs) > 0:    
            spans[cat] = (min(idxs), max(idxs))    
        
    category_transition_boundaries = []    
    for j in range(len(col_cats) - 1):    
        if col_cats[j] != col_cats[j + 1]:    
            category_transition_boundaries.append(j + 0.5)    
      
    # Compute subcategory boundaries (where subcategory prefix changes)  
    subcategory_boundaries = []  
    for j in range(len(col_subcats) - 1):  
        # Only add boundary if:  
        # 1. Subcategory changes  
        # 2. Main category stays the same (don't duplicate category boundaries)  
        # 3. Neither column is an Average column  
        if (col_subcats[j] != col_subcats[j + 1] and   
            col_cats[j] == col_cats[j + 1] and  
            "Average" not in col_subcats[j] and   
            "Average" not in col_subcats[j + 1]):  
            subcategory_boundaries.append(j + 0.5)  
        
    # Identify average columns    
    average_columns = []    
    if show_average:    
        for j, label in enumerate(data.columns):    
            if "Average" in label:    
                average_columns.append(j)    
        
    # Build RGBA image with per-column colormap    
    n_rows, n_cols = data.shape    
    rgba = np.ones((n_rows, n_cols, 4), dtype=float)    
        
    for j in range(n_cols):    
        cat = col_cats[j]    
        cmap = col_cmaps[cat]    
        for i in range(n_rows):    
            val = data_clipped.iloc[i, j]    
            if np.isnan(val):    
                rgba[i, j] = (1.0, 1.0, 1.0, 1.0)    
            else:    
                rgba[i, j] = cmap(norm_sym(val))    
        
    im = ax.imshow(rgba, aspect="auto")    
        
    # Set ticks and labels    
    ax.set_xticks(np.arange(n_cols))    
    ax.set_yticks(np.arange(n_rows))    
    ax.set_xticklabels(data.columns, rotation=45, ha="right", fontsize=9)    
        
    # Make average column labels bold and colored    
    if show_average:    
        color_map = {    
            "CAN": '#D64000',    
            "WILL": '#4DB299',    
            "HOW": '#E9B045'    
        }    
        for j, label in enumerate(data.columns):    
            if "Average" in label:    
                cat = col_cats[j]    
                tick_color = color_map.get(cat, "black")    
                ax.get_xticklabels()[j].set_fontweight('bold')    
                ax.get_xticklabels()[j].set_color(tick_color)    
        
    ax.set_yticklabels(data.index, fontsize=10, fontweight='bold')    
        
    # Add colorbar    
    def _grey_white_cmap(green_red: bool) -> mcolors.Colormap:    
        if green_red:    
            colors = ['#C2327A', '#F0434F', '#FFFFFF', '#9BC750', '#479A5F']    
        else:    
            colors = ["#4A4A4A", "#9A9A9A", "#FFFFFF", "#FFFFFF", "#FFFFFF"]    
        return mcolors.LinearSegmentedColormap.from_list("GreyWhite", colors, N=256)    
        
    norm_cb = Normalize(vmin=-color_cap, vmax=color_cap, clip=True)    
    cb_cmap = _grey_white_cmap(use_green_red)    
        
    sm = ScalarMappable(norm=norm_cb, cmap=cb_cmap)    
    sm.set_array([])    
        
    cbar = plt.colorbar(sm, ax=ax, pad=0.01, fraction=0.02)    
        
    if relative:    
        cbar.set_label(    
            "Relative Deviation (%)",    
            rotation=270,    
            labelpad=20,    
            fontsize=11,    
            fontweight="bold",    
        )    
    else:    
        cbar.set_label(    
            "Absolute Deviation (pp)",    
            rotation=270,    
            labelpad=20,    
            fontsize=11,    
            fontweight="bold",    
        )    
        
    # Annotate cells    
    for i in range(n_rows):    
        for j in range(n_cols):    
            val = data.iloc[i, j]    
            if np.isnan(val):    
                continue    
                
            r, g, b, _ = rgba[i, j]    
            text_color = "white" if _luminance(r, g, b) < 0.55 else "black"    
                
            # Add std if available    
            if data_std is not None:    
                std = data_std.iloc[i, j]    
                if not np.isnan(std):    
                    txt = f"{val:.1f}\n(±{std:.1f})"    
                    fs = 7    
                else:    
                    txt = f"{val:.1f}"    
                    fs = 8    
            else:    
                txt = f"{val:.1f}"    
                fs = 8    
                
            ax.text(    
                j, i, txt,    
                ha="center", va="center", color=text_color,    
                fontsize=fs, fontweight="bold",    
            )    
        
    # Add grid    
    ax.set_xticks(np.arange(n_cols) - 0.5, minor=True)    
    ax.set_yticks(np.arange(n_rows) - 0.5, minor=True)    
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)    
        
    # --- Grey dividers for subcategory boundaries ---  
    for boundary_pos in subcategory_boundaries:  
        ax.axvline(x=boundary_pos, color='black', linestyle='-', linewidth=1.5, zorder=10)  
      
    # --- Black dividers for category transitions (thicker) ---  
    for boundary_pos in category_transition_boundaries:    
        ax.axvline(x=boundary_pos, color="black", linestyle="-", linewidth=2.5, zorder=20)    
        
    # Thick colored frames around average columns    
    if show_average:    
        color_map = {    
            "CAN": '#D64000',    
            "WILL": '#4DB299',    
            "HOW": '#E9B045'    
        }    
            
        for j in average_columns:    
            cat = col_cats[j]    
            frame_color = color_map.get(cat, "black")    
                
            from matplotlib.patches import Rectangle    
            rect = Rectangle(    
                (j - 0.5, -0.5),    
                1,    
                n_rows,    
                linewidth=3.5,    
                edgecolor=frame_color,    
                facecolor='none',    
                zorder=30    
            )    
            ax.add_patch(rect)    
        
    # Category headers    
    header_map = {    
        "CAN": "CAN – Latent Competence",    
        "WILL": "WILL – Default Behavioral Preference",    
        "HOW": "HOW – Protocol Compliance",    
    }    
    color_map = {    
        "CAN": '#D64000',    
        "WILL": '#4DB299',    
        "HOW": '#E9B045'    
    }    
        
    trans = ax.get_xaxis_transform()    
        
    for cat in target_order:    
        if cat not in spans:    
            continue    
        start_j, end_j = spans[cat]    
        x_center = (start_j + end_j) / 2.0    
        ax.text(    
            x_center,    
            1.01,    
            header_map[cat],    
            transform=trans,    
            ha="center",    
            va="bottom",    
            fontsize=14,    
            fontweight="bold",    
            color=color_map[cat]    
        )    
        
    plt.tight_layout()    
        
    if output_path:    
        plt.savefig(output_path, dpi=300, bbox_inches='tight')    
        print(f"Saved heatmap: {output_path}")    
        
    return fig       