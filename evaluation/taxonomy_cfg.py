from typing import List, Union
from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark metric"""
    task_name: str  # e.g., "mmlu_pro.core"
    metric_name: Union[str, List[str]]  # e.g., "accuracy"
    category: str  # CAN, WILL, or HOW
    subcategory: str  # e.g., "C1", "W1", "H1"
    display_name: str  # Human-readable name for plots
    color: str # Color


COLORS = {
    'red': '#F0434F',
    'orange': '#E57439',
    'yellow': '#FFB83D',
    'lightgreen': '#9BC750',
    'green': '#479A5F',
    'lightblue': '#5BC5DB',
    'blue': '#538AE5',
    'purple': '#865ED6',
    'darkpink': '#C2327A',
    'grey': '#A1A9AD',
    'brown': '#AD6F50',
    'pink': '#F07FDD'
}

# Taxonomy mapping based on the slides
TAXONOMY_CONFIG = {
    # CAN - Latent Competence
    "C1": {
        "name": "Parametric Knowledge & Skills",
        "benchmarks": [
            BenchmarkConfig("mmlu_pro.core", "accuracy", "CAN", "C1", "Accuracy (MMLU-Pro)", COLORS['blue']), # scale [0,1]
            BenchmarkConfig("popqa.subset", "accuracy", "CAN", "C1", "Accuracy (PopQA)", COLORS['blue']), # scale [0,1]
            BenchmarkConfig("gsm8k.core", "accuracy", "CAN", "C1", "Accuracy (GSM8K)", COLORS['green']), # scale [0,1]
            BenchmarkConfig("livemathbench.full", "accuracy", "CAN", "C1", "Accuracy (livemathbench)", COLORS['green']), # scale [0,1]
            BenchmarkConfig("humaneval.full", "accuracy", "CAN", "C1", "Accuracy (HumanEval)", COLORS['yellow']), # scale [0,1]
            BenchmarkConfig("mbpp.full", "accuracy", "CAN", "C1", "Accuracy (mbpp)", COLORS['yellow']), # scale [0,1]
        ]
    },
    "C2": {
        "name": "Reasoning & Problem Solving",
        "benchmarks": [
            BenchmarkConfig("math.subset", "accuracy", "CAN", "C2", "Accuracy (MATH)", COLORS['grey']), # scale [0,1]
            BenchmarkConfig("math.subset", ["reasoning_metrics", "avg_reasoning_score"], "CAN", "C2", "Reasoning (MATH)", COLORS['pink']), # scale [0,1]
            #BenchmarkConfig("math.subset", ["reasoning_metrics", "avg_num_steps"], "CAN", "C2", "N Steps (MATH)", COLORS['darkpink']), # scale [0,inf)
            BenchmarkConfig("supergpqa.subset", "accuracy", "CAN", "C2", "Accuracy (SuperGPQA)", COLORS['grey']), # scale [0,1]
            BenchmarkConfig("supergpqa.subset", ["reasoning_metrics", "avg_reasoning_score"], "CAN", "C2", "Reasoning (SuperGPQA)", COLORS['pink']), # scale [0,1]
            #BenchmarkConfig("supergpqa.subset", ["reasoning_metrics", "avg_num_steps"], "CAN", "C2", "N Steps (SuperGPQA)", COLORS['darkpink']), # scale [0,inf)
        ]
    },
    "C2_DD": {
        "name": "Reasoning & Problem Solving",
        "benchmarks": [
            BenchmarkConfig("math.subset", "accuracy", "CAN", "C2", "Accuracy (MATH)", COLORS['grey']), # scale [0,1]
            BenchmarkConfig("math.subset", ["reasoning_metrics", "avg_reasoning_score"], "CAN", "C2", "Reasoning (MATH)", COLORS['pink']), # scale [0,1]
            BenchmarkConfig("math.subset", ["reasoning_metrics", "avg_step_validity"], "CAN", "C2", "Step Validity (MATH)", COLORS['pink']), # scale [0,1]
            BenchmarkConfig("math.subset", ["reasoning_metrics", "avg_logical_coherence"], "CAN", "C2", "Logical Coherence (MATH)", COLORS['pink']), # scale [0,1]
            BenchmarkConfig("math.subset", ["reasoning_metrics", "avg_step_consistency"], "CAN", "C2", "Step Consistency (MATH)", COLORS['pink']), # scale [0,1]
            BenchmarkConfig("math.subset", ["reasoning_metrics", "avg_num_steps"], "CAN", "C2", "N Steps (MATH)", COLORS['darkpink']), # scale [0,inf)
            BenchmarkConfig("supergpqa.subset", "accuracy", "CAN", "C2", "Accuracy (SuperGPQA)", COLORS['grey']), # scale [0,1]
            BenchmarkConfig("supergpqa.subset", ["reasoning_metrics", "avg_reasoning_score"], "CAN", "C2", "Reasoning (SuperGPQA)", COLORS['pink']), # scale [0,1]
            BenchmarkConfig("supergpqa.subset", ["reasoning_metrics", "avg_step_validity"], "CAN", "C2", "Step Validity (SuperGPQA)", COLORS['pink']), # scale [0,1]
            BenchmarkConfig("supergpqa.subset", ["reasoning_metrics", "avg_logical_coherence"], "CAN", "C2", "Logical Coherence (SuperGPQA)", COLORS['pink']), # scale [0,1]
            BenchmarkConfig("supergpqa.subset", ["reasoning_metrics", "avg_step_consistency"], "CAN", "C2", "Step Consistency (SuperGPQA)", COLORS['pink']), # scale [0,1]
            BenchmarkConfig("supergpqa.subset", ["reasoning_metrics", "avg_num_steps"], "CAN", "C2", "N Steps (SuperGPQA)", COLORS['darkpink']), # scale [0,inf)
        ]
    },
    "C3": {
        "name": "Contextual Comprehension (ICL)",
        "benchmarks": [
            BenchmarkConfig("hotpotqa.core", "accuracy", "CAN", "C3a", "Accuracy (HotpotQA)", COLORS['green']), # scale [0,1]
            BenchmarkConfig("hotpotqa.core", "evidence_hit_rate", "CAN", "C3", "Evidence Hit (HotpotQA)", COLORS['lightgreen']), # scale [0,1]
            BenchmarkConfig("boolq.core", "accuracy", "CAN", "C3", "Accuracy (BoolQ)", COLORS['green']),  # scale [0,1]
            BenchmarkConfig("boolq.core", "evidence_hit_rate", "CAN", "C3", "Evidence Hit (BoolQ)", COLORS['lightgreen']), # scale [0,1]
        ]
    },
    "C4": {
        "name": "Epistemic Faithfulness & Grounding",
        "benchmarks": [
            #BenchmarkConfig("ragtruth.core", "coverage_score", "CAN", "C4", "RAGTruth (Coverage)"),  # scale [0,1]
            BenchmarkConfig("ragtruth.core", "faithfulness_score", "CAN", "C4", "Faithfulness (RAGTruth)", COLORS['purple']),
            BenchmarkConfig("truthfulqa.full", "accuracy", "CAN", "C4", "Accuracy (TruthfulQA)", COLORS['purple']),
        ]
    },
    "C5a": {
        "name": "Robustness of Competence",
        "benchmarks": [
            # C5a - Prompt-form invariance
            BenchmarkConfig("mmlu_pro.rephrased", "accuracy", "CAN", "C5a", "Rephrased (MMLU-Pro)", COLORS['lightblue']), # scale [0,1]
            BenchmarkConfig("gsm8k.rephrased", "accuracy", "CAN", "C5a", "Rephrased (GSM8K)", COLORS['lightgreen']), # scale [0,1]
        ]
    },
    "C5b": {
        "name": "Robustness of Competence",
        "benchmarks": [
            # C5b - Domain shift robustness
            BenchmarkConfig("winogrande.core", "accuracy", "CAN", "C5b", "Domain Shift (WinoGrande)", COLORS['purple']), # scale [0,1]
            BenchmarkConfig("hellaswag.core", "accuracy", "CAN", "C5b", "Domain Shift (HellaSwag)", COLORS['brown']), # scale [0,1]
        ]
    },
    "C5c": {
        "name": "Robustness of Competence",
        "benchmarks": [
            # C5c - Multilingual stability
            BenchmarkConfig("mgsm.core", "accuracy", "CAN", "C5c", "Multilingual (MGSM)", COLORS['darkpink']), # scale [0,1]
            BenchmarkConfig("xtreme.subset", "accuracy", "CAN", "C5c", "Multilingual (XTREME)", COLORS['darkpink']), # scale [0,1]
        ]
    },

    # WILL - Policy & Safety Preferences
    "W1": {
        "name": "Willingness to Answer",
        "benchmarks": [
            BenchmarkConfig("harmbench.subset", "unsafe_refusal_rate", "WILL", "W1", "Unsafe Refusal", COLORS['red']),  # scale [0,1]
            BenchmarkConfig("ruler.4k", "compliance_rate", "WILL", "W1b", "Compliance Rate", COLORS['blue']),  # scale [0,1]
        ]
    },
    "W1a": {
        "name": "Willingness to Answer",
        "benchmarks": [
            BenchmarkConfig("harmbench.subset", "unsafe_refusal_rate", "WILL", "W1a", "Unsafe Refusal", COLORS['red']),  # scale [0,1]
        ]
    },
    "W1b": {
        "name": "Willingness to Answer",
        "benchmarks": [
            BenchmarkConfig("ruler.4k", "compliance_rate", "WILL", "W1b", "Compliance Rate", COLORS['blue']),  # scale [0,1]
        ]
    },
    "W1_DD": {
        "name": "Willingness to Answer",
        "benchmarks": [
            BenchmarkConfig("harmbench.subset", "benign_compliance_rate", "WILL", "W1", "Benign Compliance", COLORS['green']),  # scale [0,1]
            BenchmarkConfig("harmbench.subset", "unsafe_refusal_rate", "WILL", "W1", "Unsafe Refusal", COLORS['red']),  # scale [0,1]
            BenchmarkConfig("harmbench.subset", "unsafe_strong_refusal_rate", "WILL", "W1", "Unsafe Strong Refusal", COLORS['red']),  # scale [0,1]
            BenchmarkConfig("harmbench.subset", "unsafe_weak_refusal_rate", "WILL", "W1", "Unsafe Weak Refusal", COLORS['red']),  # scale [0,1]
            BenchmarkConfig("harmbench.subset", "unsafe_evasion_rate", "WILL", "W1", "Unsafe Evasion", COLORS['red']),  # scale [0,1]
        ]
    },
    "W2": {
        "name": "Helpfulness & Informational Scope",
        "benchmarks": [
            BenchmarkConfig("ragtruth.core", "coverage_score", "WILL", "W3", "Coverage (RAGTruth)", COLORS['lightgreen']), # scale [0,1]
            BenchmarkConfig("eli5.subset", "coverage_score", "WILL", "W3", "Coverage (ELI5)", COLORS['lightgreen']), # scale [0,1]
            BenchmarkConfig("ragtruth.core", "overreach_score", "WILL", "W3", "Overreach (RAGTruth)", COLORS['orange']), # scale [0,1]
            BenchmarkConfig("eli5.subset", "overreach_score", "WILL", "W3", "Overreach (ELI5)", COLORS['orange']), # scale [0,1]
        ]
    },
    "W2a": {
        "name": "Helpfulness & Informational Scope",
        "benchmarks": [
            BenchmarkConfig("ragtruth.core", "coverage_score", "WILL", "W3", "Coverage (RAGTruth)", COLORS['lightgreen']), # scale [0,1]
            BenchmarkConfig("eli5.subset", "coverage_score", "WILL", "W3", "Coverage (ELI5)", COLORS['lightgreen']), # scale [0,1]
        ]
    },
    "W2b": {
        "name": "Helpfulness & Informational Scope",
        "benchmarks": [
            BenchmarkConfig("ragtruth.core", "overreach_rate", "WILL", "W3", "Overreach (RAGTruth)", COLORS['orange']), # scale [0,1]
            BenchmarkConfig("eli5.subset", "overreach_rate", "WILL", "W3", "Overreach (ELI5)", COLORS['orange']), # scale [0,1]
        ]
    },
    "W3a": {
        "name": "Style & Level of Elaboration",
        "benchmarks": [
            BenchmarkConfig("mtbench.turn1", ["verbosity", "mean_answer_length"], "WILL", "W4a", "Verbosity (MTBenchT1)", COLORS['blue']), # scale [0,inf)
            BenchmarkConfig("oasst1.full", ["verbosity", "mean_answer_length"], "WILL", "W4a", "Verbosity (OASST1)", COLORS['blue']), # scale [0,inf)
        ]
    },
    "W3a_DD": {
        "name": "Style & Level of Elaboration",
        "benchmarks": [
            BenchmarkConfig("mtbench.turn1", ["verbosity", "mean_answer_length"], "WILL", "W4a", "Verbosity (MTBenchT1)", COLORS['blue']), # scale [0,inf)
            BenchmarkConfig("mtbench.turn1", ["verbosity", "std_length"], "WILL", "W4a", "Verbosity (std response len)", COLORS['blue']), # scale [0,inf)
            #BenchmarkConfig("mtbench.turn1", ["verbosity", "percentile_90_length"], "WILL", "W4a", "Verbosity (90th perc. len)", COLORS['blue']), # scale [0,inf)
            BenchmarkConfig("mtbench.turn1", ["verbosity", "avg_sentence_length"], "WILL", "W4a", "Verbosity (avg sentence len)", COLORS['blue']), # scale [0,inf)
            BenchmarkConfig("mtbench.turn1", ["verbosity", "total_sentences"], "WILL", "W4a", "Verbosity (total_sentences)", COLORS['blue']), # scale [0,inf)
            BenchmarkConfig("mtbench.turn1", ["hedging", "hedging_rate"], "WILL", "W4a", "Hedging", COLORS['red']), # scale [0,1]
            BenchmarkConfig("mtbench.turn1", ["directness", "directness_rate"], "WILL", "W4a", "Directness", COLORS['green']),  # scale [0,1]
            BenchmarkConfig("oasst1.full", ["verbosity", "mean_answer_length"], "WILL", "W4a", "Verbosity (OASST1)", COLORS['blue']), # scale [0,inf)
            BenchmarkConfig("oasst1.full", ["verbosity", "std_length"], "WILL", "W4a", "Verbosity (std response len) (OASST1)", COLORS['blue']), # scale [0,inf)
            BenchmarkConfig("oasst1.full", ["verbosity", "avg_sentence_length"], "WILL", "W4a", "Verbosity (avg sentence len) (OASST1)", COLORS['blue']), # scale [0,inf)
            BenchmarkConfig("oasst1.full", ["verbosity", "total_sentences"], "WILL", "W4a", "Verbosity (total_sentences) (OASST1)", COLORS['blue']), # scale [0,inf)
            BenchmarkConfig("oasst1.full", ["hedging", "hedging_rate"], "WILL", "W4a", "Hedging (OASST1)", COLORS['red']), # scale [0,1]
            BenchmarkConfig("oasst1.full", ["directness", "directness_rate"], "WILL", "W4a", "Directness (OASST1)", COLORS['green']),  # scale [0,1]
        ]
    },
    "W3b": {
        "name": "Style & Level of Elaboration",
        "benchmarks": [
            BenchmarkConfig("mtbench.turn1", ["formatting", "bullet_usage_rate"], "WILL", "W4b", "Formatting (Bullet Usage)", COLORS['lightblue']),  # scale [0,1]
            BenchmarkConfig("mtbench.turn1", ["formatting", "table_usage_rate"], "WILL", "W4b", "Formatting (Table Usage)", COLORS['lightgreen']),  # scale [0,1]
            BenchmarkConfig("mtbench.turn1", ["formatting", "emoji_usage_rate"], "WILL", "W4b", "Formatting (Emoji Usage)", COLORS['yellow']),  # scale [0,1]
            BenchmarkConfig("oasst1.full", ["formatting", "bullet_usage_rate"], "WILL", "W4b", "Formatting (Bullet Usage) (OASST1)", COLORS['lightblue']),  # scale [0,1]
            BenchmarkConfig("oasst1.full", ["formatting", "table_usage_rate"], "WILL", "W4b", "Formatting (Table Usage) (OASST1)", COLORS['lightgreen']),  # scale [0,1]
            BenchmarkConfig("oasst1.full", ["formatting", "emoji_usage_rate"], "WILL", "W4b", "Formatting (Emoji Usage) (OASST1)", COLORS['yellow']),  # scale [0,1]
        ]
    },

    "W3b_DD": {
        "name": "Style & Level of Elaboration",
        "benchmarks": [
            BenchmarkConfig("mtbench.turn1", ["formatting", "bullet_usage_rate"], "WILL", "W4b", "Formatting (Bullet Usage)", COLORS['lightblue']),  # scale [0,1]
            BenchmarkConfig("mtbench.turn1", ["formatting", "table_usage_rate"], "WILL", "W4b", "Formatting (Table Usage)", COLORS['lightgreen']),  # scale [0,1]
            BenchmarkConfig("mtbench.turn1", ["formatting", "emoji_usage_rate"], "WILL", "W4b", "Formatting (Emoji Usage)", COLORS['yellow']),  # scale [0,1]
            BenchmarkConfig("oasst1.full", ["formatting", "bullet_usage_rate"], "WILL", "W4b", "Formatting (Bullet Usage) (OASST1)", COLORS['lightblue']),  # scale [0,1]
            BenchmarkConfig("oasst1.full", ["formatting", "table_usage_rate"], "WILL", "W4b", "Formatting (Table Usage) (OASST1)", COLORS['lightgreen']),  # scale [0,1]
            BenchmarkConfig("oasst1.full", ["formatting", "emoji_usage_rate"], "WILL", "W4b", "Formatting (Emoji Usage) (OASST1)", COLORS['yellow']),  # scale [0,1]
        ]
    },

    # HOW - Protocol Compliance
    "H1": {
        "name": "Instruction Following & Constraint Satisfaction",
        "benchmarks": [
            BenchmarkConfig("ifeval.core", ["strict", "perfect_compliance_rate"], "HOW", "H1", "Pass Rate (IFEval)", COLORS['yellow']),  # scale [0,1]
            BenchmarkConfig("followbench.subset", "accuracy", "HOW", "H1", "Accuracy (FollowBench)", COLORS['yellow']),  # scale [0,1]
        ]
    },
    "H1_DD": {
        "name": "Instruction Following & Constraint Satisfaction",
        "benchmarks": [
            BenchmarkConfig("ifeval.core", ["strict", "perfect_compliance_rate"], "HOW", "H1", "Pass Rate (strict)", COLORS['yellow']),  # scale [0,1]
            BenchmarkConfig("ifeval.core", ["strict", "tier0_pass_rates", "punctuation"], "HOW", "H1", "Pass Rate (punctuation)", COLORS['yellow']),  # scale [0,1]
            BenchmarkConfig("ifeval.core", ["strict", "tier0_pass_rates", "detectable_format"], "HOW", "H1", "Pass Rate (detectable_format)", COLORS['yellow']),  # scale [0,1]
            BenchmarkConfig("ifeval.core", ["strict", "tier0_pass_rates", "length_constraints"], "HOW", "H1", "Pass Rate (length_constraints)", COLORS['yellow']),  # scale [0,1]
            BenchmarkConfig("ifeval.core", ["strict", "tier0_pass_rates", "detectable_content"], "HOW", "H1", "Pass Rate (detectable_content)", COLORS['yellow']),  # scale [0,1]
            BenchmarkConfig("ifeval.core", ["strict", "tier0_pass_rates", "combination"], "HOW", "H1", "Pass Rate (combination)", COLORS['yellow']),  # scale [0,1]
            BenchmarkConfig("ifeval.core", ["strict", "tier0_pass_rates", "change_case"], "HOW", "H1", "Pass Rate (change_case)", COLORS['yellow']),  # scale [0,1]
            BenchmarkConfig("ifeval.core", ["strict", "tier0_pass_rates", "startend"], "HOW", "H1", "Pass Rate (startend)", COLORS['yellow']),  # scale [0,1]
            BenchmarkConfig("ifeval.core", ["strict", "tier0_pass_rates", "keywords"], "HOW", "H1", "Pass Rate (keywords)", COLORS['yellow']),  # scale [0,1]
            BenchmarkConfig("ifeval.core", ["strict", "tier0_pass_rates", "language"], "HOW", "H1", "Pass Rate (language)", COLORS['yellow']),  # scale [0,1]
            BenchmarkConfig("followbench.subset", "accuracy", "HOW", "H1", "Followbench Score", COLORS['yellow']),  # scale [0,1]
        ]
    },
    "H2": {
        "name": "Output-Format Fidelity",
        "benchmarks": [
            BenchmarkConfig("mmlu_pro.schema", "accuracy", "HOW", "H2", "Format (MMLU-Pro)", COLORS['lightblue']),  # scale [0,1]
            BenchmarkConfig("gsm8k.schema", "accuracy", "HOW", "H2", "Format (GSM8K)", COLORS['lightgreen']),  # scale [0,1]
            BenchmarkConfig("mmlu_pro.table_schema", "accuracy", "HOW", "H2", "Format Table (MMLU-Pro)", COLORS['lightblue']),  # scale [0,1]
            BenchmarkConfig("gsm8k.table_schema", "accuracy", "HOW", "H2", "Format Table (GSM8K)", COLORS['lightgreen']),  # scale [0,1]
        ]
    },
    "H3": {
        "name": "Tool/Function Use & Integration",
        "benchmarks": [
            BenchmarkConfig("bfcl.subset", "selection_accuracy", "HOW", "H3", "Selection (BFCL)", COLORS['blue']),  # scale [0,1]
            BenchmarkConfig("bfcl.subset", "argument_accuracy", "HOW", "H3", "Arguments (BFCL)", COLORS['orange']),  # scale [0,1]
            #BenchmarkConfig("bfcl.subset", "integration_accuracy", "HOW", "H3", "Integration (BFCL)", COLORS['green']),  # scale [0,1]
            BenchmarkConfig("mnms.full", "selection_accuracy", "HOW", "H3", "Selection (MNMS)", COLORS['blue']),  # scale [0,1]
            BenchmarkConfig("mnms.full", "argument_accuracy", "HOW", "H3", "Arguments (MNMS)", COLORS['orange']),  # scale [0,1]
            #BenchmarkConfig("mnms.full", "integration_accuracy", "HOW", "H3", "Integration (MNMS)", COLORS['green']),  # scale [0,1]
        ]
    },
    "H3_DD": {
        "name": "Tool/Function Use & Integration",
        "benchmarks": [
            BenchmarkConfig("bfcl.subset", "selection_accuracy", "HOW", "H3", "Selection (BFCL)", COLORS['blue']),  # scale [0,1]
            BenchmarkConfig("bfcl.subset", "argument_accuracy", "HOW", "H3", "Arguments (BFCL)", COLORS['orange']),  # scale [0,1]
            #BenchmarkConfig("bfcl.subset", "integration_accuracy", "HOW", "H3", "Integration (BFCL)", COLORS['green']),  # scale [0,1]
            BenchmarkConfig("mnms.full", "selection_accuracy", "HOW", "H3", "Selection (MNMS)", COLORS['blue']),  # scale [0,1]
            BenchmarkConfig("mnms.full", "argument_accuracy", "HOW", "H3", "Arguments (MNMS)", COLORS['orange']),  # scale [0,1]
            #BenchmarkConfig("mnms.full", "integration_accuracy", "HOW", "H3", "Integration (MNMS)", COLORS['green']),  # scale [0,1]
        ]
    },
    "H4": {
        "name": "Multi-turn State & Commitment Keeping",
        "benchmarks": [
            BenchmarkConfig("mtbench.turn2", ["rating_stats", "mean_rating"], "HOW", "H4", "Turn 2 Rating (MT-Bench)", COLORS['yellow']),  # scale [0,10]
            BenchmarkConfig("structflowbench.turn2", ["rating_stats", "mean_rating"], "HOW", "H4", "Turn 2 Rating (StructFlowBench)", COLORS['yellow']),  # scale [0,10]
        ]
    },
    "H4_DD": {
        "name": "Multi-turn State & Commitment Keeping",
        "benchmarks": [
            BenchmarkConfig("mtbench.turn2", ["rating_stats", "mean_rating"], "HOW", "H4", "Turn 2 Rating (mean)", COLORS['yellow']),  # scale [0,10]
            BenchmarkConfig("mtbench.turn2", ["rating_stats", "std_rating"], "HOW", "H4", "Turn 2 Rating (std)", COLORS['yellow']),  # scale [0,10]
            BenchmarkConfig("mtbench.turn2", ["rating_stats", "median_rating"], "HOW", "H4", "Turn 2 Rating (median)", COLORS['yellow']),  # scale [0,10]
        ]
    },
    "H5": {
        "name": "Context-Window Operations",
        "benchmarks": [
            BenchmarkConfig("ruler.32k", "exact_accuracy", "HOW", "H5", "Accuracy (RULER 32k)", COLORS['blue']),  # scale [0,1]
            BenchmarkConfig("longbenchv2.full", "accuracy", "HOW", "H5", "Accuracy (LonvbenchV2)", COLORS['blue']),  # scale [0,1]
        ]
    },
    "H5_DD": {
        "name": "Context-Window Operations",
        "benchmarks": [
            BenchmarkConfig("ruler.32k", "exact_accuracy", "HOW", "H5", "Exact Acc. (RULER 32k)", COLORS['blue']),  # scale [0,1]
            BenchmarkConfig("ruler.32k", "partial_accuracy", "HOW", "H5", "Partial Acc. (RULER 32k)", COLORS['blue']),  # scale [0,1]
        ]
    },
    "H6": {
        "name": "Citation/Attribution Mechanics",
        "benchmarks": [
            BenchmarkConfig("hotpotqa.citation", "format_accuracy", "HOW", "H6", "Format Accuracy (HotpotQA)", COLORS['lightblue']),  # scale [0,1]
            BenchmarkConfig("hotpotqa.citation", "source_accuracy", "HOW", "H6", "Source Accuracy (HotpotQA)", COLORS['lightgreen']),  # scale [0,1]
            BenchmarkConfig("qasper.citation", "format_accuracy", "HOW", "H6", "Format Accuracy (QASPER)", COLORS['lightblue']),  # scale [0,1]
            BenchmarkConfig("qasper.citation", "source_accuracy", "HOW", "H6", "Source Accuracy (QASPER)", COLORS['lightgreen']),  # scale [0,1]
        ]
    },
    "H6_DD": {
        "name": "Citation/Attribution Mechanics",
        "benchmarks": [
            BenchmarkConfig("hotpotqa.citation", "format_accuracy", "HOW", "H6", "Format Accuracy (HotpotQA)", COLORS['lightblue']),  # scale [0,1]
            BenchmarkConfig("hotpotqa.citation", "source_accuracy", "HOW", "H6", "Source Accuracy (HotpotQA)", COLORS['lightgreen']),  # scale [0,1]
            BenchmarkConfig("hotpotqa.citation", "citation_usage_rate", "HOW", "H6", "Citation Usage (HotpotQA)", COLORS['yellow']),  # scale [0,1]
            BenchmarkConfig("qasper.citation", "format_accuracy", "HOW", "H6", "Format Accuracy (QASPER)", COLORS['lightblue']),  # scale [0,1]
            BenchmarkConfig("qasper.citation", "source_accuracy", "HOW", "H6", "Source Accuracy (QASPER)", COLORS['lightgreen']),  # scale [0,1]
            BenchmarkConfig("qasper.citation", "citation_usage_rate", "HOW", "H6", "Citation Usage (QASPER)", COLORS['yellow']),  # scale [0,1]
        ]
    },
}