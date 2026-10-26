"""
End-to-end CapTrack Evaluation

Runs the full CapTrack workflow for a base model and an adapted checkpoint:
    1. vLLM inference for both models   (run_inference.py)
    2. CapTrack metric computation        (compute_metrics.py)
    3. Relative comparison and heatmap    (compare_models.py)

Any additional arguments are forwarded to run_inference.py (e.g. --max_model_len, --tensor_parallel_size).

Run this script from the metric environment (environment.yml). Inference runs as a subprocess in the
vLLM environment (environment-inference.yml), selected with --inference_python.

Usage:
    python run_captrack.py --base meta-llama/Llama-3.1-8B-Instruct --adapted ./checkpoints/llama-legal-ift \\
        --inference_python $(conda run -n captrack-vllm which python)
"""

import os
import sys
import copy
import argparse
import logging
import subprocess
from pathlib import Path

from compute_metrics import TASK_TO_METRIC, TASK_CONFIGS, compute_captrack_metrics, save_metrics
from compare_models import DEFAULT_CONFIG, run_comparison
from run_inference import CAPTRACK_TASKS, output_path

logger = logging.getLogger(__name__)

assert CAPTRACK_TASKS == list(TASK_TO_METRIC.keys()), "run_inference.CAPTRACK_TASKS is out of sync with compute_metrics.TASK_TO_METRIC"


def run_inference(model: str, name: str, args: argparse.Namespace, inference_args: list) -> None:
    """Run inference in a separate process so vLLM releases all GPU memory afterwards."""
    tasks = list(args.tasks or CAPTRACK_TASKS)
    if "harmbench.subset" in tasks and "gsm8k.subset" not in tasks:
        tasks.append("gsm8k.subset")
    if all(output_path(Path(args.responses_dir) / name, t).exists() for t in tasks) and "--overwrite" not in inference_args:
        logger.info(f"✓ All responses for '{name}' already exist - skipping inference")
        return
    cmd = [args.inference_python, str(Path(__file__).parent / "run_inference.py"), "--model", model, "--name", name, "--output_dir", args.responses_dir]
    if args.tasks:
        cmd += ["--tasks", *tasks]
    # Put the inference environment's bin/ first on PATH, as if it were activated
    env = dict(os.environ, PATH=f"{Path(args.inference_python).parent}{os.pathsep}{os.environ.get('PATH', '')}")
    subprocess.run(cmd + inference_args, check=True, env=env)


def main():
    parser = argparse.ArgumentParser(
        description="Run CapTrack end-to-end: inference, metrics, and base vs. adapted comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
              # Run from the metric environment, with vLLM installed in a separate environment
              python run_captrack.py --base meta-llama/Llama-3.1-8B-Instruct --adapted ./checkpoints/llama-legal-ift \\
                  --inference_python $(conda run -n captrack-vllm which python)

              # Forward inference options to run_inference.py
              python run_captrack.py --base Qwen/Qwen3-14B --adapted ./checkpoints/qwen3-14b-dpo \\
                  --max_model_len 131072 --chat_template_kwargs '{"enable_thinking": false}'
        """
    )
    parser.add_argument("--base", type=str, required=True, help="Base (out-of-the-box) model: HF model id or local checkpoint")
    parser.add_argument("--adapted", type=str, required=True, help="Adapted (post-trained) model: HF model id or local checkpoint")
    parser.add_argument("--base_name", type=str, default=None, help="Name for the base model (default: last part of --base)")
    parser.add_argument("--adapted_name", type=str, default=None, help="Name for the adapted model (default: last part of --adapted)")
    parser.add_argument("--tasks", nargs='+', default=None, help="Specific tasks to run (default: all CapTrack tasks)")
    parser.add_argument("--responses_dir", type=str, default="./model_responses", help="Directory for model responses (default: ./model_responses)")
    parser.add_argument("--results_dir", type=str, default="./absolute_metrics", help="Directory for CapTrack metrics (default: ./absolute_metrics)")
    parser.add_argument("--comparison_dir", type=str, default="./captrack_relative_results", help="Directory for comparison results (default: ./captrack_relative_results)")
    parser.add_argument("--inference_python", type=str, default=sys.executable, help="Python interpreter of the vLLM environment (default: current interpreter)")
    args, inference_args = parser.parse_known_args()

    if any("@openai" in cfg.get("judge", "") for cfg in TASK_CONFIGS.values()) and not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY is not set - tasks that use an OpenAI LLM judge will fail (see README: LLM Judge Setup)")

    models = [
        (args.base, args.base_name or Path(args.base.rstrip('/')).name),
        (args.adapted, args.adapted_name or Path(args.adapted.rstrip('/')).name),
    ]
    if models[0][1] == models[1][1]:
        parser.error("Base and adapted model resolve to the same name - set --base_name / --adapted_name")

    # 1. Inference
    for model, name in models:
        logger.info(f"\n{'=' * 60}\n[1/3] Inference: {name}\n{'=' * 60}")
        run_inference(model, name, args, inference_args)

    # 2. Metrics
    metric_files = []
    for _, name in models:
        logger.info(f"\n{'=' * 60}\n[2/3] Metrics: {name}\n{'=' * 60}")
        metrics = compute_captrack_metrics(data_dir=args.responses_dir, tasks=args.tasks, model_name=name)
        requested_tasks = args.tasks or list(TASK_CONFIGS)
        failed_tasks = [task for task in requested_tasks if task not in metrics or "error" in metrics[task]]
        if failed_tasks:
            raise RuntimeError(f"Metric computation failed for '{name}': {', '.join(failed_tasks)}")
        metric_files.append(str(save_metrics(metrics, args.results_dir, name)))

    # 3. Comparison
    logger.info(f"\n{'=' * 60}\n[3/3] Comparison: {models[1][1]} vs. {models[0][1]}\n{'=' * 60}")
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["model_pairs"] = [(metric_files[0], metric_files[1], models[1][1])]
    config["output_dir"] = args.comparison_dir
    run_comparison(config)

    logger.info(f"✓ CapTrack evaluation complete. Results in {args.comparison_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
