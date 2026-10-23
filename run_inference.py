"""
CapTrack Inference with vLLM

Generates model responses for all CapTrack tasks and writes them to
model_responses/<model_name>/<task>.jsonl, the layout expected by compute_metrics.py.

Each task is loaded from the CapTrack Hugging Face dataset. The task's `formatted_prompt`
is sent as the user message (after an optional system prompt) through the model's chat
template, using deterministic decoding. Multi-turn tasks are generated in two passes: the
model's own turn-1 answer is placed in context before the turn-2 question.

Usage:
    python run_inference.py --model meta-llama/Llama-3.1-8B-Instruct
    python run_inference.py --model ./checkpoints/my-adapted-model --name my_adapted --limit 20
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from huggingface_hub import hf_hub_download

# Decoding is greedy, so FlashInfer's top-k/top-p sampler is not needed; disabling it avoids
# JIT-compiling CUDA kernels at startup (requires nvcc/ninja on the GPU nodes)
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


HF_DATASET = "tri-fair-lab/captrack"

# All CapTrack evaluation tasks (same keys as TASK_TO_METRIC in compute_metrics.py). Kept here so
# inference does not depend on the metric stack and can run in its own environment.
CAPTRACK_TASKS = [
    "mmlu_pro.subset", "mmlu_pro.rephrased", "mmlu_pro.schema", "mmlu_pro.table_schema",
    "gsm8k.subset", "gsm8k.rephrased", "gsm8k.schema", "gsm8k.table_schema",
    "mgsm.subset", "humaneval.full", "math.subset", "supergpqa.subset",
    "hotpotqa.subset", "hotpotqa.citation", "boolq.subset", "ragtruth.subset",
    "winogrande.subset", "hellaswag.subset", "ifeval.subset", "bfcl.subset",
    "ruler.4k", "ruler.32k", "mtbench.turn1", "mtbench.turn2",
    "harmbench.subset", "popqa.subset", "livemathbench.full", "mbpp.full",
    "truthfulqa.full", "xtreme.subset", "eli5.subset", "oasst1.full",
    "followbench.subset", "mnms.full", "structflowbench.turn2", "longbenchv2.full",
    "qasper.citation",
]

# Evaluation task names stored under a different name in the HF dataset
HF_TASK_NAMES = {"ruler.4k": "ruler.incomplete"}

# Second-turn tasks and the first-turn task they are built from
SECOND_TURN_TASKS = {
    "mtbench.turn2": "mtbench.turn1",
    "structflowbench.turn2": "structflowbench.turn1",
}

# Tasks scored on the model's reasoning steps: compute_math_supergpqa_accuracy parses both the
# intermediate steps and the final answer from `outputs`. A thinking model may place its steps
# inside <think>...</think>, so keep their raw output instead of stripping it (see generate()).
REASONING_TASKS = {"math.subset", "supergpqa.subset"}

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# Maximum number of generated tokens. Kept generous so natural responses are never cut off.
DEFAULT_MAX_TOKENS = 4096
TASK_MAX_TOKENS = {
    "math.subset": 8192,
    "supergpqa.subset": 8192,
    "livemathbench.full": 8192,
}


def load_task(task: str) -> pd.DataFrame:
    """Download (or load from cache) the CapTrack data for a task."""
    hf_name = HF_TASK_NAMES.get(task, task)
    path = hf_hub_download(HF_DATASET, f"data/{hf_name}/data.parquet", repo_type="dataset")
    return pd.read_parquet(path)


def output_path(output_dir: Path, task: str) -> Path:
    """File name looked up by compute_metrics.load_task_data (e.g. gsm8k.subset -> gsm8k_subset.jsonl)."""
    return output_dir / f"{task.replace('.', '_')}.jsonl"


class Generator:
    """Thin wrapper around a vLLM engine with CapTrack's fixed decoding settings."""

    def __init__(self, args: argparse.Namespace):
        import torch
        from vllm import LLM

        engine_kwargs = dict(
            model=args.model,
            dtype="bfloat16",
            tensor_parallel_size=args.tensor_parallel_size or max(torch.cuda.device_count(), 1),
            gpu_memory_utilization=0.9,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=8192,
            max_num_seqs=256,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            seed=0,
        )
        engine_kwargs.update(args.vllm_kwargs)
        self.llm = LLM(**engine_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.max_model_len = self.llm.model_config.max_model_len
        self.n_truncated = 0
        self.max_tokens = args.max_tokens
        self.chat_template_kwargs = args.chat_template_kwargs
        self.system_prompt = args.system_prompt
        self.system_role_supported = self._supports_system_role()
        if self.system_prompt and not self.system_role_supported:
            logger.warning("Chat template rejects the system role - prepending the system prompt to the user message")

    def _supports_system_role(self) -> bool:
        try:
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return True
        except Exception:
            return False

    def _fit(self, text: str, budget: int) -> str:
        """Truncate text in the middle so it fits in `budget` tokens (no-op for short texts)."""
        if budget <= 0:
            return ""
        if len(text) <= budget:
            return text
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= budget:
            return text
        half = budget // 2
        tail = budget - half
        return self.tokenizer.decode(ids[:half]) + self.tokenizer.decode(ids[-tail:])

    def _conversation_token_count(self, messages: List[Dict[str, str]]) -> int:
        kwargs = self.chat_template_kwargs or {}
        token_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, **kwargs
        )
        return len(token_ids)

    def build_conversation(self, user_turns: List[str], assistant_turns: List[str], max_tokens: int) -> List[Dict[str, str]]:
        """Interleave chat turns and truncate them to fit the context and generation budget."""
        first = user_turns[0]
        if self.system_prompt and not self.system_role_supported:
            first = f"{self.system_prompt}\n\n{first}"
        messages = []
        if self.system_prompt and self.system_role_supported:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": first})
        for assistant, user in zip(assistant_turns, user_turns[1:]):
            messages.append({"role": "assistant", "content": assistant})
            messages.append({"role": "user", "content": user})
        original_contents = [message["content"] for message in messages]

        generation_budget = min(max_tokens, max(self.max_model_len // 2, 1))
        prompt_budget = self.max_model_len - generation_budget
        last_user_index = len(messages) - 1
        trim_indices = [
            i for i, message in enumerate(messages)
            if message["role"] != "system" and i != last_user_index
        ]
        if messages[last_user_index]["role"] == "user":
            trim_indices.append(last_user_index)

        for index in trim_indices:
            if self._conversation_token_count(messages) <= prompt_budget:
                break

            text = messages[index]["content"]
            token_count = len(self.tokenizer.encode(text, add_special_tokens=False))
            low, high = 0, token_count
            fitted = ""
            while low <= high:
                middle = (low + high) // 2
                candidate = self._fit(text, middle)
                messages[index]["content"] = candidate
                if self._conversation_token_count(messages) <= prompt_budget:
                    fitted = candidate
                    low = middle + 1
                else:
                    high = middle - 1
            messages[index]["content"] = fitted

        if self._conversation_token_count(messages) > prompt_budget:
            raise ValueError(
                f"Conversation template exceeds max_model_len={self.max_model_len} "
                f"even after truncation; reduce max_tokens or the system prompt"
            )
        self.n_truncated += any(
            message["content"] != original
            for message, original in zip(messages, original_contents)
        )
        return messages

    def generate(self, task: str, turns: List[List[str]], answers: Optional[List[List[str]]] = None) -> List[str]:
        """Generate one response per conversation."""
        from vllm import SamplingParams

        max_tokens = self.max_tokens or TASK_MAX_TOKENS.get(task, DEFAULT_MAX_TOKENS)
        answers = answers or [[] for _ in turns]
        self.n_truncated = 0
        conversations = [self.build_conversation(u, a, max_tokens) for u, a in zip(turns, answers)]
        if self.n_truncated:
            logger.warning(f"{task}: truncated {self.n_truncated}/{len(turns)} prompts to fit max_model_len={self.max_model_len}")

        prompt_tokens = max(
            (self._conversation_token_count(conversation) for conversation in conversations),
            default=0,
        )
        max_tokens = min(max_tokens, self.max_model_len - prompt_tokens)
        if max_tokens <= 0:
            raise ValueError(f"No context remains for generation with max_model_len={self.max_model_len}")
        params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens)
        outputs = self.llm.chat(conversations, params, chat_template_kwargs=self.chat_template_kwargs)
        # Drop reasoning traces of thinking models so answer parsers see the final response only.
        # Exception: REASONING_TASKS are graded on the model's intermediate steps, which a thinking
        # model may place inside <think>...</think>; keep their raw output so the reasoning scorer
        # can parse both the steps and the final answer.
        if task in REASONING_TASKS:
            return [o.outputs[0].text for o in outputs]
        return [re.sub(r"^.*?</think>\s*", "", o.outputs[0].text, flags=re.S) for o in outputs]


def run_task(task: str, generator: Generator, output_dir: Path, limit: Optional[int], overwrite: bool,
             done: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Generate (or load existing) responses for one task and save them to disk."""
    if task in done:
        return done[task]
    path = output_path(output_dir, task)
    if path.exists() and not overwrite:
        logger.info(f"✓ {task}: found existing responses at {path} - skipping")
        done[task] = pd.read_json(path, orient='records', lines=True)
        return done[task]

    if task in SECOND_TURN_TASKS:
        turn1 = run_task(SECOND_TURN_TASKS[task], generator, output_dir, limit, overwrite, done)
        df = turn1.drop(columns=["outputs"]).copy()
        # Turn 2 context: formatted turn-1 prompt + the model's own turn-1 answer + raw turn-2 question
        df["prompt"] = df["prompt"].map(lambda p: json.loads(p) if isinstance(p, str) else p)
        df["turn1_response"] = turn1["outputs"]
        question_2 = [p[1]["user prompt"] if isinstance(p[1], dict) else p[1] for p in df["prompt"]]
        turns = [[fp, q2] for fp, q2 in zip(df["formatted_prompt"], question_2)]
        answers = [[a] for a in df["turn1_response"]]
    else:
        df = load_task(task)
        if limit:
            df = df.head(limit)
        turns = [[fp] for fp in df["formatted_prompt"]]
        answers = None

    logger.info(f"Generating {len(df)} responses for '{task}'")
    df["outputs"] = generator.generate(task, turns, answers)

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient='records', lines=True, force_ascii=False)
    logger.info(f"✓ {task}: saved responses to {path}")
    done[task] = df
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate CapTrack model responses with vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
              # All CapTrack tasks for a Hugging Face model
              python run_inference.py --model meta-llama/Llama-3.1-8B-Instruct

              # Local checkpoint, quick smoke test on 20 samples per task
              python run_inference.py --model ./checkpoints/my-model --limit 20

              # Hybrid thinking model with thinking disabled, YaRN RoPE scaling
              python run_inference.py --model Qwen/Qwen3-14B --max_model_len 131072 \\
                  --chat_template_kwargs '{"enable_thinking": false}' \\
                  --vllm_kwargs '{"hf_overrides": {"rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}}}'
        """
    )
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model id or path to a local checkpoint")
    parser.add_argument("--name", type=str, default=None, help="Model name used for the output directory (default: last part of --model)")
    parser.add_argument("--output_dir", type=str, default="./model_responses", help="Root directory for model responses (default: ./model_responses)")
    parser.add_argument("--tasks", nargs='+', default=None, help="Specific tasks to run (default: all CapTrack tasks)")
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N samples per task (for quick tests)")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate tasks that already have responses")
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="System prompt (use '' to disable)")
    parser.add_argument("--max_tokens", type=int, default=None, help="Override the per-task maximum number of generated tokens")
    parser.add_argument("--max_model_len", type=int, default=None, help="Maximum context length (default: model's native length; the paper uses 130000)")
    parser.add_argument("--tensor_parallel_size", type=int, default=None, help="Number of GPUs for tensor parallelism (default: all visible GPUs)")
    parser.add_argument("--chat_template_kwargs", type=json.loads, default=None, help="JSON dict passed to the chat template, e.g. '{\"enable_thinking\": false}'")
    parser.add_argument("--vllm_kwargs", type=json.loads, default={}, help="JSON dict of extra vllm.LLM arguments, e.g. '{\"trust_remote_code\": true}'")
    args = parser.parse_args()

    tasks = args.tasks or CAPTRACK_TASKS
    unknown = [t for t in tasks if t not in CAPTRACK_TASKS]
    if unknown:
        parser.error(f"Unknown tasks: {unknown}. Available: {CAPTRACK_TASKS}")

    name = args.name or Path(args.model.rstrip('/')).name
    output_dir = Path(args.output_dir) / name

    logger.info("=" * 60)
    logger.info("CapTrack Inference")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Responses: {output_dir}")
    logger.info(f"Tasks: {len(tasks)}")
    logger.info("=" * 60 + "\n")

    generator = Generator(args)
    done = {}
    for task in tasks:
        run_task(task, generator, output_dir, args.limit, args.overwrite, done)

    logger.info(f"✓ Inference complete. Next: python compute_metrics.py --model_name {name}")
    return 0


if __name__ == "__main__":
    exit(main())
