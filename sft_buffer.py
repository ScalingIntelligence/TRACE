"""SFT data extraction from tau2-bench eval JSONs + buffer management.

Extracts successful trajectories (reward=1.0) from tau2-bench simulation
results and converts them into SFT training samples compatible with the
GRPO training loop (using build_prompt_plus_action / logprob_action_tokens).

Each assistant turn in a successful trajectory becomes one SFT sample:
  - prompt_msgs: system + all prior messages (OpenAI chat format)
  - completion_text: JSON action string (matches GRPO rollout format)
  - tools: domain-specific tool schemas
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from adversarial_policy_game.constants import (
    AIRLINE_POLICY,
    RETAIL_POLICY,
    AIRLINE_TOOL_SCHEMAS,
    RETAIL_TOOL_SCHEMAS,
)
from adversarial_policy_game.game import compress_tool_schemas
from ppo import build_prompt_plus_action, pad_to_device


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SFTSample:
    """One assistant turn from a successful tau2-bench trajectory."""
    prompt_msgs: list          # System + prior messages (OpenAI chat format)
    completion_text: str       # JSON action string
    tools: list                # Tool schemas for this domain
    task_id: str               # For deduplication
    domain: str                # "airline" or "retail"


# ---------------------------------------------------------------------------
# System prompt construction (matches AdversarialPolicyGame.get_system_prompt)
# ---------------------------------------------------------------------------

def _build_system_prompt(policy: str) -> str:
    return (
        "<instructions>\n"
        "You are a customer service agent that helps the user according to the <policy> provided below.\n"
        "In each turn you can either:\n"
        "- Send a message to the user.\n"
        "- Make a tool call.\n"
        "You cannot do both at the same time.\n"
        "\n"
        "Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.\n"
        "</instructions>\n"
        "<policy>\n"
        f"{policy}\n"
        "</policy>"
    )


# ---------------------------------------------------------------------------
# Message conversion: tau2-bench → OpenAI chat format
# ---------------------------------------------------------------------------

def _convert_message(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a single tau2-bench message to OpenAI chat format.

    Strips tau2-specific fields (turn_idx, timestamp, cost, usage, raw_data,
    requestor, error) and returns clean OpenAI format.

    Returns None for messages that should be skipped (###STOP###, ###TRANSFER###).
    """
    role = msg["role"]
    content = msg.get("content")

    if role == "user":
        # Skip terminal signals
        if content and content.strip() in ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"):
            return None
        return {"role": "user", "content": content or ""}

    elif role == "assistant":
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            # Assistant tool call
            converted_tcs = []
            for tc in tool_calls:
                args = tc.get("arguments", {})
                # arguments can be dict or string
                if isinstance(args, dict):
                    args_str = json.dumps(args)
                else:
                    args_str = str(args)
                converted_tcs.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": args_str,
                    },
                })
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": converted_tcs,
            }
        else:
            # Assistant text message
            return {"role": "assistant", "content": content or ""}

    elif role == "tool":
        return {
            "role": "tool",
            "content": content or "",
            "tool_call_id": msg.get("id", ""),
        }

    return None


def _extract_thinking(content: str) -> Tuple[str, str]:
    """Split content into (thinking_prefix, clean_content).

    Handles Qwen3-style thinking where content may contain:
      - "<think>\\nreasoning...\\n</think>\\n\\nActual response"
      - "reasoning...\\n</think>\\n\\nActual response" (missing opening tag)

    Returns:
        (thinking_prefix, clean_content) where thinking_prefix is either
        "<think>\\n...\\n</think>\\n" or "" if no thinking found.
    """
    if not content or "</think>" not in content:
        return "", content or ""

    # Ensure opening tag
    if "<think>" not in content:
        content = "<think>\n" + content

    idx = content.index("</think>")
    think_part = content[:idx + len("</think>")]
    remaining = content[idx + len("</think>"):]
    # Strip leading whitespace/newlines from the actual response
    remaining = remaining.lstrip("\n")
    return think_part + "\n", remaining


def _make_completion_text(msg: Dict[str, Any]) -> str:
    """Create completion text for an assistant turn (matches GRPO rollout format).

    For tool calls: json.dumps({"name": ..., "arguments": ...})
    For text messages: json.dumps({"name": "respond_to_user", "arguments": {"message": ...}})

    If the message content contains <think> tags (e.g. from Qwen3-Max),
    the thinking is prepended BEFORE the JSON action so the model learns
    to reason before acting.
    """
    content = msg.get("content", "") or ""
    tool_calls = msg.get("tool_calls")

    # Extract thinking from content (if present)
    think_prefix, clean_content = _extract_thinking(content)

    if tool_calls:
        tc = tool_calls[0]  # tau2-bench uses single tool calls
        args = tc.get("arguments", {})
        # Arguments may be dict or string
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        action_json = json.dumps({"name": tc["name"], "arguments": args})
        return think_prefix + action_json
    else:
        action_json = json.dumps({"name": "respond_to_user", "arguments": {"message": clean_content}})
        return think_prefix + action_json


# ---------------------------------------------------------------------------
# SFT sample extraction
# ---------------------------------------------------------------------------

def load_sft_samples(
    path: str,
    compact_tools: bool = False,
) -> List[SFTSample]:
    """Load SFT samples from a tau2-bench eval JSON file.

    For each simulation with reward == 1.0, creates one SFTSample per
    assistant turn.

    Args:
        path: Path to tau2-bench eval JSON file.
        compact_tools: If True, strip descriptions from tool schemas.

    Returns:
        List of SFTSample objects.
    """
    path = Path(path)
    with open(path, "r") as f:
        data = json.load(f)

    # Get domain from environment_info
    env_info = data.get("info", {}).get("environment_info", {})
    domain = env_info.get("domain_name", "")

    # System prompt: use explicit one if provided, else build from policy
    explicit_system_prompt = env_info.get("system_prompt")
    if explicit_system_prompt:
        system_prompt = explicit_system_prompt
    else:
        policy_text = env_info.get("policy", "")
        if not domain or not policy_text:
            return []
        system_prompt = _build_system_prompt(policy_text)

    system_msg = {"role": "system", "content": system_prompt}

    # Tool schemas: use explicit ones if provided, else lookup by domain
    explicit_tools = env_info.get("tool_schemas")
    if explicit_tools is not None:
        tools = explicit_tools
    elif domain == "airline":
        tools = AIRLINE_TOOL_SCHEMAS
    elif domain == "retail":
        tools = RETAIL_TOOL_SCHEMAS
    else:
        tools = []

    if compact_tools and tools:
        tools = compress_tool_schemas(tools)

    # Extract from simulations
    simulations = data.get("simulations", [])
    samples: List[SFTSample] = []

    for sim in simulations:
        reward_info = sim.get("reward_info", {})
        reward = reward_info.get("reward", 0)
        if reward != 1.0:
            continue

        task_id = str(sim.get("task_id", ""))
        messages = sim.get("messages", [])

        # Build conversation incrementally, create sample at each assistant turn
        converted_history: List[Dict[str, Any]] = []

        for msg in messages:
            if msg["role"] == "assistant":
                # Create SFT sample: system + history so far is the prompt
                prompt_msgs = [system_msg] + list(converted_history)
                completion_text = _make_completion_text(msg)

                samples.append(SFTSample(
                    prompt_msgs=prompt_msgs,
                    completion_text=completion_text,
                    tools=tools,
                    task_id=task_id,
                    domain=domain,
                ))

            # Add this message to history (after creating the sample)
            converted = _convert_message(msg)
            if converted is not None:
                converted_history.append(converted)

    return samples


def load_sft_samples_multi(
    paths: List[str],
    compact_tools: bool = False,
    max_samples_per_file: int = None,
) -> List[SFTSample]:
    """Load SFT samples from multiple tau2-bench eval JSON files."""
    all_samples: List[SFTSample] = []
    for p in paths:
        try:
            samples = load_sft_samples(p, compact_tools=compact_tools)
            if max_samples_per_file is not None and len(samples) > max_samples_per_file:
                samples = samples[:max_samples_per_file]
                print(f"[SFT]   {p}: truncated to {max_samples_per_file} samples (--max-samples-per-file)")
            all_samples.extend(samples)
        except Exception as e:
            print(f"[SFT] Warning: failed to load {p}: {e}")
    return all_samples


# ---------------------------------------------------------------------------
# SFT Buffer
# ---------------------------------------------------------------------------

class SFTBuffer:
    """Pre-tokenized buffer of SFT samples for joint SFT+RL training.

    Loads successful tau2-bench trajectories, tokenizes them using
    build_prompt_plus_action, and provides random mini-batch sampling.
    """

    def __init__(
        self,
        paths: List[str],
        tokenizer,
        compact_tools: bool = False,
        max_samples_per_file: int = None,
        max_seq_len: int = None,
        min_prompt_len: int = 128,
    ):
        self._tokenizer = tokenizer
        self._compact_tools = compact_tools
        self._max_samples_per_file = max_samples_per_file
        self._max_seq_len = max_seq_len
        self._min_prompt_len = min_prompt_len
        self._seen_tasks: Set[str] = set()  # (domain, task_id) for dedup

        # Pre-tokenized storage
        self._seqs_cpu: List[torch.Tensor] = []
        self._prompt_lens: List[int] = []
        self._action_lens: List[int] = []

        # Stats
        self._truncated_count = 0
        self._dropped_count = 0

        # Load initial data
        if paths:
            self._load_and_tokenize(paths)
            if self._truncated_count or self._dropped_count:
                print(f"[SFT] Front-truncation: {self._truncated_count} truncated, "
                      f"{self._dropped_count} dropped (action too long for max_seq_len={max_seq_len})")

    @staticmethod
    def _task_key(domain: str, task_id: str) -> str:
        """Dedup key: one trajectory per unique (domain, task_id)."""
        return f"{domain}::{task_id}"

    def _load_and_tokenize(self, paths: List[str]) -> int:
        """Load samples from paths, dedup by (domain, task_id), tokenize.

        Only adds trajectories for task_ids not already in the buffer.
        This means the initial seed data is kept, and refresh() only
        adds genuinely new successful tasks.

        Returns count of new samples added.
        """
        new_count = 0
        for p in paths:
            try:
                samples = load_sft_samples(p, compact_tools=self._compact_tools)
            except Exception as e:
                print(f"[SFT] Warning: failed to load {p}: {e}")
                continue

            if self._max_samples_per_file is not None and len(samples) > self._max_samples_per_file:
                samples = samples[:self._max_samples_per_file]
                print(f"[SFT]   {p}: truncated to {self._max_samples_per_file} samples (--max-samples-per-file)")

            # Which task_ids in this file are new?
            new_task_ids = set()
            for sample in samples:
                key = self._task_key(sample.domain, sample.task_id)
                if key not in self._seen_tasks:
                    new_task_ids.add(sample.task_id)

            if not new_task_ids:
                continue

            # Tokenize only samples from new tasks
            for sample in samples:
                if sample.task_id not in new_task_ids:
                    continue
                try:
                    seq, pl, al = build_prompt_plus_action(
                        self._tokenizer,
                        sample.prompt_msgs,
                        sample.completion_text,
                        tools=sample.tools,
                    )
                    total_len = seq.shape[0]

                    # Front-truncate prompt if sequence exceeds max_seq_len
                    if self._max_seq_len and total_len > self._max_seq_len:
                        excess = total_len - self._max_seq_len
                        new_pl = pl - excess
                        if new_pl < self._min_prompt_len:
                            # Action is too long or prompt would be too short
                            self._dropped_count += 1
                            continue
                        seq = seq[excess:]
                        pl = new_pl
                        self._truncated_count += 1

                    self._seqs_cpu.append(seq)
                    self._prompt_lens.append(pl)
                    self._action_lens.append(al)
                    new_count += 1
                except Exception as e:
                    print(f"[SFT] Warning: tokenization failed for task {sample.task_id}: {e}")
                    continue

            # Mark new tasks as seen
            for tid in new_task_ids:
                domain = next(s.domain for s in samples if s.task_id == tid)
                self._seen_tasks.add(self._task_key(domain, tid))

        return new_count

    def refresh(self, new_paths: List[str]) -> None:
        """Extract reward=1.0 from new eval files, dedup, re-tokenize new samples."""
        new_paths_str = [str(p) for p in new_paths]
        new_count = self._load_and_tokenize(new_paths_str)
        if new_count > 0:
            print(f"[SFT] Added {new_count} new samples (total: {len(self)})")

    def sample_batch(
        self,
        batch_size: int,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
        """Random sample with replacement, pad and transfer to device.

        Returns:
            (input_ids, attention_mask, prompt_lens, action_lens)
        """
        n = len(self._seqs_cpu)
        indices = [random.randint(0, n - 1) for _ in range(batch_size)]

        seqs = [self._seqs_cpu[i] for i in indices]
        prompt_lens = [self._prompt_lens[i] for i in indices]
        action_lens = [self._action_lens[i] for i in indices]

        ids, attn = pad_to_device(seqs, self._tokenizer.pad_token_id, device)
        return ids, attn, prompt_lens, action_lens

    def __len__(self) -> int:
        return len(self._seqs_cpu)
