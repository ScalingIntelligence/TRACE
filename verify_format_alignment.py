#!/usr/bin/env python3
"""
Verify that the model sees IDENTICAL token sequences during GRPO training
and tau2-bench evaluation.

Tests:
  1. System prompt text matches tau2-bench exactly
  2. Tool schemas match between training and eval paths
  3. Message construction produces identical dicts
  4. Token-level prompt comparison (initial state)
  5. Token-level prompt comparison (multi-turn with tool calls)
  6. build_prompt_plus_action prompt prefix matches eval tokenization
  7. Special token IDs
  8. enable_thinking effect
  9. String→re-encode roundtrip vs direct tokenization

Usage:
    conda activate games
    python verify_format_alignment.py
"""
import sys
sys.path.insert(0, "/home/ubuntu/lambda-stanford/tarun/games")
sys.path.insert(0, "/home/ubuntu/lambda-stanford/tarun/games/tau2-bench/src")

import json
import torch

# ── imports from training pipeline ──────────────────────────────────
from config import Config
from inference import messages_for_game, tools_for_game, build_prompt_text
from ppo import build_prompt_plus_action

from adversarial_policy_game import AdversarialPolicyGame
from adversarial_policy_game.scenarios import generate_scenario
from adversarial_policy_game.database import get_pydantic_db
from adversarial_policy_game.tools import ToolExecutor
from adversarial_policy_game.constants import AIRLINE_POLICY, RETAIL_POLICY

# ── imports from tau2-bench ─────────────────────────────────────────
from tau2.agent.llm_agent import (
    SYSTEM_PROMPT as TAU2_SYSTEM_PROMPT_TEMPLATE,
    AGENT_INSTRUCTION as TAU2_AGENT_INSTRUCTION,
)

# ====================================================================
PASS = "✓ PASS"
FAIL = "✗ FAIL"
results = []


def check(name: str, passed: bool, detail: str = ""):
    tag = PASS if passed else FAIL
    results.append(passed)
    print(f"  {tag}: {name}")
    if detail:
        print(f"        {detail}")
    return passed


def heading(n: int, title: str):
    print(f"\n{'=' * 80}")
    print(f"TEST {n}: {title}")
    print("=" * 80)


# ====================================================================
def make_game(seed: int) -> AdversarialPolicyGame:
    """Create a game with manually-initialized state (no LLM user needed)."""
    game = AdversarialPolicyGame.__new__(AdversarialPolicyGame)
    game.max_steps = 30
    game._user_client = None  # won't call LLM user
    game.done = False
    game.current_player = 0
    game.rewards = {0: 0.0}
    game.invalid_player = None

    game._scenario = generate_scenario(seed)
    game._tools = ToolExecutor(
        game._scenario.domain,
        get_pydantic_db(game._scenario.domain),
    )
    game._conversation = [{"role": "user", "text": game._scenario.initial_message}]
    game._step_count = 0
    game._transferred = False
    game._pending_stop = False
    return game


def add_multi_turn(game: AdversarialPolicyGame):
    """Inject synthetic multi-turn conversation entries."""
    sc = game._scenario
    if sc.domain == "airline":
        game._conversation.extend([
            {
                "role": "tool_call",
                "text": json.dumps({
                    "name": "get_user_details",
                    "arguments": {"user_id": "john_doe_123"},
                }),
            },
            {
                "role": "tool_result",
                "text": json.dumps({
                    "user_id": "john_doe_123",
                    "name": "John Doe",
                    "email": "john@example.com",
                    "membership": "gold",
                    "reservation_numbers": ["ABC123", "DEF456"],
                }),
            },
            {"role": "assistant", "text": "I can see your account, John. How can I help you today?"},
            {"role": "user", "text": "I need to cancel reservation ABC123 right now. It's an emergency!"},
            {
                "role": "tool_call",
                "text": json.dumps({
                    "name": "get_reservation_details",
                    "arguments": {"reservation_id": "ABC123"},
                }),
            },
            {
                "role": "tool_result",
                "text": json.dumps({
                    "reservation_id": "ABC123",
                    "status": "active",
                    "cabin": "economy",
                    "insurance": "no",
                    "created_at": "2024-04-01",
                }),
            },
        ])
    else:
        game._conversation.extend([
            {
                "role": "tool_call",
                "text": json.dumps({
                    "name": "get_order_details",
                    "arguments": {"order_id": "#W1234567"},
                }),
            },
            {
                "role": "tool_result",
                "text": json.dumps({
                    "order_id": "#W1234567",
                    "status": "pending",
                    "items": [
                        {"name": "Laptop", "price": 999.99},
                        {"name": "Mouse", "price": 29.99},
                    ],
                }),
            },
            {"role": "assistant", "text": "I can see your order #W1234567. What would you like to do?"},
            {"role": "user", "text": "Cancel my order! I changed my mind!"},
        ])


# ====================================================================
def main():
    print("=" * 80)
    print("TRAINING ↔ EVAL FORMAT ALIGNMENT VERIFICATION")
    print("=" * 80)

    # ── Load tokenizer ──────────────────────────────────────────────
    print("\nLoading tokenizer...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_NAME, trust_remote_code=True,
    )
    print(f"  Model: {Config.MODEL_NAME}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Config.ENABLE_THINKING = {Config.ENABLE_THINKING}")

    # ── Test multiple seeds / domains ───────────────────────────────
    # Pick seeds that cover both airline and retail
    test_seeds = [0, 1, 2, 5, 10, 42, 99, 123]
    seen_domains = set()

    for seed in test_seeds:
        game = make_game(seed)
        sc = game._scenario
        if sc.domain in seen_domains and len(seen_domains) >= 2:
            continue
        seen_domains.add(sc.domain)

        print(f"\n{'#' * 80}")
        print(f"# SEED {seed}: T{sc.template_id}:{sc.template_name} | {sc.domain} | {sc.pressure_type.value}")
        print(f"{'#' * 80}")

        # ==============================================================
        # TEST 1: System prompt
        # ==============================================================
        heading(1, "System Prompt Text")

        game_system = game.get_system_prompt()
        tau2_system = TAU2_SYSTEM_PROMPT_TEMPLATE.format(
            domain_policy=AIRLINE_POLICY if sc.domain == "airline" else RETAIL_POLICY,
            agent_instruction=TAU2_AGENT_INSTRUCTION,
        )

        check("System prompt matches tau2-bench", game_system == tau2_system,
              f"game={len(game_system)} chars, tau2={len(tau2_system)} chars")

        if game_system != tau2_system:
            # Show first diff
            for i, (a, b) in enumerate(zip(game_system, tau2_system)):
                if a != b:
                    print(f"        First diff at char {i}:")
                    print(f"        Game: {repr(game_system[max(0,i-40):i+40])}")
                    print(f"        Tau2: {repr(tau2_system[max(0,i-40):i+40])}")
                    break

        # ==============================================================
        # TEST 2: Tool schemas
        # ==============================================================
        heading(2, "Tool Schemas")

        game_tools = game.get_tool_schemas()
        inf_tools = tools_for_game(game)

        check("get_tool_schemas() == tools_for_game()",
              json.dumps(game_tools) == json.dumps(inf_tools),
              f"{len(game_tools)} tools for {sc.domain}")

        # ==============================================================
        # TEST 3: Message dicts (initial state)
        # ==============================================================
        heading(3, "Message Dicts — Initial State")

        train_msgs = messages_for_game(0, "", None, env=game)
        eval_msgs = [{"role": "system", "content": game.get_system_prompt()}] + game.get_messages()

        check("messages_for_game() == eval path messages",
              train_msgs == eval_msgs,
              f"train={len(train_msgs)} msgs, eval={len(eval_msgs)} msgs")

        if train_msgs != eval_msgs:
            for i in range(max(len(train_msgs), len(eval_msgs))):
                t = train_msgs[i] if i < len(train_msgs) else "<missing>"
                e = eval_msgs[i] if i < len(eval_msgs) else "<missing>"
                if t != e:
                    print(f"        Msg {i} TRAIN: {json.dumps(t, default=str)[:200]}")
                    print(f"        Msg {i} EVAL : {json.dumps(e, default=str)[:200]}")

        # ==============================================================
        # TEST 4: Token-level (initial state)
        # ==============================================================
        heading(4, "Token-Level — Initial State")

        train_tools = tools_for_game(game)
        eval_tools = game.get_tool_schemas()

        # String prompts
        train_str = build_prompt_text(tokenizer, train_msgs, tools=train_tools)
        eval_str = tokenizer.apply_chat_template(
            eval_msgs, tools=eval_tools,
            tokenize=False, add_generation_prompt=True,
            enable_thinking=Config.ENABLE_THINKING,
        )
        check("Prompt strings identical (initial)",
              train_str == eval_str,
              f"{len(train_str)} chars")

        # Token IDs via apply_chat_template
        train_ids = tokenizer.apply_chat_template(
            train_msgs, tools=train_tools,
            add_generation_prompt=True, return_tensors="pt",
            enable_thinking=Config.ENABLE_THINKING,
        )[0]
        eval_ids = tokenizer.apply_chat_template(
            eval_msgs, tools=eval_tools,
            add_generation_prompt=True, return_tensors="pt",
            enable_thinking=Config.ENABLE_THINKING,
        )[0]
        check("Token IDs identical (initial)",
              torch.equal(train_ids, eval_ids),
              f"{len(train_ids)} tokens")

        if not torch.equal(train_ids, eval_ids):
            for i in range(min(len(train_ids), len(eval_ids))):
                if train_ids[i] != eval_ids[i]:
                    print(f"        First diff at pos {i}: train={train_ids[i].item()} eval={eval_ids[i].item()}")
                    ctx = 10
                    print(f"        Train[{i-ctx}:{i+ctx}]: {tokenizer.decode(train_ids[max(0,i-ctx):i+ctx])}")
                    print(f"        Eval [{i-ctx}:{i+ctx}]: {tokenizer.decode(eval_ids[max(0,i-ctx):i+ctx])}")
                    break

        # ==============================================================
        # TEST 5: Multi-turn conversation
        # ==============================================================
        heading(5, "Multi-Turn Conversation")

        add_multi_turn(game)

        train_msgs_mt = messages_for_game(0, "", None, env=game)
        eval_msgs_mt = [{"role": "system", "content": game.get_system_prompt()}] + game.get_messages()
        train_tools_mt = tools_for_game(game)
        eval_tools_mt = game.get_tool_schemas()

        check("Multi-turn messages match",
              train_msgs_mt == eval_msgs_mt,
              f"train={len(train_msgs_mt)} msgs, eval={len(eval_msgs_mt)} msgs")

        if train_msgs_mt != eval_msgs_mt:
            for i in range(max(len(train_msgs_mt), len(eval_msgs_mt))):
                t = train_msgs_mt[i] if i < len(train_msgs_mt) else "<missing>"
                e = eval_msgs_mt[i] if i < len(eval_msgs_mt) else "<missing>"
                if t != e:
                    print(f"        Msg {i} TRAIN: {json.dumps(t, default=str)[:200]}")
                    print(f"        Msg {i} EVAL : {json.dumps(e, default=str)[:200]}")

        # String-level
        train_str_mt = build_prompt_text(tokenizer, train_msgs_mt, tools=train_tools_mt)
        eval_str_mt = tokenizer.apply_chat_template(
            eval_msgs_mt, tools=eval_tools_mt,
            tokenize=False, add_generation_prompt=True,
            enable_thinking=Config.ENABLE_THINKING,
        )
        check("Multi-turn prompt strings identical",
              train_str_mt == eval_str_mt,
              f"{len(train_str_mt)} chars")

        if train_str_mt != eval_str_mt:
            for i, (a, b) in enumerate(zip(train_str_mt, eval_str_mt)):
                if a != b:
                    ctx = 60
                    print(f"        First diff at char {i}:")
                    print(f"        Train: {repr(train_str_mt[max(0,i-ctx):i+ctx])}")
                    print(f"        Eval : {repr(eval_str_mt[max(0,i-ctx):i+ctx])}")
                    break

        # Token IDs
        train_ids_mt = tokenizer.apply_chat_template(
            train_msgs_mt, tools=train_tools_mt,
            add_generation_prompt=True, return_tensors="pt",
            enable_thinking=Config.ENABLE_THINKING,
        )[0]
        eval_ids_mt = tokenizer.apply_chat_template(
            eval_msgs_mt, tools=eval_tools_mt,
            add_generation_prompt=True, return_tensors="pt",
            enable_thinking=Config.ENABLE_THINKING,
        )[0]
        check("Multi-turn token IDs identical",
              torch.equal(train_ids_mt, eval_ids_mt),
              f"{len(train_ids_mt)} tokens")

        # ==============================================================
        # TEST 6: build_prompt_plus_action prompt prefix
        # ==============================================================
        heading(6, "build_prompt_plus_action Prompt Prefix")

        sample_action = (
            '<tool_call>\n'
            '{"name": "get_reservation_details", "arguments": {"reservation_id": "XYZ789"}}\n'
            '</tool_call>'
        )

        full_ids, prompt_len, action_len = build_prompt_plus_action(
            tokenizer, train_msgs_mt, sample_action, tools=train_tools_mt,
        )
        bpa_prompt_ids = full_ids[:prompt_len]

        check("build_prompt_plus_action prefix matches eval tokenization",
              torch.equal(bpa_prompt_ids, eval_ids_mt),
              f"prompt={prompt_len} tokens, action={action_len} tokens")

        if not torch.equal(bpa_prompt_ids, eval_ids_mt):
            print(f"        BPA prompt:  {len(bpa_prompt_ids)} tokens")
            print(f"        Direct eval: {len(eval_ids_mt)} tokens")
            for i in range(min(len(bpa_prompt_ids), len(eval_ids_mt))):
                if bpa_prompt_ids[i] != eval_ids_mt[i]:
                    print(f"        First diff at pos {i}: bpa={bpa_prompt_ids[i].item()} eval={eval_ids_mt[i].item()}")
                    break

        # ==============================================================
        # TEST 7: String→re-encode roundtrip
        # ==============================================================
        heading(7, "String → Re-encode Roundtrip")

        # build_prompt_text returns a string; vLLM /completions receives this string.
        # Verify that encoding this string produces the same IDs as direct tokenization.
        roundtrip_ids = tokenizer.encode(train_str_mt, return_tensors="pt")[0]

        # Direct tokenization (what vLLM chat/completions would do internally)
        direct_ids = eval_ids_mt

        check("String roundtrip tokens == direct tokenization",
              torch.equal(roundtrip_ids, direct_ids),
              f"roundtrip={len(roundtrip_ids)} tokens, direct={len(direct_ids)} tokens")

        if not torch.equal(roundtrip_ids, direct_ids):
            # This is expected to potentially differ due to special token handling
            # Let's find where
            for i in range(min(len(roundtrip_ids), len(direct_ids))):
                if roundtrip_ids[i] != direct_ids[i]:
                    print(f"        First diff at pos {i}:")
                    print(f"        Roundtrip: {roundtrip_ids[i].item()} = {repr(tokenizer.decode([roundtrip_ids[i]]))}")
                    print(f"        Direct:    {direct_ids[i].item()} = {repr(tokenizer.decode([direct_ids[i]]))}")
                    # Show more context
                    ctx = 5
                    print(f"        Roundtrip[{max(0,i-ctx)}:{i+ctx}]: {tokenizer.decode(roundtrip_ids[max(0,i-ctx):i+ctx])}")
                    print(f"        Direct   [{max(0,i-ctx)}:{i+ctx}]: {tokenizer.decode(direct_ids[max(0,i-ctx):i+ctx])}")
                    break

    # ==================================================================
    # TEST 8: enable_thinking effect (run once)
    # ==================================================================
    heading(8, "enable_thinking Effect")

    game_t8 = make_game(42)
    msgs_t8 = [{"role": "system", "content": game_t8.get_system_prompt()}] + game_t8.get_messages()
    tools_t8 = game_t8.get_tool_schemas()

    with_thinking = tokenizer.apply_chat_template(
        msgs_t8, tools=tools_t8,
        tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )
    without_thinking = tokenizer.apply_chat_template(
        msgs_t8, tools=tools_t8,
        tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )

    print(f"  With thinking ends:    ...{repr(with_thinking[-100:])}")
    print(f"  Without thinking ends: ...{repr(without_thinking[-100:])}")
    print(f"  Length diff: {len(with_thinking) - len(without_thinking)} chars")

    if with_thinking != without_thinking:
        check("enable_thinking changes prompt (expected for Qwen3)", True,
              f"Config.ENABLE_THINKING={Config.ENABLE_THINKING} — eval must match!")
    else:
        check("enable_thinking has NO effect", True,
              "No alignment concern")

    # ==================================================================
    # TEST 9: Special tokens
    # ==================================================================
    heading(9, "Special Tokens")

    for token_name in ["<tool_call>", "</tool_call>", "<think>", "</think>",
                       "<|im_start|>", "<|im_end|>"]:
        tid = tokenizer.convert_tokens_to_ids(token_name)
        print(f"  {token_name:20s} → ID {tid}")

    # Check generation prompt ending
    game_t9 = make_game(42)
    msgs_t9 = messages_for_game(0, "", None, env=game_t9)
    tools_t9 = tools_for_game(game_t9)
    ids_t9 = tokenizer.apply_chat_template(
        msgs_t9, tools=tools_t9,
        add_generation_prompt=True, return_tensors="pt",
        enable_thinking=Config.ENABLE_THINKING,
    )[0]
    last_10 = ids_t9[-10:].tolist()
    last_decoded = [(tid, repr(tokenizer.decode([tid]))) for tid in last_10]
    print(f"\n  Last 10 prompt tokens:")
    for tid, decoded in last_decoded:
        print(f"    {tid:8d} → {decoded}")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY")
    print("=" * 80)

    passed = sum(results)
    total = len(results)
    print(f"\n  {passed}/{total} checks passed")

    if passed == total:
        print("\n  ALL CHECKS PASSED")
        print("  → Training and eval produce IDENTICAL token sequences.")
        print("  → The model sees the exact same prompt during GRPO training")
        print("    and tau2-bench function-calling evaluation.")
    else:
        failed = total - passed
        print(f"\n  {failed} CHECK(S) FAILED — see details above.")

    print(f"\n  IMPORTANT: Ensure your vLLM eval server configuration matches:")
    print(f"    enable_thinking = {Config.ENABLE_THINKING}")
    print(f"    model = {Config.MODEL_NAME}")


if __name__ == "__main__":
    main()
