#!/usr/bin/env python3
"""RAC Router v2: Top-K trajectories + similarity threshold → deduplicate to skills → classifier.

1. Retrieve top-K most similar game trajectories
2. Filter by minimum similarity threshold
3. Deduplicate to unique skills (each skill gets its best-matching trajectory)
4. Classifier picks from only these candidate skills (with descriptions + examples)
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import yaml

tau2_src = Path(__file__).resolve().parent.parent.parent.parent / "tau2-bench" / "src"
sys.path.insert(0, str(tau2_src))
tau2_data = Path(__file__).resolve().parent.parent.parent.parent / "tau2-bench" / "data"
os.environ.setdefault("TAU2_DATA_DIR", str(tau2_data))

from tau2.data_model.message import UserMessage
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
from tau2.run import get_tasks
from tau2.user.user_simulator import UserSimulator


def get_task_seed(base_seed, trial, task_id):
    combined = f"tau2_seed_{base_seed}_trial_{trial}_task_{task_id}"
    hash_bytes = hashlib.md5(combined.encode()).digest()
    return int.from_bytes(hash_bytes[:4], "big") % 1000000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--domain", default="airline")
    parser.add_argument("--port", type=int, default=9002)
    parser.add_argument("--gpu", type=str, default="cuda:4")
    parser.add_argument("--emb-model", type=str, default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--task-ids", nargs="*", type=str, default=None)
    parser.add_argument("--solve-files", nargs="*", default=None)
    parser.add_argument("--retrieval-cache", type=str, default=None,
                        help="Path to save/load retrieval results. "
                             "If file exists, skip embedding and load cached results. "
                             "If not, run embedding and save results.")
    parser.add_argument("--example-chars", type=int, default=None,
                        help="Max chars for example text in classifier prompt (default: full)")
    parser.add_argument("--routing-mode", type=str, default="classifier",
                        choices=["classifier", "llm"],
                        help="classifier: single-token structured output; llm: chain-of-thought reasoning")
    args = parser.parse_args()

    with open(args.corpus) as f:
        corpus = json.load(f)
    with open(args.config) as f:
        config = yaml.safe_load(f)

    orch = config["orchestrator"]
    skill_descs = {s["name"]: s.get("description", "").strip() for s in orch["skills"]}

    print(f"Corpus: {len(corpus)} trajectories")

    # Load or skip embedding model based on cache
    retrieval_cache = {}
    use_cache = args.retrieval_cache and os.path.exists(args.retrieval_cache)
    if use_cache:
        with open(args.retrieval_cache) as f:
            retrieval_cache = json.load(f)
        print(f"Loaded retrieval cache: {len(retrieval_cache)} tasks")
        emb_model = None
        corpus_embeddings = None
    else:
        print(f"Loading embedding model on {args.gpu}...")
        from sentence_transformers import SentenceTransformer
        emb_model = SentenceTransformer(
            args.emb_model, device=args.gpu,
            tokenizer_kwargs={"padding_side": "left"},
        )
        corpus_texts = [t["text"] for t in corpus]
        corpus_embeddings = emb_model.encode(corpus_texts)

    expected_routing = {}
    if args.solve_files:
        for spec in args.solve_files:
            skill_name, path = spec.split("=", 1)
            with open(path) as f:
                data = json.load(f)
            for sim in data["simulations"]:
                if sim.get("reward_info", {}).get("reward", 0) == 1:
                    expected_routing.setdefault(sim["task_id"], []).append(skill_name)

    if args.task_ids:
        task_ids = [str(t) for t in args.task_ids]
    elif expected_routing:
        task_ids = sorted(expected_routing.keys(), key=int)
    else:
        task_ids = None

    tasks = get_tasks(task_set_name=args.domain, task_ids=task_ids)

    port_url = f"http://localhost:{args.port}/v1"
    user_llm = config.get("user_llm", config.get("agent_llm"))
    if user_llm and user_llm.startswith("vllm://"):
        user_llm = "openai/" + user_llm[7:]
    user_llm_args = {"temperature": 0.0, **config.get("user_llm_args", {})}
    vllm_config = config.get("vllm", {})
    if vllm_config.get("base_url"):
        user_llm_args.setdefault("api_base", port_url)
        user_llm_args.setdefault("api_key", "EMPTY")

    from tau2.registry import registry
    env = registry.get_env_constructor(args.domain)()

    from openai import OpenAI
    client = OpenAI(base_url=port_url, api_key="EMPTY")
    model_name = orch["orchestrator_model"]
    if model_name.startswith("openai/"):
        model_name = model_name[len("openai/"):]

    print(f"Testing RAC v2 (topk={args.topk}, threshold={args.threshold}) on {len(tasks)} tasks\n")

    correct = 0
    total = 0

    for task in sorted(tasks, key=lambda t: int(t.id)):
        tid = task.id
        task_seed = get_task_seed(config.get("seed", 42), 0, tid)

        try:
            user_tools = env.get_user_tools()
        except Exception:
            user_tools = None
        user_sim = UserSimulator(
            tools=user_tools, instructions=str(task.user_scenario),
            llm=user_llm, llm_args=deepcopy(user_llm_args),
        )
        user_sim.set_seed(task_seed)
        user_state = user_sim.get_init_state()
        greeting = deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
        user_msg, _ = user_sim.generate_next_message(greeting, user_state)

        # Retrieve top-K trajectories (from cache or embedding model)
        if use_cache and tid in retrieval_cache:
            # Load cached: {skill: [score, corpus_idx]}
            cached = retrieval_cache[tid]
            candidates = {}
            for skill, (score, cidx) in cached.items():
                text = corpus[cidx]["text"]
                if args.example_chars:
                    text = text[:args.example_chars]
                candidates[skill] = (score, text)
        else:
            query_emb = emb_model.encode(
                [user_msg.content],
                prompt="Instruct: Given a customer service request, identify which skill category best handles it\nQuery: "
            )
            similarities = emb_model.similarity(query_emb, corpus_embeddings)[0]
            top_indices = similarities.argsort(descending=True)[:args.topk]

            # Filter by threshold and deduplicate to skills (best corpus idx per skill)
            candidate_idx = {}  # skill -> (score, corpus_idx)
            for idx in top_indices:
                score = float(similarities[idx])
                if score < args.threshold:
                    continue
                skill = corpus[idx]["skill"]
                if skill not in candidate_idx or score > candidate_idx[skill][0]:
                    candidate_idx[skill] = (score, int(idx))

            # If nothing passes threshold, take the single best
            if not candidate_idx:
                best_idx = int(top_indices[0])
                skill = corpus[best_idx]["skill"]
                candidate_idx[skill] = (float(similarities[best_idx]), best_idx)

            # Save to cache
            retrieval_cache[tid] = {s: [sc, ci] for s, (sc, ci) in candidate_idx.items()}

            # Build candidates with text
            candidates = {}
            for skill, (score, cidx) in candidate_idx.items():
                text = corpus[cidx]["text"]
                if args.example_chars:
                    text = text[:args.example_chars]
                candidates[skill] = (score, text)

        # Build classifier prompt with only candidate skills
        # Sort by retrieval score descending so highest-match is label A
        label_to_skill = {}
        prompt_lines = []
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)
        for i, (skill, (score, example)) in enumerate(sorted_candidates):
            label = chr(ord("A") + i)
            label_to_skill[label] = skill
            desc = skill_descs.get(skill, "")
            prompt_lines.append(f"{label}: {skill} (relevance: {score:.2f}) — {desc}")
            prompt_lines.append(f"   Similar scenario: {example}")

        # If only 1 candidate, skip classifier/llm
        if len(candidates) == 1:
            predicted_skill = list(candidates.keys())[0]
        elif args.routing_mode == "llm":
            # LLM mode: chain-of-thought reasoning then SELECTED_SKILL:
            llm_prompt = (
                "You are a routing classifier. Analyze the customer's request and "
                "select which skill best matches.\n\n"
                + "\n".join(prompt_lines)
                + "\n\nThink step by step about which skill is most appropriate, "
                "then end your response with exactly:\nSELECTED_SKILL: <skill_name>"
            )
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": llm_prompt},
                        {"role": "user", "content": user_msg.content},
                    ],
                    temperature=0.0, max_tokens=256, seed=task_seed,
                )
                response = completion.choices[0].message.content.strip()
                # Extract SELECTED_SKILL: <name>
                import re
                match = re.search(r"SELECTED_SKILL:\s*(\S+)", response)
                if match:
                    selected = match.group(1).strip()
                    # Match against candidate skill names
                    if selected in candidates:
                        predicted_skill = selected
                    else:
                        # Try label match
                        predicted_skill = label_to_skill.get(selected, list(candidates.keys())[0])
                else:
                    predicted_skill = list(candidates.keys())[0]
            except Exception as e:
                predicted_skill = list(candidates.keys())[0]
        else:
            # Classifier mode: single-token structured output
            classifier_prompt = (
                "You are a routing classifier. Select which skill best matches "
                "the customer's request.\n\n"
                + "\n".join(prompt_lines)
                + "\n\nOnly output the label."
            )

            labels = list(label_to_skill.keys())
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": classifier_prompt},
                        {"role": "user", "content": user_msg.content},
                        {"role": "user", "content": "Which skill?"},
                    ],
                    extra_body={"structured_outputs": {"choice": labels}},
                    logprobs=True,
                    top_logprobs=max(len(labels) - 1, 1),
                    temperature=0.0, max_tokens=1, seed=task_seed,
                )
                chosen_label = completion.choices[0].message.content.strip()
                predicted_skill = label_to_skill.get(chosen_label, list(candidates.keys())[0])
            except Exception as e:
                predicted_skill = list(candidates.keys())[0]

        # Check
        exp = expected_routing.get(tid)
        cand_str = " ".join(f"{s}:{sc:.2f}" for s, (sc, _) in candidates.items())
        if exp:
            total += 1
            is_correct = predicted_skill in exp
            if is_correct:
                correct += 1
            status = "✓" if is_correct else "✗"
            print(f"  T{tid}: {status}  got={predicted_skill:>25}  candidates=[{cand_str}]  n={len(candidates)}")
            if not is_correct:
                print(f"       needed: {exp}")
        else:
            print(f"  T{tid}: -> {predicted_skill:>25}  candidates=[{cand_str}]  n={len(candidates)}")

    if total:
        print(f"\n{'='*60}")
        print(f"Exact rate: {correct}/{total} ({100*correct/total:.1f}%)")
        print(f"{'='*60}")

    # Save retrieval cache if requested and we computed new retrievals
    if args.retrieval_cache and not use_cache and retrieval_cache:
        with open(args.retrieval_cache, "w") as f:
            json.dump(retrieval_cache, f, indent=2)
        print(f"\nSaved retrieval cache to {args.retrieval_cache} ({len(retrieval_cache)} tasks)")


if __name__ == "__main__":
    main()
