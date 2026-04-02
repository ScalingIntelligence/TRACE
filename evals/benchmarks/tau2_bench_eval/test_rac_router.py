#!/usr/bin/env python3
"""Retrieval-Augmented Classification (RAC) router.

Stage 1: Embedding retrieval selects the most relevant game trajectories
Stage 2: Classifier with descriptions + retrieved trajectories makes the decision

Tests exact match accuracy on tau2-bench airline tasks.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
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
    parser.add_argument("--corpus", required=True, help="trajectory_corpus.json")
    parser.add_argument("--config", required=True, help="Orchestrator config for descriptions")
    parser.add_argument("--domain", default="airline")
    parser.add_argument("--port", type=int, default=9002)
    parser.add_argument("--gpu", type=str, default="cuda:4")
    parser.add_argument("--emb-model", type=str, default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--retrieve-k", type=int, default=15,
                        help="Number of trajectories to retrieve per query")
    parser.add_argument("--task-ids", nargs="*", type=str, default=None)
    parser.add_argument("--solve-files", nargs="*", default=None)
    args = parser.parse_args()

    # Load corpus
    with open(args.corpus) as f:
        corpus = json.load(f)
    print(f"Corpus: {len(corpus)} trajectories")

    # Load config for skill descriptions
    with open(args.config) as f:
        config = yaml.safe_load(f)
    orch = config["orchestrator"]
    skills = orch["skills"]
    skill_names = [s["name"] for s in skills]
    skill_descs = {s["name"]: s.get("description", "").strip() for s in skills}

    # Load embedding model
    print(f"Loading embedding model on {args.gpu}...")
    from sentence_transformers import SentenceTransformer
    emb_model = SentenceTransformer(args.emb_model, device=args.gpu)

    # Embed corpus
    print("Embedding corpus...")
    corpus_texts = [t["text"] for t in corpus]
    corpus_embeddings = emb_model.encode(corpus_texts)
    print(f"Embedded {len(corpus_texts)} documents")

    # Parse solve files
    expected_routing = {}
    if args.solve_files:
        for spec in args.solve_files:
            skill_name, path = spec.split("=", 1)
            with open(path) as f:
                data = json.load(f)
            for sim in data["simulations"]:
                if sim.get("reward_info", {}).get("reward", 0) == 1:
                    expected_routing.setdefault(sim["task_id"], []).append(skill_name)

    # Get tasks
    if args.task_ids:
        task_ids = [str(t) for t in args.task_ids]
    elif expected_routing:
        task_ids = sorted(expected_routing.keys(), key=int)
    else:
        task_ids = None

    tasks = get_tasks(task_set_name=args.domain, task_ids=task_ids)

    # User sim setup
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

    # Set up classifier client
    from openai import OpenAI
    client = OpenAI(base_url=port_url, api_key="EMPTY")
    model_name = orch["orchestrator_model"]
    if model_name.startswith("openai/"):
        model_name = model_name[len("openai/"):]

    print(f"\nTesting RAC router (retrieve={args.retrieve_k}) on {len(tasks)} tasks\n")

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
            tools=user_tools,
            instructions=str(task.user_scenario),
            llm=user_llm,
            llm_args=deepcopy(user_llm_args),
        )
        user_sim.set_seed(task_seed)
        user_state = user_sim.get_init_state()
        greeting = deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
        user_msg, _ = user_sim.generate_next_message(greeting, user_state)

        # Stage 1: Retrieve top-K trajectories
        query_emb = emb_model.encode(
            [user_msg.content],
            prompt="Instruct: Given a customer service request, identify which skill category best handles it\nQuery: "
        )
        similarities = emb_model.similarity(query_emb, corpus_embeddings)[0]
        top_indices = similarities.argsort(descending=True)[:args.retrieve_k]

        retrieved = []
        for idx in top_indices:
            retrieved.append({
                "skill": corpus[idx]["skill"],
                "type": corpus[idx]["type"],
                "text": corpus[idx]["text"][:200],
                "score": float(similarities[idx]),
            })

        # Stage 2: Build classifier prompt with descriptions + retrieved examples
        # Build label mapping (same as orchestrator classifier)
        label_to_skill = {}
        label_lines = []
        for i, s in enumerate(skills):
            label = chr(ord("A") + i)
            label_to_skill[label] = s["name"]
            desc = skill_descs[s["name"]]
            label_lines.append(f"{label}: {s['name']} — {desc}")

        # Show TOP 1 match per skill — equal representation
        by_skill = {}
        for r in retrieved:
            if r["skill"] not in by_skill:
                by_skill[r["skill"]] = r

        # Only include retrieved examples for skills that have matches
        # Keep it minimal to let descriptions dominate
        examples_text = ""
        for skill_name in skill_names:
            if skill_name in by_skill:
                r = by_skill[skill_name]
                examples_text += f"\n  {skill_name} example: {r['text'][:120]}"

        classifier_prompt = (
            "You are a routing classifier. Given the customer's request and "
            "similar training scenarios retrieved for each skill, select the "
            "skill that best matches.\n\n"
            "Skills:\n" + "\n".join(label_lines) + "\n\n"
            "Retrieved similar scenarios:" + examples_text + "\n"
            "Only output the label (e.g. A, B, C)."
        )

        # Build messages
        classifier_messages = [
            {"role": "system", "content": classifier_prompt},
            {"role": "user", "content": user_msg.content},
            {"role": "user", "content": "Which skill should handle this?"},
        ]

        labels = list(label_to_skill.keys())

        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=classifier_messages,
                extra_body={"structured_outputs": {"choice": labels}},
                temperature=0.0,
                max_tokens=1,
                seed=task_seed,
            )
            chosen_label = completion.choices[0].message.content.strip()
            predicted_skill = label_to_skill.get(chosen_label, "general_service")
        except Exception as e:
            print(f"  T{tid}: ERROR {e}")
            continue

        # Check
        exp = expected_routing.get(tid)
        if exp:
            total += 1
            is_correct = predicted_skill in exp
            if is_correct:
                correct += 1
            status = "✓" if is_correct else "✗"

            # Show retrieval distribution
            ret_dist = Counter(r["skill"] for r in retrieved)
            dist_str = " ".join(f"{s}:{n}" for s, n in ret_dist.most_common())

            print(f"  T{tid}: {status}  got={predicted_skill:>30}  retrieved=[{dist_str}]")
            if not is_correct:
                print(f"       needed: {exp}")
        else:
            print(f"  T{tid}: -> {predicted_skill} (no solver)")

    if total:
        print(f"\n{'='*60}")
        print(f"Exact rate: {correct}/{total} ({100*correct/total:.1f}%)")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
