#!/usr/bin/env python3
"""Test embedding-based routing using gold trajectories as retrieval documents.

Each gold trajectory from the game scenario generators is embedded as a document.
At routing time, the user's conversation is embedded as a query and matched
against the trajectory documents. The skill of the best-matching trajectory
determines the routing.

This is topk=1 exact match: the single closest trajectory's skill is selected.
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
    parser.add_argument("--trajectories-json", required=True,
                        help="Path to gold_trajectories.json")
    parser.add_argument("--config", required=True,
                        help="Config YAML for user sim settings")
    parser.add_argument("--domain", default="airline")
    parser.add_argument("--port", type=int, default=9002)
    parser.add_argument("--gpu", type=str, default="cuda:4")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--task-ids", nargs="*", type=str, default=None)
    parser.add_argument("--solve-files", nargs="*", default=None)
    parser.add_argument("--topk", type=int, default=1)
    args = parser.parse_args()

    # Load gold trajectories
    with open(args.trajectories_json) as f:
        all_trajectories = json.load(f)

    # Build document list: each trajectory is a separate document, labeled with skill
    doc_texts = []
    doc_skills = []
    doc_labels = []
    for skill_name, trajs in all_trajectories.items():
        for traj in trajs:
            doc_texts.append(traj["text"])
            doc_skills.append(skill_name)
            doc_labels.append(f"{skill_name}/{traj['type']}: {traj['desc']}")

    print(f"Built {len(doc_texts)} trajectory documents across {len(all_trajectories)} skills")

    # Load embedding model
    print(f"Loading embedding model {args.model} on {args.gpu}...")
    from sentence_transformers import SentenceTransformer
    emb_model = SentenceTransformer(args.model, device=args.gpu)

    # Encode trajectory documents
    print("Encoding trajectory documents...")
    doc_embeddings = emb_model.encode(doc_texts)

    # Load config for user sim
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Parse solve files for expected routing
    expected_routing = {}
    if args.solve_files:
        solve_sets = {}
        for spec in args.solve_files:
            skill_name, path = spec.split("=", 1)
            with open(path) as f:
                data = json.load(f)
            solved = {sim["task_id"] for sim in data["simulations"]
                      if sim.get("reward_info", {}).get("reward", 0) == 1}
            solve_sets[skill_name] = solved
            print(f"  {skill_name}: {len(solved)} tasks solved")
        all_solved = set()
        for solved in solve_sets.values():
            all_solved |= solved
        for tid in all_solved:
            expected_routing[tid] = [s for s, solved in solve_sets.items() if tid in solved]
        print(f"  Solvable tasks: {len(expected_routing)}\n")

    # Get task IDs
    if args.task_ids:
        task_ids = [str(t) for t in args.task_ids]
    elif expected_routing:
        task_ids = sorted(expected_routing.keys(), key=int)
    else:
        task_ids = None

    tasks = get_tasks(task_set_name=args.domain, task_ids=task_ids)

    # User sim config
    port_url = f"http://localhost:{args.port}/v1"
    user_llm = config.get("user_llm", config.get("agent_llm"))
    if user_llm and user_llm.startswith("vllm://"):
        user_llm = "openai/" + user_llm[7:]
    user_llm_args = {"temperature": 0.0, **config.get("user_llm_args", {})}
    vllm_config = config.get("vllm", {})
    if vllm_config.get("base_url"):
        user_llm_args.setdefault("api_base", port_url)
        user_llm_args.setdefault("api_key", vllm_config.get("api_key", "EMPTY"))
    base_seed = config.get("seed", 42)

    from tau2.registry import registry
    env = registry.get_env_constructor(args.domain)()

    print(f"Testing embedding trajectory routing (topk={args.topk}) on {len(tasks)} tasks\n")

    correct = 0
    total = 0
    results = []

    for task in sorted(tasks, key=lambda t: int(t.id)):
        tid = task.id
        task_seed = get_task_seed(base_seed, 0, tid)

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

        # Encode user message as query
        query_embedding = emb_model.encode(
            [user_msg.content],
            prompt="Instruct: Given a customer service request, identify which skill category best handles it\nQuery: "
        )

        # Find closest trajectory
        similarities = emb_model.similarity(query_embedding, doc_embeddings)[0]

        # Get top-k matches
        top_indices = similarities.argsort(descending=True)[:args.topk]
        top_skills = [doc_skills[i] for i in top_indices]
        top_scores = [float(similarities[i]) for i in top_indices]
        top_labels = [doc_labels[i] for i in top_indices]

        # Majority vote for topk > 1, or direct for topk=1
        if args.topk == 1:
            predicted_skill = top_skills[0]
        else:
            from collections import Counter
            skill_counts = Counter(top_skills)
            predicted_skill = skill_counts.most_common(1)[0][0]

        # Check
        exp = expected_routing.get(tid)
        if exp:
            total += 1
            is_correct = predicted_skill in exp
            if is_correct:
                correct += 1
            status = "✓" if is_correct else "✗"
            print(f"  T{tid}: {status}  got={predicted_skill:>30}  match={top_labels[0][:50]:>50}  score={top_scores[0]:.4f}")
            if not is_correct:
                print(f"       needed: {exp}")
            results.append((tid, exp, predicted_skill, top_labels[0], top_scores[0]))
        else:
            print(f"  T{tid}: -> {predicted_skill:>30}  match={top_labels[0][:50]:>50}  score={top_scores[0]:.4f}  (no solver)")

    # Summary
    if total:
        print(f"\n{'='*60}")
        print(f"Exact rate: {correct}/{total} ({100*correct/total:.1f}%)")

        # Per-skill breakdown
        by_skill = defaultdict(lambda: {"correct": 0, "total": 0})
        for tid, exp, pred, _, _ in results:
            for skill in exp:
                by_skill[skill]["total"] += 1
                if pred in exp:
                    by_skill[skill]["correct"] += 1
        print("\nPer-skill:")
        for skill, counts in sorted(by_skill.items(), key=lambda x: -x[1]["total"]):
            print(f"  {skill:>30}: {counts['correct']}/{counts['total']}")

        print("\nMisclassified:")
        for tid, exp, pred, match, score in results:
            if pred not in exp:
                print(f"  T{tid}: needed={exp}, got={pred}, matched={match[:40]}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
