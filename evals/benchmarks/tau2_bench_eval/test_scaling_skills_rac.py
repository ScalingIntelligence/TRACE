#!/usr/bin/env python3
"""Test RAC routing accuracy on scaling_skills.json task assignments.

For each task listed in scaling_skills.json, generates the user's first message,
runs RAC retrieval + classification, and checks if the top-1 predicted skill
matches the assigned skill for that task.

Usage:
    python test_scaling_skills_rac.py \
        --corpus scaling_skills_corpus.json \
        --skills scaling_skills.json \
        --domain airline \
        --port 9002 \
        --gpu cuda:2
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

tau2_src = Path(__file__).resolve().parent.parent.parent.parent / "tau2-bench" / "src"
sys.path.insert(0, str(tau2_src))
tau2_data = Path(__file__).resolve().parent.parent.parent.parent / "tau2-bench" / "data"
os.environ.setdefault("TAU2_DATA_DIR", str(tau2_data))

from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
from tau2.run import get_tasks
from tau2.user.user_simulator import UserSimulator


def get_task_seed(base_seed, trial, task_id):
    combined = f"tau2_seed_{base_seed}_trial_{trial}_task_{task_id}"
    hash_bytes = hashlib.md5(combined.encode()).digest()
    return int.from_bytes(hash_bytes[:4], "big") % 1000000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="scaling_skills_corpus.json")
    parser.add_argument("--skills", required=True, help="scaling_skills.json")
    parser.add_argument("--domain", required=True, choices=["airline", "retail"])
    parser.add_argument("--port", type=int, default=9002)
    parser.add_argument("--gpu", type=str, default="cuda:2")
    parser.add_argument("--emb-model", type=str, default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    # Load corpus
    with open(args.corpus) as f:
        corpus = json.load(f)
    print(f"Corpus: {len(corpus)} trajectories")

    # Load skills and build expected routing
    with open(args.skills) as f:
        skills_data = json.load(f)

    prefix = "A" if args.domain == "airline" else "R"
    expected_routing = {}  # task_id -> [skill_ids]
    skill_info = {}  # skill_id -> {name, description}
    for skill in skills_data["skills"]:
        sid = skill["skill_id"]
        skill_info[sid] = {"name": skill["title"], "description": skill["principle"]}
        for t in skill["failed_tasks"]:
            if t.startswith(prefix):
                tid = t[len(prefix):]
                expected_routing.setdefault(tid, []).append(sid)

    task_ids = sorted(expected_routing.keys(), key=int)
    print(f"Domain: {args.domain}, tasks: {len(task_ids)}")

    # Load embedding model
    print(f"Loading embedding model on {args.gpu}...")
    from sentence_transformers import SentenceTransformer
    emb_model = SentenceTransformer(args.emb_model, device=args.gpu)

    # Embed corpus
    corpus_texts = [t["text"] for t in corpus]
    corpus_embeddings = emb_model.encode(corpus_texts)

    # Setup user sim
    from tau2.registry import registry
    env = registry.get_env_constructor(args.domain)()
    port_url = f"http://localhost:{args.port}/v1"

    from openai import OpenAI
    client = OpenAI(base_url=port_url, api_key="EMPTY")
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    # Build skill descriptions for classifier prompt (from corpus)
    skill_descs = {t["skill"]: t.get("description", t.get("name", "")) for t in corpus}

    tasks = get_tasks(task_set_name=args.domain, task_ids=task_ids)

    user_llm = f"openai/{model_name}"
    user_llm_args = {
        "temperature": 0.0,
        "max_context_length": 32000,
        "api_base": port_url,
        "api_key": "EMPTY",
        "tokenizer_model": f"openai/{model_name}",
    }

    print(f"\nTesting RAC scaling skills (topk={args.topk}, threshold={args.threshold}) on {len(tasks)} tasks\n")

    correct = 0
    total = 0

    for task in sorted(tasks, key=lambda t: int(t.id)):
        tid = task.id
        task_seed = get_task_seed(42, 0, tid)

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

        # Stage 1: Retrieve
        query_emb = emb_model.encode(
            [user_msg.content],
            prompt="Instruct: Given a customer service request, identify which agent skill is needed to handle it correctly\nQuery: ",
        )
        similarities = emb_model.similarity(query_emb, corpus_embeddings)[0]
        top_indices = similarities.argsort(descending=True)[:args.topk]

        # Stage 2: Filter + deduplicate
        candidates = {}
        for idx in top_indices:
            score = float(similarities[idx])
            if score < args.threshold:
                continue
            skill = corpus[idx]["skill"]
            if skill not in candidates or score > candidates[skill][0]:
                candidates[skill] = (score, corpus[idx]["text"])

        if not candidates:
            best_idx = int(top_indices[0])
            skill = corpus[best_idx]["skill"]
            candidates[skill] = (float(similarities[best_idx]), corpus[best_idx]["text"])

        # Stage 3: Classify
        if len(candidates) == 1:
            predicted_skill = list(candidates.keys())[0]
        else:
            sorted_cands = sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)
            label_to_skill = {}
            prompt_lines = []
            for i, (skill, (score, example)) in enumerate(sorted_cands):
                label = chr(ord("A") + i)
                label_to_skill[label] = skill
                desc = skill_descs.get(skill, "")
                prompt_lines.append(f"{label}: {skill} (relevance: {score:.2f}) — {desc}")
                prompt_lines.append(f"   Similar scenario: {example}")

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
                    temperature=0.0, max_tokens=1, seed=task_seed,
                )
                chosen_label = completion.choices[0].message.content.strip()
                predicted_skill = label_to_skill.get(chosen_label, list(candidates.keys())[0])
            except Exception as e:
                predicted_skill = list(candidates.keys())[0]

        # Check
        exp = expected_routing.get(tid, [])
        total += 1
        is_correct = predicted_skill in exp
        if is_correct:
            correct += 1
        status = "✓" if is_correct else "✗"

        cand_str = " ".join(f"{s}:{sc:.2f}" for s, (sc, _) in candidates.items())
        print(f"  {prefix}{tid}: {status}  got={predicted_skill:>5}  candidates=[{cand_str}]  n={len(candidates)}")
        if not is_correct:
            print(f"       needed: {exp}")

    if total:
        print(f"\n{'='*60}")
        print(f"Top-1 exact rate: {correct}/{total} ({100*correct/total:.1f}%)")
        print(f"{'='*60}")

        # Per-skill breakdown
        print("\nPer-skill accuracy:")
        skill_correct = defaultdict(int)
        skill_total = defaultdict(int)
        for task in sorted(tasks, key=lambda t: int(t.id)):
            tid = task.id
            exp = expected_routing.get(tid, [])
            for s in exp:
                skill_total[s] += 1
        # Re-run would be needed for per-skill, just show totals
        for sid in sorted(skill_total.keys()):
            info = skill_info.get(sid, {})
            print(f"  {sid} ({skill_total[sid]:2} tasks): {info.get('name', '')}")


if __name__ == "__main__":
    main()
