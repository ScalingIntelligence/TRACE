#!/usr/bin/env python3
"""Measure RAC routing accuracy at 1, 2, 4, 8, 16 skills.

Reports two metrics per skill count:
  - top-1 solved: RAC correctly identifies the skill
  - oracle solved: correct skill appears anywhere in the candidate set
"""

import argparse
import hashlib
import json
import os
import sys
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
    return int.from_bytes(hashlib.md5(combined.encode()).digest()[:4], "big") % 1000000


def run_rac(corpus, skill_descs, tasks, user_sim_factory, emb_model, corpus_embeddings,
            client, model_name, topk, threshold):
    """Run RAC and return list of (tid, needed, predicted, candidates)."""
    results = []
    for task, (tid, needed) in tasks:
        task_seed = get_task_seed(42, 0, tid)
        user_sim = user_sim_factory(task, task_seed)
        user_state = user_sim.get_init_state()
        greeting = deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
        user_msg, _ = user_sim.generate_next_message(greeting, user_state)

        query_emb = emb_model.encode(
            [user_msg.content],
            prompt="Instruct: Given a customer service request, identify which agent skill is needed to handle it correctly\nQuery: ",
        )
        similarities = emb_model.similarity(query_emb, corpus_embeddings)[0]
        top_indices = similarities.argsort(descending=True)[:topk]

        candidates = {}
        for idx in top_indices:
            score = float(similarities[idx])
            if score < threshold:
                continue
            skill = corpus[idx]["skill"]
            if skill not in candidates or score > candidates[skill][0]:
                candidates[skill] = (score, corpus[idx]["text"])

        if not candidates:
            best_idx = int(top_indices[0])
            skill = corpus[best_idx]["skill"]
            candidates[skill] = (float(similarities[best_idx]), corpus[best_idx]["text"])

        if len(candidates) == 1:
            predicted = list(candidates.keys())[0]
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
                predicted = label_to_skill.get(chosen_label, list(candidates.keys())[0])
            except Exception:
                predicted = list(candidates.keys())[0]

        results.append((tid, needed, predicted, list(candidates.keys())))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-airline", required=True)
    parser.add_argument("--corpus-retail", required=True)
    parser.add_argument("--skills", required=True)
    parser.add_argument("--port", type=int, default=9002)
    parser.add_argument("--gpu", type=str, default="cuda:2")
    parser.add_argument("--emb-model", type=str, default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    with open(args.skills) as f:
        skills_data = json.load(f)

    # Sort skills by coverage descending
    sorted_skills = sorted(skills_data["skills"], key=lambda s: int(s["skill_id"][1:]))
    skill_order = [s["skill_id"] for s in sorted_skills]

    # Load corpora
    with open(args.corpus_airline) as f:
        corpus_airline = json.load(f)
    with open(args.corpus_retail) as f:
        corpus_retail = json.load(f)

    # Load embedding model
    print(f"Loading embedding model on {args.gpu}...")
    from sentence_transformers import SentenceTransformer
    emb_model = SentenceTransformer(args.emb_model, device=args.gpu)

    # Setup user sim
    from tau2.registry import registry
    from openai import OpenAI
    port_url = f"http://localhost:{args.port}/v1"
    client = OpenAI(base_url=port_url, api_key="EMPTY")
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    user_llm = f"openai/{model_name}"
    user_llm_args = {
        "temperature": 0.0, "max_context_length": 32000,
        "api_base": port_url, "api_key": "EMPTY",
        "tokenizer_model": f"openai/{model_name}",
    }

    # Build task -> skill mapping
    task_skill = {}  # "A8" -> "S2"
    for s in skills_data["skills"]:
        for t in s["failed_tasks"]:
            task_skill[t] = s["skill_id"]

    print(f"\n{'K':>3} | {'Tasks':>5} | {'Top-1':>7} | {'Oracle':>7} | {'Top-1%':>7} | {'Oracle%':>7}")
    print("-" * 55)

    for K in [1, 2, 4, 8, 16]:
        active_skills = set(skill_order[:K])

        total_top1 = 0
        total_oracle = 0
        total_tasks = 0

        for domain, prefix, full_corpus in [("airline", "A", corpus_airline), ("retail", "R", corpus_retail)]:
            # Filter corpus to active skills
            corpus = [t for t in full_corpus if t["skill"] in active_skills]
            if not corpus:
                continue

            # Filter tasks to those assigned to active skills
            domain_tasks = []
            for task_id, skill_id in task_skill.items():
                if task_id.startswith(prefix) and skill_id in active_skills:
                    tid = task_id[len(prefix):]
                    domain_tasks.append((tid, skill_id))

            if not domain_tasks:
                continue

            # Build skill descriptions from corpus
            skill_descs = {t["skill"]: t.get("description", "") for t in corpus}

            # Embed corpus
            corpus_texts = [t["text"] for t in corpus]
            corpus_embeddings = emb_model.encode(corpus_texts)

            # Get tau2 tasks
            task_ids = [tid for tid, _ in domain_tasks]
            tau2_tasks = get_tasks(task_set_name=domain, task_ids=task_ids)
            tau2_map = {t.id: t for t in tau2_tasks}

            env = registry.get_env_constructor(domain)()

            def make_user_sim(task, seed):
                try:
                    user_tools = env.get_user_tools()
                except Exception:
                    user_tools = None
                sim = UserSimulator(
                    tools=user_tools, instructions=str(task.user_scenario),
                    llm=user_llm, llm_args=deepcopy(user_llm_args),
                )
                sim.set_seed(seed)
                return sim

            task_pairs = [(tau2_map[tid], (tid, needed)) for tid, needed in domain_tasks if tid in tau2_map]

            if K == 1:
                # Trivial: only one skill, always correct
                total_top1 += len(task_pairs)
                total_oracle += len(task_pairs)
                total_tasks += len(task_pairs)
            else:
                results = run_rac(
                    corpus, skill_descs, task_pairs, make_user_sim,
                    emb_model, corpus_embeddings, client, model_name,
                    args.topk, args.threshold,
                )
                for tid, needed, predicted, cands in results:
                    total_tasks += 1
                    if predicted == needed:
                        total_top1 += 1
                    if needed in cands:
                        total_oracle += 1

        top1_pct = 100 * total_top1 / total_tasks if total_tasks else 0
        oracle_pct = 100 * total_oracle / total_tasks if total_tasks else 0
        print(f"{K:>3} | {total_tasks:>5} | {total_top1:>4}/{total_tasks:<3}| {total_oracle:>4}/{total_tasks:<3}| {top1_pct:>6.1f}% | {oracle_pct:>6.1f}%")


if __name__ == "__main__":
    main()
