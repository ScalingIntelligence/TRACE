"""Produce compact summaries of each trajectory for Phase 1 analysis."""
import json
import sys
from pathlib import Path


def compact_messages(messages, max_content_chars=300, max_tool_chars=200):
    out = []
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if len(content) > max_content_chars:
            content = content[:max_content_chars] + "...[truncated]"
        tool_calls = m.get("tool_calls") or []
        tc_summary = []
        for tc in tool_calls:
            name = tc.get("name") or (tc.get("function") or {}).get("name", "?")
            args = tc.get("arguments")
            if args is None:
                args = (tc.get("function") or {}).get("arguments", "")
            if isinstance(args, dict):
                args = json.dumps(args)
            if args and len(args) > max_tool_chars:
                args = args[:max_tool_chars] + "...[truncated]"
            tc_summary.append(f"{name}({args})")
        rec = {"role": role}
        if content:
            rec["content"] = content
        if tc_summary:
            rec["tool_calls"] = tc_summary
        # tool results: content contains the output
        if role == "tool" and content:
            rec["content"] = content[:max_content_chars]
        out.append(rec)
    return out


def summarize_file(path, domain):
    with open(path) as f:
        data = json.load(f)
    tasks = {t["id"]: t for t in data["tasks"]}
    sims = data["simulations"]
    out = []
    for s in sims:
        tid = s["task_id"]
        task = tasks.get(tid, {})
        reward_info = s.get("reward_info", {})
        reward = reward_info.get("reward", 0.0)
        breakdown = reward_info.get("reward_breakdown", {})
        # description of task
        task_desc = task.get("description", "")
        if isinstance(task_desc, dict):
            task_desc = task_desc.get("purpose", "") or json.dumps(task_desc)[:500]
        user_scenario = task.get("user_scenario", {})
        if isinstance(user_scenario, dict):
            instructions = user_scenario.get("instructions", "") or ""
        else:
            instructions = str(user_scenario)
        if isinstance(instructions, dict):
            instructions = json.dumps(instructions)[:600]

        # Reward details
        db_check = reward_info.get("db_check") or {}
        nl_assertions = reward_info.get("nl_assertions") or {}
        action_checks = reward_info.get("action_checks") or {}
        communicate_checks = reward_info.get("communicate_checks") or {}

        failure_reasons = []
        for k, v in [("db_check", db_check), ("action_checks", action_checks),
                     ("nl_assertions", nl_assertions),
                     ("communicate_checks", communicate_checks)]:
            if isinstance(v, dict):
                r = v.get("reward", 1.0)
                if r is not None and r < 1.0:
                    # try to extract informative detail
                    info = v.get("info") or v.get("details") or v
                    try:
                        info_s = json.dumps(info)[:500]
                    except Exception:
                        info_s = str(info)[:500]
                    failure_reasons.append({"component": k, "reward": r, "info": info_s})

        msgs = compact_messages(s["messages"])
        out.append({
            "task_id": f"{domain}_{tid}",
            "domain": domain,
            "reward": reward,
            "breakdown": breakdown,
            "task_purpose": str(task_desc)[:400],
            "user_instructions": str(instructions)[:800],
            "termination_reason": s.get("termination_reason"),
            "n_messages": len(msgs),
            "failure_reasons": failure_reasons,
            "messages": msgs,
        })
    return out


if __name__ == "__main__":
    airline = summarize_file(
        "evals/benchmarks/tau2_bench_eval/data/simulations/qwen-3-30b-airline-qwen3-30b.json",
        "airline")
    retail = summarize_file(
        "evals/benchmarks/tau2_bench_eval/data/simulations/qwen-3-30b-retail-qwen3-30b.json",
        "retail")
    all_trajs = airline + retail
    Path("pipeline/_trajectories_full.json").write_text(json.dumps(all_trajs, indent=1))
    # mini version with only metadata + failure reasons for initial scan
    mini = []
    for t in all_trajs:
        mini.append({
            "task_id": t["task_id"],
            "reward": t["reward"],
            "breakdown": t["breakdown"],
            "task_purpose": t["task_purpose"],
            "user_instructions": t["user_instructions"][:400],
            "n_messages": t["n_messages"],
            "termination_reason": t["termination_reason"],
            "failure_reasons": t["failure_reasons"],
        })
    Path("pipeline/_trajectories_mini.json").write_text(json.dumps(mini, indent=1))
    print(f"wrote {len(all_trajs)} trajectories")
    print(f"  airline: {len(airline)}  retail: {len(retail)}")
    pass_count = sum(1 for t in all_trajs if t["reward"] == 1.0)
    print(f"  passes: {pass_count}  fails: {len(all_trajs)-pass_count}")
