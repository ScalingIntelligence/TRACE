"""Recursive skill evolution: analyze failures and generate new skills via LLM."""
import json
import re
from typing import Dict, List, Optional, Tuple

from .prompts import SKILL_EVOLUTION_PROMPT
from .skillbank import SkillBank


def _extract_json_array(text: str) -> list:
    """Extract a JSON array from LLM output that may contain markdown fences."""
    # Try to find JSON in code blocks first
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Try raw JSON array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return []


def _format_trajectory_for_prompt(sim: dict, max_turns: int = 15) -> str:
    """Format a single failed simulation for the evolution prompt."""
    task_id = sim.get("task_id", "?")
    reward = sim.get("reward_info", {}).get("reward", 0.0)
    reason = sim.get("termination_reason", "unknown")
    messages = sim.get("messages", [])

    lines = [f"### Task {task_id} (reward={reward}, termination={reason})"]
    for msg in messages[:max_turns]:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")

        if tool_calls:
            tc_str = json.dumps(tool_calls, indent=None)[:500]
            lines.append(f"  [{role}] <tool_call> {tc_str}")
        elif content:
            lines.append(f"  [{role}] {content[:300]}")

    if len(messages) > max_turns:
        lines.append(f"  ... ({len(messages) - max_turns} more turns)")

    return "\n".join(lines)


def analyze_failures(
    simulations: List[dict],
    threshold: float = 0.4,
    max_failures: int = 10,
) -> List[dict]:
    """Select worst-performing failures for skill evolution.

    Args:
        simulations: List of simulation dicts with reward_info and messages.
        threshold: Only evolve if overall success rate < threshold.
        max_failures: Max number of failures to analyze.

    Returns:
        List of failed simulation dicts, or empty if threshold is met.
    """
    successes = sum(1 for s in simulations
                    if s.get("reward_info", {}).get("reward", 0) > 0)
    total = len(simulations)
    success_rate = successes / max(1, total)

    if success_rate >= threshold:
        return []

    failures = [s for s in simulations
                if s.get("reward_info", {}).get("reward", 0) == 0]
    return failures[:max_failures]


def evolve_skills(
    skillbank: SkillBank,
    failed_sims: List[dict],
    generate_fn,
    max_new: int = 3,
    domain: str = "airline",
) -> Tuple[int, List[dict]]:
    """Generate new skills by analyzing failures.

    Args:
        skillbank: Current SkillBank to augment.
        failed_sims: Failed simulation dicts.
        generate_fn: Callable(prompt: str) -> str. LLM generation function.
        max_new: Max new skills to generate.
        domain: Domain name for ID prefixes.

    Returns:
        (count_added, new_skills_list)
    """
    if not failed_sims:
        return 0, []

    # Format current skills
    current_skills = skillbank.get_skills_for_prompt(top_k=100)

    # Format failed trajectories
    traj_texts = [_format_trajectory_for_prompt(s) for s in failed_sims]
    failed_text = "\n\n".join(traj_texts)

    # Determine next skill ID prefix
    prefix = "A" if domain == "airline" else "R"
    next_prefix = f"{prefix}E"  # E for evolved

    prompt = SKILL_EVOLUTION_PROMPT.format(
        domain=domain,
        current_skills=current_skills,
        failed_trajectories=failed_text,
        max_new=max_new,
        next_id_prefix=next_prefix,
    )

    # Call LLM
    response = generate_fn(prompt)

    # Parse new skills
    try:
        new_skills = _extract_json_array(response)
    except (json.JSONDecodeError, Exception):
        return 0, []

    # Validate and add
    valid_skills = []
    for skill in new_skills[:max_new]:
        if all(k in skill for k in ("skill_id", "title", "principle", "when_to_apply")):
            valid_skills.append(skill)

    count = skillbank.add_skills(valid_skills)
    if count > 0:
        skillbank.save()

    return count, valid_skills


def create_vllm_generate_fn(base_url: str, model_name: str, temperature: float = 0.7):
    """Create a generate function that calls a vLLM server.

    Returns:
        Callable(prompt: str) -> str
    """
    import requests

    def generate(prompt: str) -> str:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 2048,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    return generate
