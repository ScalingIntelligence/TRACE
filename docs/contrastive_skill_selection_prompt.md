# Contrastive Skill Selection — Judge Agent Prompt

## Overview

This document contains the prompt to give to a judge agent (e.g. Claude Code) for
analyzing simulation trajectories and identifying skills the model consistently
fails at. The judge reads pass/fail trajectories, categorizes tasks into
domain-agnostic skills, and outputs a structured JSON.

Run the prompt N times (e.g. 5) with different attempt numbers, then feed all
outputs into `analyze_contrastive_skills.py` to measure consistency.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `{THRESHOLD}` | 8 | Min % of total tasks a skill must fail on to be included |
| `{TOP_K}` | 5 | Max number of skills to return per attempt |
| `{ATTEMPT_NUMBER}` | 1-5 | Which attempt this is (for robustness tracking) |
| `{RETAIL_PATH}` | — | Path to retail simulation JSON |
| `{AIRLINE_PATH}` | — | Path to airline simulation JSON |

## Prompt

```
You are a contrastive skill analyst. You are given simulation results from
a customer service agent evaluated on two domains (retail and airline).
Each simulation contains a task with required actions (evaluation_criteria.actions)
and a reward (reward_info.reward: 1.0 = pass, <1.0 = fail).

The simulation files are:
- Retail: {RETAIL_PATH}
- Airline: {AIRLINE_PATH}

Each file has this structure:
- tasks: list of task objects with id, evaluation_criteria.actions (list of {name, arguments})
- simulations: list of simulation objects with task_id, reward_info.reward

## Your job

1. **Read the simulation files** and for each task, extract:
   - The set of required action names from evaluation_criteria.actions
   - Whether the agent passed or failed (reward_info.reward)
   - The task_id

2. **Categorize each task into a domain-agnostic skill** based on the
   action patterns. Think about skills in terms of COMPLEXITY and STRUCTURE,
   not domain-specific names. Use categories like:
   - How many mutating (state-changing) operations are required?
   - Does the task require searching/looking up information before acting?
   - Does it require calculation?
   - Is it read-only, single-mutation, or multi-mutation?

   Good skill names describe the PATTERN, not the domain:
   ✓ "two_step_modification", "update_with_search", "simple_cancellation"
   ✗ "flight_rebooking", "exchange_delivered_order" (too domain-specific)

   Distinguish between read-only actions (get_*, find_*, search_*, calculate)
   and mutating actions (everything else, except transfer_to_human_agents
   which is its own category "escalation_to_human").

3. **For each skill, compute:**
   - fail_count and pass_count
   - success_rate = pass_count / (pass_count + fail_count)
   - pct_of_total = fail_count / total_tasks * 100
   - which domains it appears in (retail, airline, or both)
   - list of failed task IDs (prefixed R for retail, A for airline)
   - list of passed task IDs

4. **Filter:** Only keep skills where pct_of_total >= {THRESHOLD}%
   (the failure impact is large enough to matter for training).

5. **Sort** by fail_count descending (biggest weakness first).

6. **Return at most {TOP_K} skills.**

## Output format

Return a JSON object with this exact structure:
{
  "attempt": {ATTEMPT_NUMBER},
  "result": {
    "skill_name": {
      "assigned_model": "base",
      "success_rate": <float 0-1>,
      "failed_cases": ["R1", "A5", ...],
      "passed_cases": ["R0", "A2", ...],
      "fail_count": <int>,
      "pass_count": <int>,
      "pct_of_total": <float>,
      "total_cases": <int>,
      "domains": ["retail", "airline"]
    },
    ...
  }
}

## Important guidelines

- Skills should be GENERAL enough to apply across domains. If a skill only
  appears in one domain, that's fine, but the NAME should still be generic.
- Don't be too strict about success vs failure. A skill with 50% fail rate
  but high volume (many tasks) is still a strong training candidate because
  the absolute number of failures is large.
- The contrastive signal is: does this skill have MORE failures than successes,
  or at least a large absolute number of failures? Skills the model sometimes
  passes and sometimes fails are STILL valuable — they indicate inconsistency
  that training can fix.
- Aim for 3-6 meaningful skill categories, not 15+ overly-specific ones and
  not 2 overly-broad ones.
```

## After running the prompt

1. Collect outputs from all N attempts into a single JSON array:
   ```json
   [
     {"attempt": 1, "result": {...}},
     {"attempt": 2, "result": {...}},
     ...
   ]
   ```

2. Run the analysis:
   ```bash
   python analyze_contrastive_skills.py results.json \
     --threshold 0 --plot --output analysis.png --export selected_skills.json
   ```

3. Check the consistency matrix to see which skills appeared in most attempts.
   Skills appearing in 4+/5 attempts are strong candidates for training.

## Lessons learned

| Problem we hit | How the prompt prevents it |
|----------------|--------------------------|
| Domain-specific skill names (retail ≠ airline) | Explicitly requires domain-agnostic names with good/bad examples |
| Too strict filtering (high fail rate AND high impact) | Tells judge that high-volume ~50% skills are also valid candidates |
| Action combos too granular (each unique action set = one skill) | Instructs categorization by complexity pattern, not exact action sets |
| Separate retail/airline outputs needed manual merging | Single unified output from the start |
| Needed 3 iterations to get right abstraction level | Examples of good vs bad granularity (3-6 categories, not 15+ or 2) |

## Related files

- `evals/benchmarks/tau2_bench_eval/contrastive_skill_selection.py` — automated version of this analysis
- `evals/benchmarks/tau2_bench_eval/analyze_contrastive_skills.py` — consistency analysis, tables, and plots
- `evals/benchmarks/tau2_bench_eval/selected_skills.json` — latest selected skills output
