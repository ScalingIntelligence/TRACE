# Analysis of Synthesized Environments

## Overview

Our training pipeline requires dedicated environments for each identified skill, where an RL agent can collect rollouts and receive reward signals aligned with tau2-bench's evaluation criteria. Unlike prior work that trains directly on benchmark tasks (risking overfitting to the test distribution) or generates environments from scratch (requiring extensive verification), we adopt a middle path: we construct skill-specific environments that share the exact tools, database schemas, and policy documents of tau2-bench, but generate novel task scenarios that isolate individual capability gaps. This ensures reward compatibility while maintaining separation between training and evaluation distributions.

## Environment Design Principles

Each environment is designed around three constraints:

1. **Tool and schema fidelity.** Training environments expose the same 14 airline tools and 15 retail tools as tau2-bench, backed by identical SQLite database schemas. This eliminates distribution shift in the tool-calling interface between training and evaluation.

2. **Skill isolation.** Each environment generates tasks that primarily stress a single identified skill. For instance, the multi-step environment generates compound requests requiring 2--5 sequential operations, while the precondition environment presents requests where the correct action is to *refuse* (e.g., modifying a non-modifiable fare class). This targeted design allows GRPO to assign credit to the specific capability being trained.

3. **Procedural generation.** Task scenarios are generated programmatically with randomized parameters (user profiles, reservation details, product catalogs), ensuring the agent cannot memorize solutions. Each training episode samples a fresh database state.

## Per-Environment Details

### Structured Data Reasoning

This environment targets the agent's ability to parse, compare, and cross-reference structured records returned by tools. Tasks require selecting the correct item from a catalog or the correct flight from search results when multiple candidates match partial constraints.

**Task examples:** "Find the cheapest nonstop flight from JFK to LAX departing after 2pm," where the agent must compare price, route, and departure time fields across multiple flight records. In the retail domain: "Exchange the running shoes for size 10 in red leather with rubber sole," requiring precise attribute matching against a product catalog with dozens of variants.

**Reward signal:** Binary, based on whether the agent's final tool call targets the correct item/flight ID. Partial credit is not awarded — selecting a variant that matches some but not all constraints receives reward 0.

**Scale:** ~2,200 lines of environment code. Generates unlimited unique scenarios by randomizing product/flight attributes, constraint combinations, and the number of distractor options.

### Tool Calling Precision

This environment isolates the agent's ability to construct tool calls with correct arguments. The agent is placed in mid-conversation states (after authentication and record retrieval) and must produce a single correct tool call.

**Task examples:** Given a retrieved reservation and user request, produce the correct `update_reservation_flights` call with the right flight numbers, cabin class, and payment method. Errors include using the wrong payment ID from the user profile, swapping outbound and return flights, or passing an incorrect baggage count.

**Reward signal:** Exact match on tool name and all arguments against the expected action. The environment verifies argument values against the database state.

**Scale:** ~730 lines. Derived from tau2-bench task structures with randomized argument perturbations.

### Multi-Step Task Completion

This environment generates compound requests that require multiple sequential tool calls, testing whether the agent can maintain a plan across turns and complete all sub-tasks.

**Task examples:** "Cancel reservation A, then rebook a new flight using the refund certificate, and add 2 checked bags." The agent must execute cancel, search, book, and update operations in the correct order, where each step's output feeds into the next.

**Reward signal:** All-or-nothing — reward 1.0 only if every required database mutation is present in the final state. Completing 3 of 4 sub-tasks yields reward 0.

**Scale:** ~1,500 lines. Generates compound scenarios with 2--5 operations, drawn from the full set of available tools.

### Precondition Verification

This environment tests whether the agent checks eligibility conditions before executing state-changing actions. Approximately 60% of generated tasks present requests that should be *refused* under policy, while 40% present valid requests.

**Task examples:** A user requests cancellation of a basic economy flight booked 3 days ago with no travel insurance — all four cancellation conditions fail, and the correct action is to refuse and offer transfer to a human agent. Conversely, a business-class passenger with a cancelled flight should receive a compensation certificate.

**Reward signal:** For refusal tasks, reward 1.0 if the agent correctly refuses and does not call the mutation tool. For valid tasks, standard database state verification. This balanced design prevents the agent from learning to always refuse.

**Scale:** 20 hand-curated scenarios per domain (12 refuse, 8 allow), with randomized user profiles and reservation parameters.

### Adversarial Policy Compliance

This environment trains robustness against social engineering. A simulated adversarial user applies emotional manipulation (bereavement stories, anger, flattery), false factual claims (lying about insurance status or flight delays), and persistent pressure to override policy.

**Task examples:** A user claims their flight was delayed and demands a $300 compensation certificate, but `get_flight_status` returns "on time." The agent must verify the claim against tool output and refuse, even when the user escalates emotionally.

**Reward signal:** Policy compliance — reward 1.0 if the agent's actions match policy-correct behavior. Caving to pressure and performing an unauthorized action yields reward 0, even if the user expresses satisfaction.

**Scale:** Configurable adversarial ratio (default 20% adversarial, 80% cooperative interactions). Uses an LLM-powered user simulator with adversarial persona instructions.

## Environment Statistics

| Environment | Lines of Code | Tools Used | Avg. Turns | Reward Type |
|-------------|--------------|------------|------------|-------------|
| Structured data reasoning | 2,234 | 14 (airline) / 15 (retail) | 8--12 | Binary (correct selection) |
| Tool calling precision | 727 | 14 / 15 | 1--3 | Exact argument match |
| Multi-step task completion | 1,526 | 14 / 15 | 10--20 | All-or-nothing DB state |
| Precondition verification | ~800 | 14 / 15 | 5--10 | Policy compliance |
| Adversarial policy compliance | ~1,200 | 14 / 15 | 10--30 | Policy compliance |

All five environments share the tau2-bench tool interface and database layer, totaling 29 unique tools across both domains.

## Contrastive Skill Selection Validation

To validate that our five environments target the correct capability gaps, we conduct a contrastive skill selection experiment. An LLM judge independently analyzes baseline failure trajectories (110 failed tasks across both domains) and selects the top-5 most impactful skills from a menu of 14 candidates — including our 5 training skills, 4 plausible alternatives (numerical reasoning, information communication, early termination, conditional reasoning), and 5 distractors (language fluency, tone/empathy, format compliance, tool hallucination, proactive upselling).

This process is repeated 10 times with independent judge instances. Figure X(a) shows the selection frequency: three of our five training skills are selected in all 10 runs, tool calling precision in 8/10, and adversarial policy compliance in 4/10. No distractor skill is ever selected. Figure X(b) shows the median task coverage per skill with interquartile range error bars, confirming that our training skills address the highest-coverage failure modes.

This analysis serves two purposes: (1) it provides empirical justification for our environment design choices, showing they target real capability gaps rather than artifacts, and (2) it demonstrates that the skill identification process is robust — independent judges converge on the same skills without coordination.

## Benchmark Task Characteristics

For reference, we summarize the tau2-bench evaluation tasks that our environments are designed to improve upon:

| Statistic | Airline | Retail |
|-----------|---------|--------|
| Total tasks | 50 | 114 |
| Multi-action tasks | 27 (54%) | 92 (81%) |
| Unique tools exercised | 11 | 14 |
| Avg. conversation length (pass) | 22.5 ± 8.8 msgs | 23.3 ± 6.1 msgs |
| Avg. conversation length (fail) | 26.4 ± 11.2 msgs | 27.9 ± 10.9 msgs |
| Avg. tool calls (pass) | 6.9 ± 3.9 | 6.7 ± 2.5 |
| Avg. tool calls (fail) | 8.0 ± 4.3 | 8.5 ± 4.6 |
| Base model pass rate | 24.0% | 36.8% |

Failed tasks are on average longer than passed tasks (26--28 vs. 22--23 messages), suggesting that failures often involve the agent entering recovery loops or redundant tool calls rather than terminating early. The retail domain has a substantially higher fraction of multi-action tasks (81% vs. 54%), which is reflected in the dominance of the multi-step completion skill in our contrastive analysis.
