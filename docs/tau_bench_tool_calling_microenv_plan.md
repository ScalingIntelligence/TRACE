# Tau Bench Tool-Calling Microenvironment: Revised Design Plan

## 1. Executive Summary

We analyze the Qwen3-30B-A3B evaluation results on tau2-bench (airline + retail) and design a programmatically verifiable microenvironment that trains the model's ability to execute correct tool calls in the tau bench format. The microenvironment uses the **exact same tools, database, policy, format, user LLM simulator (GPT-4.1), and multi-turn episode structure** as tau bench. It differs only in that the **task scenarios are simpler** — requiring fewer write actions, providing clearer user intent, and reducing reasoning overhead — so the model can learn the mechanics of correct tool execution at a higher base success rate before facing full tau bench complexity.

**Critical revision from v1**: We replace the scripted deterministic user with the **exact same LLM-based user simulator** from tau2-bench (GPT-4.1, same system prompt template, same role-flipping mechanism). We also provide a more honest analysis of what "tool-calling competence" actually entails and rigorously verify the easiness claim using per-task data.

---

## 2. Comprehensive Evaluation Results

### 2.1 Overall Performance

| Domain | Total | Success | Fail | Success Rate |
|--------|-------|---------|------|-------------|
| Airline | 50 | 12 | 38 | 24.0% |
| Retail | 114 | 42 | 72 | 36.8% |
| **Combined** | **164** | **54** | **110** | **32.9%** |

### 2.2 Reward Decomposition of Failures

The reward in tau bench is: `reward = DB_check × COMMUNICATE_check`

| Category | Count | % of Failures | Meaning |
|----------|-------|---------------|---------|
| DB_FAIL + COMM_PASS | 80 | 72.7% | Model communicated correctly but DB state was wrong |
| BOTH_FAIL | 15 | 13.6% | Both DB actions and communication wrong |
| TIMEOUT (max_steps) | 8 | 7.3% | Ran out of steps (thrashing/looping) |
| DB_PASS + COMM_FAIL | 6 | 5.5% | Actions correct but failed to communicate info |
| TOO_MANY_ERRORS | 1 | 0.9% | 10+ consecutive tool errors |

### 2.3 Success Rate by Task Complexity (Critical Finding)

**This is the most important table in this plan.**

| Write Actions Required | Airline Tasks | Airline Success | Airline Rate | Retail Tasks | Retail Success | Retail Rate | Combined Rate |
|------------------------|--------------|-----------------|-------------|-------------|----------------|-------------|---------------|
| 0 (info/refusal) | 20 | 11 | **55.0%** | 12 | 3 | 25.0% | **43.8%** |
| Exactly 1 | 16 | 1 | **6.2%** | 59 | 26 | **44.1%** | **36.0%** |
| 2+ | 14 | 0 | **0.0%** | 43 | 13 | 30.2% | **22.8%** |

**Key observations:**
1. **Airline write actions are catastrophically hard**: 6.2% on single-write, 0% on multi-write. The model almost never correctly executes an airline write tool.
2. **Retail single-write is moderately easier**: 44.1% vs 36.8% overall — a meaningful but modest improvement.
3. **Refusal/info tasks are where the model excels**: 55% on airline (policy compliance), 25% on retail.
4. **Combined single-write (36.0%) is barely above overall (32.9%)** — the raw numbers do NOT show single-write is dramatically easier. The easiness argument requires deeper analysis (see Section 4).

### 2.4 Honest Failure Decomposition: Mechanical vs Reasoning vs System

The original plan classified all DB failures as "tool-calling errors." This overclaims. A more accurate decomposition:

#### Category A: Mechanical Tool-Calling Errors (~22% of failures, ~24 cases)
Failures where the model understood what to do but the API call was mechanically wrong:
- **Case sensitivity** (~8 cases): `find_user_id_by_name_zip({"first_name": "mei"})` instead of `"Mei"`
- **Argument format errors** (~4 cases): Missing `#W` prefix on order IDs, wrong field names
- **Batch splitting** (~3 cases): Making 2 sequential calls instead of 1 batched call (second fails)
- **Missing paired calls** (~5 cases): `modify_pending_order_items` without `modify_pending_order_address`
- **Payment format** (~4 cases): Wrong payment method ID format or wrong amount calculation

#### Category B: Reasoning Errors Manifesting as Wrong Tool Calls (~55% of failures, ~60 cases)
Failures where the model's reasoning about WHAT action to take was wrong:
- **Wrong variant/flight selection** (~31 cases): Retrieved product/flight data but selected wrong item (not cheapest, wrong constraint match, wrong water resistance level, etc.)
- **Policy violations** (~10 cases): Cancelled when policy forbids it, refunded wrong amount, accepted false insurance claims
- **Wrong tool selection** (~8 cases): Used `exchange` instead of `return`, `update` instead of `cancel+book`
- **Incomplete multi-step** (~7 cases): Completed first action but missed subsequent required actions
- **Wrong entity targeting** (~4 cases): Operated on wrong order/reservation

#### Category C: System/Interaction Failures (~16% of failures, ~18 cases)
- **Max steps timeout** (8): Looping, thrashing, or wasting turns
- **Communication-only** (6): DB correct but missed info relay
- **Authentication failure** (3): Couldn't look up user
- **Too many errors** (1): Stuck in error loop

#### Category D: Comprehension Failures (~7% of failures, ~8 cases)
The model misunderstood the user's natural language intent:
- "Exchange for one with the same water resistance as your OTHER earbuds" → model kept same water resistance
- "Change to cheapest desk lamp" → model picked 2nd cheapest
- "Cancel reason" → wrong reason string selected

### 2.5 Instance-by-Instance Failure Trace

*(Preserved from v1 — see Section 3 of original plan for full per-task breakdown)*

### 2.6 Success Patterns

54 total successes. Key patterns:
- **Policy compliance/refusal** (11 airline, 3 retail): Model correctly refuses disallowed actions
- **Simple single-action retail** (26 retail single-write): Clean info-gather → correct write call
- **Multi-action retail** (13 retail 2+ write): Model occasionally chains operations correctly
- **Clean conversation flow**: Successes average 6.8 tool calls (range 2-16), with a clean authenticate → gather → act → communicate pattern

---

## 3. What "Tool-Calling Competence" Actually Means

### 3.1 It Is NOT a Separable Mechanical Skill

The original plan framed tool-calling as a pure mechanical competence gap: "the model knows WHAT to do but can't HOW." This is partially correct but oversimplified.

**Tool-calling competence in tau bench = the entire pipeline from context to correct API call:**

```
User intent → Information gathering → Constraint reasoning →
Entity selection → Argument construction → API call execution
```

Each step can fail independently:
- **Information gathering**: Missing a `get_product_details` call, not searching enough flights
- **Constraint reasoning**: Not applying "cheapest available" correctly, ignoring policy rules
- **Entity selection**: Picking wrong variant, wrong flight, wrong order
- **Argument construction**: Wrong types, missing fields, case errors
- **API call execution**: Wrong tool name, batch vs sequential

Only the last two are "mechanical." The first three are reasoning-intensive. The data shows ~22% of failures are mechanical and ~55% are reasoning-based. **The skill we're training is the full pipeline, not just the last step.**

### 3.2 Why This Full Pipeline Is Still the Right Training Target

Even though most failures involve reasoning, training on the full pipeline is valuable because:

1. **Reasoning and mechanics are coupled**: The model learns what arguments to look for (e.g., "I need to call `get_product_details` to find variant IDs for the exchange call") by seeing the full context-to-call pipeline succeed.

2. **The mechanical errors compound**: A case-sensitivity error on `find_user_id_by_name_zip` derails the entire episode, wasting all subsequent correct reasoning. Fixing ~22% of failures (mechanical) unlocks the model's existing reasoning for those episodes.

3. **The reasoning errors are learnable through exposure**: The model selected the 2nd-cheapest variant because it couldn't reliably scan a table of 12 variants. More practice with variant selection on simpler tables (3-5 options) builds the skill.

4. **Transfer from simple to complex**: Single-action tasks with correct execution teach the model what a correct `exchange_delivered_order_items` call looks like. This pattern transfers to multi-action tasks.

---

## 4. Is the Microenvironment Actually Easier? (Rigorous Analysis)

### 4.1 Why Raw Success Rates Are Misleading

The raw single-write success rate (36.0% combined) is barely above the overall average (32.9%). This does NOT mean our microenvironment is only marginally easier. The raw numbers are misleading because:

1. **The single-write category includes extremely hard airline tasks** (6.2% success) alongside moderately easy retail tasks (44.1%).
2. **The tau bench single-write tasks were NOT designed to be simple** — many involve nuanced variant selection, ambiguous user phrasing, and complex policy constraints.
3. **Our microenvironment tasks will be intentionally simplified** along specific axes that the tau bench tasks are not.

### 4.2 The Simplification Axes

Our microenvironment keeps everything the same as tau bench EXCEPT the `UserInstructions` given to the user LLM simulator. Simpler instructions → simpler conversation → fewer failure modes → higher success rate.

| Simplification Axis | Tau Bench | Microenvironment | Expected Impact |
|---------------------|-----------|-------------------|-----------------|
| **Actions per task** | 1-5 write actions | **Always 1** | Eliminates multi-step orchestration failures (15.8% of failures) |
| **User intent clarity** | Ambiguous, requires inference ("same water resistance as OTHER earbuds") | **Explicit** ("exchange for the blue 6-hour IPX4 variant") | Eliminates comprehension failures (~7%) |
| **Variant/flight selection** | 12+ options, constraints like "cheapest available" | **3-5 options, explicit target or obvious selection** | Reduces wrong-selection failures |
| **Adversarial pressure** | User lies, manipulates, pressures | **Cooperative user** (+ 20% policy-gated for non-regression) | Eliminates spurious action failures (13.7%) in cooperative cases |
| **Policy complexity** | Complex conditional rules, edge cases | **Standard cases only** (cancellation of clearly-eligible items, returns within window) | Reduces policy reasoning load |

### 4.3 Expected Success Rate Estimation

We estimate the microenvironment success rate by analyzing how each simplification removes failure modes:

**Retail single-write (current: 44.1%, 26/59 success)**

The 33 retail single-write failures break down as:
- Wrong variant selection: ~15 (simplification removes ~10 of these)
- Mechanical errors (case, format): ~8 (training should fix ~4)
- Wrong tool: ~5 (clearer intent fixes ~3)
- Missing paired calls: ~3 (single-action design eliminates these)
- Comprehension: ~2 (explicit instructions fix these)

Estimated remaining failures after simplification: ~11 out of 59 → **~81% expected success rate**

But this is for tau-bench's specific 59 tasks. For our generated tasks, which are designed to be at this simplified difficulty:

**Estimated retail microenvironment success: ~60-70%**

(Lower than the 81% because our generated tasks will include some harder variants, and the model will still make errors on unfamiliar entity combinations.)

**Airline single-write (current: 6.2%, 1/16 success)**

The 15 airline single-write failures break down as:
- Wrong flight selection: ~7 (explicit flight removes ~5)
- Wrong payment calculation: ~4 (simpler payment fixes ~2)
- Policy violation: ~2 (cooperative user fixes these)
- Wrong reservation: ~2 (single reservation focus fixes ~1)

Estimated remaining failures after simplification: ~5 out of 16 → **~69% expected success rate**

But again, for generated tasks at this difficulty:

**Estimated airline microenvironment success: ~35-50%**

(Lower because airline tools are fundamentally harder — `update_reservation_flights` requires correctly structured flight arrays, cabin assignments, and payment breakdowns even in the simplest case.)

**Combined weighted estimate (60% retail, 40% airline):**

**Expected microenvironment success rate: ~50-60%** vs tau bench's 32.9%

### 4.4 Why This Is "Easier to Learn" (Not Just "Higher Success Rate")

For GRPO/PPO training, "easier to learn" means more than higher base success rate. The microenvironment is easier across multiple dimensions:

**1. Better GRPO signal (higher reward variance within groups)**

GRPO requires contrast between successful and failed rollouts within the same prompt group. With K=6 rollouts per scenario:
- At 33% success: E[successes per group] = 2.0. P(all fail) = 9.0%. Many groups have too few successes.
- At 55% success: E[successes per group] = 3.3. P(all fail) = 0.8%. Almost every group has useful contrast.

**2. Simpler credit assignment**

In a multi-write tau bench task, if the episode fails, which write call was wrong? With single-write tasks, there is exactly one write action — either it's correct or it's not. The reward signal directly attributes to that single decision.

**3. Shorter episodes**

Single-action tasks typically require 8-12 turns (authenticate → gather info → act → confirm). Multi-action tasks require 15-30+. Shorter episodes mean:
- Fewer tokens per training sample → more samples per training batch
- Less context for the model to manage → less noise in what's learned
- Faster iteration → more training episodes per wall-clock hour

**4. Consistent difficulty distribution**

Tau bench tasks range from trivial (refusal: "sorry, can't do that") to nearly impossible (5-write multi-entity airline rebooking). This high variance wastes training signal — the model either gets easy tasks it already solves or impossible tasks with no learning signal. The microenvironment targets the ~50-60% zone consistently.

### 4.5 Formal "Easier to Learn" Argument

**Definition**: Task distribution D₁ is *easier to learn* than D₂ for policy π if:
1. E[R | π, D₁] > E[R | π, D₂] (higher expected reward = more positive examples)
2. Var[R | π, D₁] is in the useful range for the RL algorithm (not too low, not too high)
3. The skills learned on D₁ transfer to improve E[R | π', D₂] where π' is the updated policy

We verify each:
1. **E[R | π_base, D_micro] ≈ 50-60% > 32.9% = E[R | π_base, D_tau]** ✓ (Section 4.3)
2. **Var[R | π_base, D_micro]**: At 55% binary reward, Var = 0.55 × 0.45 = 0.2475. This is near-optimal for binary reward GRPO. ✓
3. **Transfer**: Argued in Section 6. The model learns tool-calling patterns (correct argument structure, variant selection, API sequencing) that apply identically to harder tasks. ✓

---

## 5. Microenvironment Design

### 5.1 Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                 Microenvironment                     │
│                                                      │
│  ┌──────────────┐    ┌──────────────┐               │
│  │ GPT-4.1 User │◄──►│ Agent (Qwen) │               │
│  │  Simulator   │    │  (training)  │               │
│  │  (tau2-bench │    │              │               │
│  │   exact)     │    └──────┬───────┘               │
│  └──────────────┘           │                        │
│                        Tool calls                    │
│                             │                        │
│                    ┌────────▼────────┐               │
│                    │  tau2-bench     │               │
│                    │  Environment    │               │
│                    │  (same tools,   │               │
│                    │   same DB,      │               │
│                    │   same policy)  │               │
│                    └────────┬────────┘               │
│                             │                        │
│                    ┌────────▼────────┐               │
│                    │  DB Hash        │               │
│                    │  Verification   │               │
│                    │  (same as       │               │
│                    │   tau bench)    │               │
│                    └─────────────────┘               │
└─────────────────────────────────────────────────────┘
```

**What's identical to tau bench:**
- Tools: Same `FlightTools`, `RetailTools`, `GenericToolKit` classes from tau2-bench
- Database: Same `db.json` files (airline + retail)
- Policy: Same `policy.md` files
- Agent system prompt: Same format (`<instructions>...</instructions><policy>...</policy>`)
- User simulator: **Same GPT-4.1 with same system prompt template** (see Section 5.2)
- Message format: Same `AssistantMessage`, `UserMessage`, `ToolMessage` types
- Verification: Same `get_db_hash()` comparison
- Reward: `DB_check × COMMUNICATE_check` (same as tau bench)

**What's different:**
- `UserInstructions` (the scenario given to the user LLM): Simpler, always single-write, clearer intent
- Task set: Infinite seed-generated vs 164 fixed
- No `nl_assertions` (LLM judge) component of reward — only programmatic checks

### 5.2 User LLM Simulator (Exact tau2-bench Reuse)

We use tau2-bench's `UserSimulator` class directly:

```python
from tau2.user.user_simulator import UserSimulator
from tau2.data_model.tasks import UserInstructions

# Create user simulator with same config as tau bench evaluation
user_sim = UserSimulator(
    tools=None,  # No user tools for standard scenarios
    instructions=generated_user_instructions,  # Our simplified scenario
    llm="gpt-4.1",  # Same model
    llm_args={"temperature": 0.0, "seed": task_seed},  # Same params
)
```

The user simulator:
1. Receives a system prompt: `{simulation_guidelines}\n<scenario>\n{instructions}\n</scenario>`
2. Uses the same `simulation_guidelines.md` from tau2-bench
3. Flips message roles (agent messages → "user" role, user messages → "assistant" role) per `UserState.flip_roles()`
4. Generates responses via LiteLLM/GPT-4.1
5. Ends conversation with `###STOP###` when task is complete

**Our only change**: the `UserInstructions` we generate. The format is identical to tau bench task instructions:

```python
UserInstructions(
    domain="retail",
    reason_for_call="You want to cancel your pending order #W2417020 because you no longer need the laptop.",
    known_info="You are Emma Smith. Your zip code is 10192.",
    unknown_info="You do not remember your email address.",
    task_instructions="You are polite and direct."
)
```

### 5.3 Scenario Generation

Each scenario is generated from a seed and produces:
1. `UserInstructions` — fed to the GPT-4.1 user simulator
2. `EvaluationCriteria` — expected actions, communicate_info, reward_basis
3. `InitialState` — any DB initialization needed (usually none for simple tasks)

```python
def generate_scenario(seed: int, domain: str) -> GeneratedTask:
    """Generate a simplified tau bench task from a seed.

    Returns a Task object in the exact same format as tau bench tasks,
    with simplified UserInstructions.
    """
    rng = random.Random(seed)
    db = load_db(domain)

    # Select scenario type
    scenario_type = rng.choices(
        ["single_action_explicit",    # User states exactly what they want
         "single_action_selection",    # User needs model to find right variant/flight
         "policy_gated"],             # Action may or may not be allowed
        weights=[50, 30, 20]
    )[0]

    # Pick a random user
    user_id = rng.choice(list(db.users.keys()))
    user = db.users[user_id]

    if scenario_type == "single_action_explicit":
        return generate_explicit_action(rng, user, db, domain)
    elif scenario_type == "single_action_selection":
        return generate_selection_action(rng, user, db, domain)
    else:
        return generate_policy_gated(rng, user, db, domain)
```

#### Scenario Type 1: Single-Action Explicit (50% of training)

**Target failure modes**: Mechanical errors (Category A), wrong tool selection

The user explicitly states what action they want. The model must authenticate, verify details, and execute the correct tool call.

**Retail examples:**
```python
# Cancel pending order
UserInstructions(
    domain="retail",
    reason_for_call="You want to cancel your order #W2417020 because you no longer need it.",
    known_info="Your name is Emma Smith. Your zip code is 10192.",
    task_instructions="You are polite and cooperative."
)
# Expected: cancel_pending_order(order_id="#W2417020", reason="no longer needed")

# Return delivered items
UserInstructions(
    domain="retail",
    reason_for_call="You want to return the Bluetooth Speaker from order #W6247578. You want the refund on your original payment method.",
    known_info="Your name is Mei Kovacs. Your zip code is 78250.",
    task_instructions="You are concise and business-like."
)
# Expected: return_delivered_order_items(order_id="#W6247578", item_ids=["..."], payment_method_id="...")
```

**Airline examples:**
```python
# Cancel reservation
UserInstructions(
    domain="airline",
    reason_for_call="You want to cancel reservation 3FRNFB. You have travel insurance and the reason is a medical emergency.",
    known_info="Your name is Emma Kim. Your user id is emma_kim_9957.",
    task_instructions="You are straightforward."
)
# Expected: cancel_reservation(reservation_id="3FRNFB")

# Update baggage
UserInstructions(
    domain="airline",
    reason_for_call="You want to add 1 checked bag to reservation ABC123.",
    known_info="Your name is John Lee. Your user id is john_lee_1234.",
    task_instructions="You are friendly."
)
# Expected: update_reservation_baggages(reservation_id="ABC123", ...)
```

**Why this is easier**: No ambiguity about what to do. The model just needs to (1) authenticate, (2) look up the entity, (3) construct the correct API call. Eliminates comprehension and selection errors.

**Expected success rate**: ~65-75% (model already succeeds on similar clear-intent tasks in tau bench)

#### Scenario Type 2: Single-Action Selection (30% of training)

**Target failure modes**: Wrong variant/flight selection (Category B reasoning errors)

The user states a preference that requires the model to look up options and select the correct one. This is harder than Type 1 but still single-action.

**Retail examples:**
```python
UserInstructions(
    domain="retail",
    reason_for_call="You want to exchange the Desk Lamp in order #W9300146 for a cheaper one. You'd like the cheapest available option.",
    known_info="Your name is Aarav Anderson. Your zip code is 19031.",
    task_instructions="You are patient and agreeable."
)
# Model must: get_product_details → find cheapest variant → exchange_delivered_order_items
# This is the same as tau bench Task 44, which the model failed on (picked 2nd cheapest)

UserInstructions(
    domain="retail",
    reason_for_call="You want to exchange the Wireless Earbuds in order #W3470184 for a pair with IPX4 water resistance (to match your other earbuds). Get the cheapest IPX4 option.",
    known_info="Your name is Aarav Anderson. Your zip code is 19031.",
    task_instructions="You are precise and detail-oriented."
)
# Explicit "IPX4" constraint — no ambiguity about which water resistance level
```

**Airline examples:**
```python
UserInstructions(
    domain="airline",
    reason_for_call="You want to change reservation XYZ789 to the cheapest direct economy flight on May 24 from JFK to LAX.",
    known_info="Your name is Michael Chen. Your user id is michael_chen_5566.",
    task_instructions="You are efficient."
)
# Model must: search_direct_flight → find cheapest → update_reservation_flights
```

**Simplification vs tau bench**: We reduce the number of viable options (use entities with 3-5 variants instead of 12+), and make the selection criterion unambiguous ("cheapest," "largest," "same color different size"). We also make the constraint explicit rather than requiring inference from context.

**Expected success rate**: ~40-55% (harder than Type 1, but simpler than tau bench versions of these tasks)

#### Scenario Type 3: Policy-Gated (20% of training)

**Target failure modes**: Spurious actions (Category B), reinforces refusal capability

**50% of these are genuinely forbidden (model must refuse), 50% are allowed (model must act)**. This prevents the model from learning to always refuse or always act.

```python
# Forbidden: Cancel non-pending order
UserInstructions(
    domain="retail",
    reason_for_call="You want to cancel order #W1234567 because you changed your mind.",
    known_info="Your name is Jane Park. Your zip code is 90210.",
    task_instructions="You are polite but insistent."
)
# Order is delivered → model should refuse and explain why

# Allowed: Cancel eligible reservation
UserInstructions(
    domain="airline",
    reason_for_call="You want to cancel reservation ABC123 due to a medical emergency. You have travel insurance.",
    known_info="Your name is Emma Kim. Your user id is emma_kim_9957.",
    task_instructions="You are worried but polite."
)
# Reservation has insurance, valid reason → model should proceed
```

**Expected success rate**: ~50-60% (model is already decent at refusal, but sometimes over-refuses or under-refuses)

### 5.4 Ground Truth Oracle (Verification)

For each generated scenario, we compute the expected final state programmatically:

```python
def compute_ground_truth(scenario, db):
    """Compute expected DB state and evaluation criteria."""

    gold_env = environment_constructor()

    # Apply the expected golden actions to a copy of the DB
    for action in scenario.expected_actions:
        gold_env.make_tool_call(
            tool_name=action.name,
            requestor="assistant",
            **action.arguments,
        )

    # The gold DB hash is what we compare against
    gold_db_hash = gold_env.get_db_hash()

    return EvaluationCriteria(
        actions=scenario.expected_actions,       # Same format as tau bench
        communicate_info=scenario.communicate_info,  # What agent should tell user
        reward_basis=[RewardType.DB, RewardType.COMMUNICATE],  # Same as tau bench
    )
```

**This is the exact same verification mechanism as tau bench** (`evaluator_env.py` lines 87-118):
1. Create a gold environment and replay expected actions
2. Create a predicted environment and replay the agent's actual trajectory
3. Compare `get_db_hash()` outputs
4. Check `communicate_info` via substring matching in agent messages

**Key property**: The verification is **fully programmatic** (no LLM judge). Tau bench uses an LLM judge for `nl_assertions`, but we omit this component. Our `reward_basis = [DB, COMMUNICATE]` uses only deterministic checks.

### 5.5 Why We Keep COMMUNICATE in the Reward

The original plan proposed dropping COMMUNICATE to isolate the tool-calling signal. We now KEEP it because:

1. **It matches tau bench exactly** — no reward format gap to create transfer issues
2. **The model is already good at communication** — in 72.7% of DB failures, COMMUNICATE passed. So COMMUNICATE rarely penalizes the model; it's not a bottleneck.
3. **It provides additional positive signal** — correct communication reinforces the right conversation patterns
4. **It catches completeness** — e.g., "refund amount is $17.99" must be communicated after an exchange

### 5.6 GameEnv Protocol Implementation

```python
class TauToolCallingEnv:
    """Simplified tau bench with LLM user and programmatic verification.

    Implements the GameEnv protocol for GRPO/PPO training.
    Uses the EXACT same tau2-bench infrastructure:
    - Same tools (via tau2 domain environment)
    - Same DB (via tau2 db.json)
    - Same user simulator (GPT-4.1 via tau2 UserSimulator)
    - Same verification (DB hash + communicate check)
    """

    supports_structured_messages = True  # For function-calling format

    # GameEnv protocol
    done: bool
    current_player: int  # 0 = agent (always, single-player)
    rewards: Dict[int, float]
    invalid_player: Optional[int]

    def __init__(self, domain="retail", max_steps=30, user_client=None):
        self.domain = domain
        self.max_steps = max_steps
        self.user_client = user_client  # UserLLMClient for GPT-4.1

        # Load tau2-bench infrastructure
        self.env_constructor = registry.get_env_constructor(domain)
        self.environment = self.env_constructor()
        self.tools = self.environment.get_tools()
        self.policy = self.environment.get_policy()
        self.tool_defs = [t.openai_schema for t in self.tools]

    def reset(self, seed: int) -> None:
        """Generate new scenario and initialize episode."""
        self.scenario = generate_scenario(seed, self.domain)

        # Initialize tau2-bench user simulator with our generated instructions
        self.user_sim = UserSimulator(
            tools=None,
            instructions=self.scenario.user_instructions,
            llm="gpt-4.1",
            llm_args={"temperature": 0.0, "seed": seed},
        )

        # Fresh environment for this episode
        self.environment = self.env_constructor()

        # Initialize orchestrator (same as tau bench)
        # First message: agent says "Hi! How can I help you today?"
        # User LLM responds with their request

        self.step_count = 0
        self.done = False
        self.current_player = 0
        self.rewards = {0: 0.0}
        self.invalid_player = None

    def get_system_prompt(self) -> str:
        """Same agent system prompt as tau bench."""
        return (
            "<instructions>\n"
            "You are a customer service agent that helps the user according to "
            "the <policy> provided below.\n"
            "In each turn you can either:\n"
            "- Send a message to the user.\n"
            "- Make a tool call.\n"
            "You cannot do both at the same time.\n\n"
            "Try to be helpful and always follow the policy. "
            "Always make sure you generate valid JSON only.\n"
            "</instructions>\n"
            f"<policy>\n{self.policy}\n</policy>"
        )

    def step(self, action: Optional[str]) -> None:
        """Process agent action (tool call or text message to user)."""
        # Parse action as tool call or text
        # If tool call: execute on environment, return result
        # If text: send to user LLM, get response
        # Check termination: max_steps, user ###STOP###
        # On termination: compute reward via DB hash + communicate check
        ...
```

### 5.7 Training Format Alignment

The training format matches tau bench evaluation exactly:

```python
# Same tokenization as tau bench evaluation
tokenizer.apply_chat_template(
    msgs,
    add_generation_prompt=True,
    tools=tau_bench_tool_defs,  # Same OpenAI-format tool schemas
    enable_thinking=True,
)
```

Each agent turn in the multi-turn episode produces a `StepSample` for training, with the episode-level reward assigned to all turns (same as existing game implementations).

---

## 6. Failure Coverage and Transfer Analysis

### 6.1 Which Failures Does the Microenvironment Address?

| Failure Category | Count | % | Microenv Coverage | Mechanism |
|-----------------|-------|---|-------------------|-----------|
| **A: Mechanical tool errors** | ~24 | 22% | **Direct** | Practice with correct API formats, entity IDs, payment methods |
| **B: Wrong variant/flight selection** | ~31 | 28% | **Direct (Type 2)** | Simpler selection tasks build variant-selection skills |
| **B: Policy violations** | ~10 | 9% | **Direct (Type 3)** | Policy-gated scenarios train correct refusal |
| **B: Wrong tool selected** | ~8 | 7% | **Direct (Type 1)** | Explicit intent → model learns correct tool for each action type |
| **B: Incomplete multi-step** | ~7 | 6% | **Indirect** | Single-action mastery is prerequisite; multi-step NOT directly trained |
| **B: Wrong entity targeting** | ~4 | 4% | **Direct (Type 1)** | Single-entity scenarios eliminate cross-entity confusion |
| **C: Timeouts** | 8 | 7% | **Indirect** | Shorter, more decisive episodes train efficient conversation flow |
| **C: Communication-only** | 6 | 5% | **Partial** | COMMUNICATE reward component addresses some |
| **C: Auth/system failures** | 4 | 4% | **Direct** | Repeated authentication practice |
| **D: Comprehension failures** | ~8 | 7% | **Direct (Type 2)** | Explicit constraints reduce but don't eliminate comprehension load |

**Direct coverage: ~77 out of 110 failures (70%)**
**Indirect coverage: ~15 additional failures (14%)**
**Not covered: ~18 failures (16%)** — primarily multi-step orchestration, complex policy edge cases, and some communication-only failures

### 6.2 Transfer Mechanism

Why does training on simplified single-action tasks improve performance on harder tau bench tasks?

1. **API call patterns transfer directly**: Once the model learns that `exchange_delivered_order_items` requires `item_ids` (list), `new_item_ids` (list), `order_id` (string with #W prefix), and `payment_method_id` (string), this knowledge applies identically in multi-action contexts.

2. **Variant selection transfers**: Learning to scan a table of product variants and select by price/attribute in a simple context teaches the same scanning skill needed in complex contexts.

3. **Authentication patterns transfer**: Correctly capitalizing names, handling zip codes, using `find_user_id_by_name_zip` vs `find_user_id_by_email` — these patterns are context-independent.

4. **Policy compliance transfers**: Learning when cancellation IS and ISN'T allowed transfers directly (same policy text).

5. **Conversation flow transfers**: The authenticate → gather → act → communicate pattern is the same regardless of task complexity.

### 6.3 What Does NOT Transfer (Limitations)

1. **Multi-step orchestration**: Single-action training does not teach the model to sequence 3 write calls in the right order. This requires separate training (future microenvironment or curriculum stage).

2. **Cross-entity reasoning**: Tasks requiring comparison across multiple orders/reservations are not in scope.

3. **Adversarial resistance**: Our 20% policy-gated scenarios provide some adversarial exposure, but not the sustained multi-turn pressure of tau bench's hardest adversarial tasks.

4. **Complex policy edge cases**: Some tau bench tasks test obscure policy rules that are unlikely to appear in our generated scenarios.

---

## 7. Non-Regression Analysis

### 7.1 The Model's Existing Strengths (54 Successes)

| Strength Category | Count | Domain | Risk of Regression |
|-------------------|-------|--------|-------------------|
| Policy refusal | 14 | Mostly airline | **Low** — Type 3 (policy-gated) reinforces this |
| Simple single-write retail | 26 | Retail | **None** — Type 1 trains the same pattern |
| Multi-write retail | 13 | Retail | **Low** — single-write doesn't conflict |
| Info/communication | 1 | Mixed | **Low** — COMMUNICATE reward preserves this |

### 7.2 Potential Regression Risks and Mitigations

**Risk 1: Action bias** — Training heavily on action tasks could make the model less likely to refuse.
- **Mitigation**: 20% of scenarios are policy-gated, with 50% genuinely forbidden. The model receives POSITIVE reward for correct refusal.

**Risk 2: Conversation style shift** — Simplified conversations could make the model less capable of handling complex user interactions.
- **Mitigation**: We use the SAME GPT-4.1 user LLM, which produces natural, varied responses. The model must handle the same conversation dynamics.

**Risk 3: Overfitting to simple tasks** — The model could learn shortcuts that don't transfer.
- **Mitigation**: KL penalty prevents excessive drift from base policy. Type 2 scenarios (selection tasks) maintain reasoning requirements. Periodic evaluation on full tau bench tasks catches regression.

### 7.3 Formal Non-Regression Argument

For any task T ∈ S (current success set):
1. The microenvironment uses the same tools, DB, and policy as T
2. When the microenvironment generates a scenario similar to T, the expected actions are the same actions the model already takes correctly on T
3. The model receives reward=1.0 for these actions → positive reinforcement
4. The KL penalty prevents the model from drifting far enough from base behavior to lose existing capabilities
5. Therefore: P(success on T after training) ≥ P(success on T before training) - ε, where ε → 0 with sufficient KL penalty

---

## 8. Database and Tool Reuse

### 8.1 Database
The microenvironment loads the **exact same** `db.json` files from tau2-bench:
- `/root/games/tau2-bench/data/tau2/domains/airline/db.json`
- `/root/games/tau2-bench/data/tau2/domains/retail/db.json`

### 8.2 Tools
Imported directly from tau2-bench:
- `/root/games/tau2-bench/src/tau2/domains/airline/tools.py` (FlightTools)
- `/root/games/tau2-bench/src/tau2/domains/retail/tools.py` (RetailTools)
- `/root/games/tau2-bench/src/tau2/environment/toolkit.py` (GenericToolKit)

Tool schemas extracted via the same `as_tool()` / `openai_schema` mechanism.

### 8.3 Policy
Loaded from the same policy markdown files:
- `/root/games/tau2-bench/data/tau2/domains/airline/policy.md`
- `/root/games/tau2-bench/data/tau2/domains/retail/policy.md`

### 8.4 User Simulator
**Exact reuse** of tau2-bench's `UserSimulator`:
- `/root/games/tau2-bench/src/tau2/user/user_simulator.py`
- Guidelines: `/root/games/tau2-bench/data/tau2/user_simulator/simulation_guidelines.md`
- LLM: GPT-4.1, temperature 0.0, deterministic seed per task

---

## 9. Implementation Plan

### Phase 1: Infrastructure
1. Create `tau_tool_calling_env/` directory
2. Import and wrap tau2-bench Environment, Tools, DB, UserSimulator
3. Implement `GeneratedTask` dataclass matching tau bench `Task` format
4. Implement `compute_ground_truth()` using tau bench's environment evaluator

### Phase 2: Scenario Generation
5. Implement `generate_explicit_action()` (Type 1) for retail
6. Implement `generate_explicit_action()` (Type 1) for airline
7. Implement `generate_selection_action()` (Type 2) for retail
8. Implement `generate_selection_action()` (Type 2) for airline
9. Implement `generate_policy_gated()` (Type 3) for both domains

### Phase 3: GameEnv Protocol
10. Implement `TauToolCallingEnv` (GameEnv protocol)
11. Integrate with tau2-bench Orchestrator for message routing
12. Implement reward computation (DB hash + communicate check)
13. Register in `game_registry.py`

### Phase 4: Training Integration
14. Implement `extract_action()` for tool call parsing (reuse from adversarial_policy_game)
15. System prompt matching tau bench agent format
16. `StepSample` generation from multi-turn episodes

### Phase 5: Validation
17. Unit tests: oracle correctness (generated ground truth matches manual verification)
18. Format alignment: compare tokenized output with tau bench evaluation
19. Success rate calibration: run base model on 100 generated scenarios, verify ~50-60% success
20. Non-regression: evaluate base model on tau bench success tasks

---

## 10. Expected Impact

### 10.1 Direct Coverage
- ~70% of tau bench failures (77/110) are directly targeted
- ~14% (15/110) are indirectly addressed
- ~16% (18/110) are out of scope (primarily multi-step orchestration)

### 10.2 Cost Per Episode
With GPT-4.1 as user simulator:
- ~5 user LLM calls per episode × ~500 tokens per call = ~2,500 tokens per episode
- At GPT-4.1 pricing: ~$0.005-0.01 per episode
- For GRPO with batch_size=64, K=6: 384 episodes × $0.008 ≈ $3.07 per training batch
- For 1000 batches: ~$3,070 total user LLM cost
- This is significant but manageable for a focused training run

### 10.3 Conservative Success Rate Improvement Estimate
If training improves the model's tool-calling accuracy on the ~70% of addressable failure modes by 40%:
- Current failures: 110
- Addressable: 77
- Fixed: 77 × 0.40 = ~31
- New successes: 54 + 31 = 85
- New success rate: 85/164 = **~52%** (up from 32.9%)

### 10.4 Curriculum Extension (Future Work)
After mastering single-action tasks (~60%+ success on microenvironment), the next curriculum stage would introduce:
- **Stage 2**: Two-action tasks (e.g., modify items + modify address)
- **Stage 3**: Three+ action tasks with cross-entity reasoning
- **Stage 4**: Full tau bench task distribution

Each stage builds on the skills learned in the previous stage, creating a smooth learning gradient from ~55% → ~70% → full tau bench performance.
