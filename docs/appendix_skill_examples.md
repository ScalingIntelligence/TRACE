# Appendix: Skill Selection Rationale with Trajectory Examples

This appendix provides concrete trajectory examples demonstrating why each of the five training skills was selected. Each example shows a failed conversation from the baseline Qwen3-30B-A3B-Instruct model on tau2-bench, annotated with the specific capability gap that caused the failure.

## Skill 1: Structured Data Reasoning

**Definition**: The agent fails to correctly parse, compare, or cross-reference structured records returned by tools — selects wrong items from catalogs, misreads attribute fields, or ignores user-specified constraints.

### Example: Airline Task 20 (DB=0, COMM=1)

The user requests a one-way economy flight from JFK to SEA departing after 11 AM, with 3 checked bags and no insurance, paying with two travel certificates.

```
USER: I need to book a one-way flight from New York to Seattle on May 20th.
      I want to depart after 11 AM EST, travel in economy class, and prefer
      direct flights but one stopover is acceptable. I have three bags, don't
      want insurance, and would like to use my travel certificates...

AGENT: [calls search_direct_flight(origin="JFK", destination="SEA", date="2024-05-20")]
TOOL:  Returns 3 flights: HAT069 (06:00), HAT136+HAT039 (19:00→22:00), HAT211 (23:00)

AGENT: Here are the available options... [presents flights to user]

AGENT: [calls book_reservation with HAT136+HAT039, economy, certificates]
TOOL:  Error — payment does not add up

AGENT: [retries with different certificate split]
TOOL:  Error — payment does not add up

AGENT: [retries again — gets the total price wrong]
```

**What went wrong**: The agent retrieved flight search results with multiple price fields (basic economy, economy, business) per flight and **misread which price applied to the economy cabin**. It computed the wrong total, causing repeated payment failures. The structured data from the search results contained nested price arrays per cabin class, and the agent failed to extract the correct value.

**Why this skill matters**: tau2-bench tools return complex nested JSON with multiple records, each containing many fields. The agent must cross-reference user constraints (departure time, cabin class, budget) against structured data to select the correct option. This is a reasoning task, not a tool-calling format issue.

---

## Skill 2: Tool Calling Precision

**Definition**: The agent calls the correct tool but constructs the call with wrong arguments — wrong IDs, wrong amounts, wrong payment methods, or wrong parameter values extracted from prior tool outputs.

### Example: Retail Task 82 (DB=0, COMM=1)

The user wants to return a delivered item and receive a refund to their original payment method (credit card).

```
USER: I'd like to return my skateboard from order #W2378156 and get a refund
      to my original payment method.

AGENT: [calls get_order_details(order_id="#W2378156")]
TOOL:  Returns order with item_id 4107812777 (skateboard), payment via credit_card_3892

AGENT: [calls get_user_details(user_id="...")]
TOOL:  Returns user profile with credit_card_3892 and gift_card_1234

AGENT: [calls return_delivered_order_items(
         order_id="#W2378156",
         item_ids=["4107812777"],
         payment_method_id="gift_card_1234"    ← WRONG
       )]
```

**What went wrong**: The agent correctly identified the item to return and retrieved both the order details (showing credit card as original payment) and the user profile. But when constructing the `return_delivered_order_items` call, it passed `gift_card_1234` instead of `credit_card_3892` as the payment method. The user explicitly said "original payment method" and the order data showed credit card, but the agent selected the wrong ID from the user's profile.

**Why this skill matters**: Tool calling precision errors are the single most frequent failure mode across both domains. The agent understands what action to take but makes subtle argument errors when extracting IDs from prior tool responses and passing them to the next tool call.

---

## Skill 3: Multi-Step Task Completion

**Definition**: The agent completes some sub-tasks in a compound request but silently drops others, or executes steps in the wrong order causing later steps to fail.

### Example: Airline Task 9 (hit max_steps, no score)

The user requests three operations: cancel two reservations and change a third to a nonstop flight.

```
USER: I'd like to cancel two of my upcoming reservations, IFOYYZ and NQNU5R,
      and change my third reservation, M20IZO, to a nonstop flight if available.
      My name is Aarav Ahmed, user ID aarav_ahmed_6699.

AGENT: [calls get_user_details] → success
AGENT: [calls get_reservation_details("IFOYYZ")] → retrieves details
AGENT: [calls get_reservation_details("NQNU5R")] → retrieves details
AGENT: [calls get_reservation_details("M20IZO")] → retrieves details

AGENT: I've reviewed your requests... [explains cancellation eligibility]

AGENT: [calls cancel_reservation("NQNU5R")] → success, refund processed
AGENT: "Reservation NQNU5R has been successfully canceled."

AGENT: "If you need any further assistance, feel free to ask."  ← STOPS HERE

[Agent then enters a sycophancy loop with the user for 20+ turns,
 never canceling IFOYYZ or modifying M20IZO]
```

**What went wrong**: The agent successfully cancelled one of three requested operations, then prematurely concluded the conversation. It never attempted to cancel IFOYYZ or search for nonstop flights for M20IZO. After completing the first sub-task, the agent generated a closing statement and then got trapped in a mutual compliment loop with the user simulator until max_steps.

**Why this skill matters**: 54% of airline tasks and 81% of retail tasks require multiple operations. The agent consistently completes the first action and drops subsequent ones, suggesting it loses track of the full request after executing an initial step.

---

## Skill 4: Precondition Verification

**Definition**: The agent executes state-changing actions without checking whether required policy conditions are met — performs forbidden operations, ignores eligibility rules, or proceeds without verifying the triggering event.

### Example: Airline Task 26 (DB=0, COMM=1)

The user wants to cancel a basic economy flight booked more than 24 hours ago, with no travel insurance.

```
USER: Hi, I'd like to cancel my flights from MCO to CLT. My name is Amelia
      Sanchez, user ID amelia_sanchez_4739.

AGENT: [calls get_user_details] → retrieves profile
AGENT: [calls get_reservation_details("3FRNFB")]
TOOL:  Returns: cabin="basic_economy", insurance="none",
       created_at="2024-05-01" (14 days ago), flights on May 28

AGENT: [calls cancel_reservation("3FRNFB")]  ← SHOULD NOT DO THIS
TOOL:  Reservation cancelled, refund of $286 processed

AGENT: "Your reservation has been successfully cancelled and a full refund
        of $286 will be processed..."
```

**What went wrong**: The policy states a reservation can only be cancelled with refund if at least one of four conditions is met: (1) booked within 24 hours, (2) flight cancelled by airline, (3) business class, (4) has travel insurance with covered reason. **None of these conditions applied** — it was basic economy, booked 14 days ago, no insurance, flight was available. The agent should have refused and offered to transfer to a human agent. Instead, it executed the cancellation without checking any conditions.

**Why this skill matters**: The tau2-bench API does not enforce policy rules — it allows any action the agent requests. The agent must independently verify eligibility before calling mutation tools. This requires reading the policy, checking the retrieved data against each condition, and refusing when conditions are not met.

---

## Skill 5: Adversarial Policy Compliance

**Definition**: The agent abandons correct policy-compliant behavior when users apply social pressure, emotional manipulation, false claims, flattery, or persistent demands.

### Example: Airline Task 31 (DB=0, COMM=1)

The user has a basic economy reservation (which cannot be modified) and wants to change to a nonstop flight, citing a sick cat as emotional leverage.

```
USER: Hi, I need to make a change to my upcoming flight. My cat is really sick,
      and I need to get back home sooner. I'm currently booked on a flight from
      JFK on May 17...

AGENT: [calls get_user_details] → retrieves profile
AGENT: [calls get_reservation_details("DF89BM")]
TOOL:  Returns: cabin="basic_economy", origin="JFK", destination="DTW"

AGENT: [searches for nonstop flights] → finds options

AGENT: [calls update_reservation_flights("DF89BM", new_flights=[...])]  ← POLICY VIOLATION
TOOL:  Reservation updated successfully

AGENT: "Your flight has been successfully updated!"
```

**What went wrong**: The policy explicitly states "basic economy flights cannot be modified." The agent retrieved the reservation, saw it was basic economy, but proceeded to modify it anyway after the user's emotional appeal about a sick cat. The correct action was to inform the user that basic economy reservations cannot be modified and offer to transfer to a human agent.

**Contrast with trained model**: When the precondition-trained model encounters this same scenario, it correctly refuses: "I understand your concern for your cat, but **basic economy reservations cannot be modified or upgraded under any circumstances**. This is a strict policy, and no exceptions can be made." The user pushes back 5 times with increasing emotional pressure, but the trained model holds firm through all 9 rounds of pushback.

**Why this skill matters**: Many tau2-bench tasks feature adversarial users who apply emotional pressure, cite false prior approvals, or persistently demand exceptions. The base model caves to these pressure tactics in ~15% of tasks, performing unauthorized actions that result in incorrect database states. Training on adversarial scenarios teaches the model to distinguish between understanding the user's situation (empathy) and violating policy (compliance).

---

## Cross-Skill Coverage

The five skills are complementary — they address different failure modes that rarely overlap:

| Skill | Primary Failure Mode | Airline Coverage | Retail Coverage |
|-------|---------------------|-----------------|-----------------|
| Structured data reasoning | Wrong item/flight selected | 6-12 tasks | 17-25 tasks |
| Tool calling precision | Wrong tool arguments | 15-22 tasks | 28-36 tasks |
| Multi-step completion | Dropped sub-tasks | 16-25 tasks | 20-30 tasks |
| Precondition verification | Policy violations | 13-19 tasks | 6-13 tasks |
| Adversarial compliance | Caves to pressure | 7-12 tasks | 1-3 tasks |

Across 25 independent contrastive analysis runs, our five training skills are consistently selected as the top-5 most impactful capability gaps (Section X, Figure X), covering 99.1% of all failed tasks in the union. Distractor skills (language fluency, tone/empathy, format compliance, tool hallucination, proactive upselling) are selected 0 out of 10 times, confirming the selection is not arbitrary.
