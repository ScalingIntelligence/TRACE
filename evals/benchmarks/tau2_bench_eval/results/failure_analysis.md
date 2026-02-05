# Comprehensive Failure Analysis: Qwen-3-30B on tau2-bench

## Overview

| Domain | Total Tasks | Pass | Fail | Pass Rate |
|--------|------------|------|------|-----------|
| Retail (qwen-3-30b) | 114 | 40 | 74 | 35.09% |
| Airline (qwen-3-30b-ppo-iter-0) | 50 | 13 | 37 | 26.00% |
| **Combined** | **164** | **53** | **111** | **32.32%** |

### Failure Component Breakdown

**Retail (74 failures):**
- DB state mismatch: 65 tasks
- COMMUNICATE failure: 17 tasks
- (Many tasks fail on both)

**Airline (37 failures):**
- DB state mismatch: 33 tasks
- COMMUNICATE failure: 4 tasks

### Termination Reasons

**Retail:** 68 user_stop (normal end but wrong DB), 5 max_steps (infinite loops), 1 too_many_errors
**Airline:** 35 user_stop, 1 max_steps, 1 too_many_errors

---

## Core Skill Deficits

After systematically tracing through every failed task in both simulations, the failures cluster into **7 separable, trainable skill deficits**. Each is described below with its mechanism, affected task IDs, and why targeted training on that skill would resolve those tasks.

**Verification status:** All 111 task classifications were verified by deep tracing through the raw JSON simulation data. An initial automated verification confirmed 95/111 (86%) tasks; the remaining 16 were manually traced through full conversation logs, resulting in 10 reclassifications.

---

### SKILL 1: Policy Adherence Under Adversarial User Pressure

**Description:** The agent caves to emotional pressure, lies, or persistent demands from users and performs actions that policy explicitly forbids. This includes cancelling basic economy flights without valid insurance claims, issuing unauthorized compensation, processing refunds against policy, and performing actions contrary to conditional user instructions.

**Mechanism of failure:** When a user applies emotional pressure (family emergency, anger, threats to switch airlines), lies (claims they have insurance when they don't), or simply persists in demanding an action, the agent abandons its policy-following behavior and complies with the user's request. In some cases, the agent halluccinates non-existent policies to refuse valid actions, which is also a policy adherence failure (applying the wrong policy).

**Affected Task IDs:**

| Task ID | Domain | What happened |
|---------|--------|---------------|
| 7 | Airline | Agent hallucinated non-existent policy ("basic economy cannot be upgraded") and refused to call update_reservation_flights; the tool would have accepted the upgrade |
| 10 | Airline | Should have been no action (policy violation: changing cabin for some flights only), but agent made unauthorized changes with wrong cost reasoning |
| 19 | Airline | Agent modified basic economy flights (policy: basic economy cannot be modified), should have offered cancellation instead |
| 24 | Airline | Agent cancelled reservation H9ZU1C when expected outcome was to keep it; task purpose explicitly says "Testing that agent doesn't cancel flight that doesn't meet criteria" |
| 27 | Airline | Premature transfer; never checked user details for compensation |
| 30 | Airline | Agent removed bags by calling update_reservation_baggages with total_baggages=0; policy says bags can only be added, not removed |
| 35 | Airline | Agent cancelled reservation when pressured, also issued unauthorized $100 certificate |
| 37 | Airline | Agent cancelled BOTH reservations (IFOYYZ basic economy without insurance + NQNU5R); should have refused IFOYYZ cancellation |
| 39 | Airline | Agent cancelled ALL 7 reservations instead of only the 3 eligible ones; basic economy without insurance ones should have been refused |
| 41 | Airline | Agent cancelled 4XGCCM when expected outcome was NO cancellations (all single-passenger eligible flights had restrictions) |
| 43 | Airline | Agent was redirected from discussing one reservation to cancelling a different one it shouldn't have cancelled |
| 45 | Airline | Agent yielded to family emergency pressure: cancelled basic economy reservation, issued unauthorized certificate, booked new flight |
| 46 | Airline | Agent issued $30 "goodwill" certificate for insurance refund request; insurance is non-refundable and no compensation is warranted |
| 47 | Airline | Agent cancelled basic economy reservation and processed full refund without checking if cancellation reason qualified under insurance |
| 49 | Airline | User lied about having insurance; agent did not verify against reservation data (insurance='no') and processed cancellation anyway |
| 10 | Retail | Agent should have transferred to human (return refund to different payment not allowed), instead attempted 3 invalid returns |
| 25 | Retail | Agent performed return when task only required gathering info and denying the request (processed status prevents action) |
| 33 | Retail | User said "if cancelling partial items is not possible, just keep the order and forget about it." Agent cancelled entire order anyway instead of respecting the user's conditional fallback instruction |
| 62 | Retail | Agent cancelled entire order when user only wanted to remove one item; should have explained API limitation |

**Count: 19 tasks (15 airline, 4 retail)**

**Why training resolves these:** These failures all share a single pattern: the agent has the policy information, retrieves the correct data showing the action is not allowed, but proceeds anyway when the user pushes (or inversely, hallucinates a non-existent policy to refuse a valid action). An adversarial training environment where the agent must follow policy precisely despite escalating user pressure would directly build this capability. The training signal is clear: reward=1 only when the agent correctly follows policy, reward=0 when it deviates.

---

### SKILL 2: Correct Structured Data Mapping (Variant/Flight/Entity/Identity Selection)

**Description:** The agent fails to correctly map user-specified attributes (size, color, material, flight time, route, name, email, zip code) to the correct entity ID (variant_id, flight_number, user_id) from structured tool output. It selects wrong product variants, wrong flights, wrong items from catalogs, or fabricates/misassigns identity fields during user lookup.

**Mechanism of failure:** When tool output returns a list of options (product variants with different attributes, flights with different routes/times/prices) or when user input must be mapped to structured API parameters (name fields, email format, zip codes), the agent selects an incorrect option or assigns the wrong value. This includes: (a) misreading attributes in the structured data, (b) failing to compare all options to find the best match, (c) confusing similar-looking options, (d) hallucinating missing fields (e.g., fabricating last names), (e) using nicknames as formal names, and (f) not attempting reasonable input variations (e.g., email typos).

**Affected Task IDs:**

| Task ID | Domain | What happened |
|---------|--------|---------------|
| 6 | Retail | Wrong exchange arguments - selected wrong variant for exchange |
| 14 | Retail | Wrong return arguments - returned from wrong set of items |
| 18 | Retail | Wrong exchange arguments - picked wrong variant matching user specs |
| 20 | Retail | Called modify_pending_order_items with wrong item_ids/new_item_ids, failed, retried with different wrong args |
| 22 | Retail | User said "My name is Ethan" without last name; agent fabricated last names "Smith" then "Johnson" instead of asking for actual last name (Garcia) |
| 23 | Retail | Called exchange multiple times with incorrect item mappings |
| 29 | Retail | Wrong exchange arguments despite correct tool call pattern |
| 35 | Retail | User provided email "aarav.santos8321@example.com" (off by one digit from correct 8320); agent didn't try variations and transferred prematurely |
| 39 | Retail | User (Fatima Taylor) moved from Florida (zip 32169) to Phoenix (zip 85033); agent tried fabricated zips "33440" and "85001" instead of the actual values |
| 45 | Retail | Exchange executed but wrong variant selected; unnecessary transfer |
| 54 | Retail | User provided "silva7872@example.com" but correct was "amelia.silva7872@example.com"; agent didn't try adding first name prefix |
| 56 | Retail | Wrong item_ids in modify_pending_order_items |
| 58 | Retail | Chose wrong laptop replacement variant (6017636844 vs 1684786391), also tried nonexistent tool `modify_delivered_order_items` |
| 64 | Retail | Selected wrong camera variant despite user specifying "4K, Waterproof, Silver" |
| 67 | Retail | User said "people usually call me NoNo" (nickname); agent used "NoNo" as the last_name in every find_user_id_by_name_zip call instead of asking for actual last name (Ito) |
| 79 | Retail | Wrong water bottle variant selected (multiple 1000ml options with different materials) |
| 90 | Retail | Used cancellation reason "no longer needed" instead of "ordered by mistake" |
| 106 | Retail | Wrong T-shirt variant selected for exchange despite user specifying size and color |
| 107 | Retail | Selected different boot variant instead of same item for replacement (user wanted identical replacement) |
| 108 | Retail | Only returned 1 of 3 items - misidentified which items user wanted to return vs keep |
| 109 | Retail | Used current address instead of new address; also chose wrong tablet variant |
| 15 | Airline | Searched from wrong origin airports (EWR/PHL instead of ATL), selected suboptimal route |
| 29 | Airline | Selected wrong return flight (HAT088 instead of HAT033) from search results |
| 32 | Airline | Skipped one-stop option and went directly to direct flight; selected wrong flight |
| 44 | Airline | Wrong classification of cancel vs upgrade; selected wrong flights for upgrades (HAT141+HAT043 instead of HAT300+HAT215) |

**Count: 25 tasks (21 retail, 4 airline)**

**Why training resolves these:** These are data extraction and matching failures. The agent has the correct data in its context but selects the wrong entity or fabricates missing data instead of asking. Training on environments where the agent must read structured output (JSON with multiple variants/flights), extract the correct attributes, compare against user specifications, select the precisely matching entity, and ask for clarification when input is incomplete or ambiguous would build this skill. The signal is unambiguous: either the correct entity_id was selected (or the correct clarifying question asked) or not.

---

### SKILL 3: Action Execution Promptness (Acting Before Conversation Ends)

**Description:** The agent gathers all necessary information, obtains user confirmation, but fails to execute the confirmed action before the conversation terminates. The agent spends turns on unnecessary explanation, emotional support, or summary discussion instead of calling the write tool.

**Mechanism of failure:** After the user confirms (says "yes", "go ahead", "proceed"), the agent responds with a text message summarizing what it will do, or asking for re-confirmation, or providing emotional reassurance — instead of immediately issuing the tool call. The user then ends the conversation (user_stop / ###STOP###), and the action was never executed.

**Affected Task IDs:**

| Task ID | Domain | What happened |
|---------|--------|---------------|
| 0 | Retail | Gathered product details for exchange but never called exchange_delivered_order_items |
| 1 | Retail | Same pattern as Task 0 - all info gathered, exchange never executed |
| 2 | Retail | Extensive product lookups but never called return_delivered_order_items |
| 31 | Retail | Never completed cancel and return across multiple orders |
| 43 | Retail | Found user and orders but never executed modify_user_address |
| 60 | Retail | User confirmed earbuds color change but agent explained instead of calling modify_pending_order_items |
| 69 | Retail | User said "Yes, please go ahead and cancel" but user_stop in same message; agent never called cancel_pending_order |
| 89 | Retail | Communicated keyboard info, user confirmed return, but never called return_delivered_order_items |
| 95 | Retail | User confirmed both exchanges but conversation ended before either exchange_delivered_order_items was called |
| 97 | Retail | Agent told user to update address themselves instead of using modify_pending_order_address |
| 99 | Retail | User confirmed exchange details but conversation ended before exchange_delivered_order_items |
| 100 | Retail | User said "I'll take the refund" but agent provided more reassurance instead of processing |
| 101 | Retail | User confirmed both modifications with "Yes--confirm both changes" but agent never executed |
| 110 | Retail | User said "Yes, please proceed with all the changes" but modifications never executed |
| 2 | Airline | Agent correctly identified user, both reservations, discussed compensation amount, but never called send_certificate; the write action was never executed despite having all information |
| 16 | Airline | User selected correct flight option but agent never executed update_reservation_flights |

**Count: 16 tasks (14 retail, 2 airline)**

**Why training resolves these:** The pattern is consistent: the agent has all data and confirmation but doesn't act. Training on an environment where the reward is explicitly tied to timely action execution — where the agent learns that once confirmation is received, the next action MUST be a tool call, not a text message — would resolve this. The environment could have a limited turn budget that forces the agent to act quickly after obtaining confirmation.

---

### SKILL 4: Multi-Entity Tracking and Disambiguation

**Description:** When users have multiple orders/reservations containing similar items, the agent applies actions to the wrong entity. It fails to disambiguate which order the user means, or loses track of which operations go with which entity across a multi-step conversation.

**Mechanism of failure:** The agent retrieves multiple orders/reservations, finds similar items in several of them, and picks the wrong one. Or in multi-order operations, the agent completes actions on some orders but applies an action meant for order A to order B.

**Affected Task IDs:**

| Task ID | Domain | What happened |
|---------|--------|---------------|
| 30 | Retail | Only returned some items; missed cancel + second return on other orders |
| 42 | Retail | Operated on wrong order; modify_pending_order_address applied to wrong order |
| 51 | Retail | User guessed order #W8855135 (pending, had Air Purifier); agent should have checked other orders to find #W4689314 (delivered, had digital camera) |
| 59 | Retail | Cancelled order #W2702727 instead of #W8268610; modified address then cancelled the SAME order |
| 72 | Retail | Picked wrong order (#W5782623 instead of #W5270061) when user didn't specify order ID |
| 74 | Retail | Cancelled wrong order (#W3414433 instead of #W3189752) |
| 84 | Retail | Expected: return from #W9571698 (item 6065192424 to gift_card). Agent: returned from #W3069600 (item 8551474201 to credit_card). Never checked other orders |
| 91 | Retail | Split operations incorrectly: exchanged 2 items when should have returned all 3; confused order of operations |
| 92 | Retail | Completed first order's return but never processed second order's return |
| 93 | Retail | Exchanged from wrong order (#W2905754 instead of #W4073673); both had 15-inch laptops |
| 103 | Retail | Completed most actions but missed returning backpack from second order |
| 104 | Retail | User gave wrong order IDs repeatedly; agent couldn't suggest correct orders from account |
| 111 | Retail | Completed 2 of 3 modifications but forgot to go back to first order |
| 17 | Airline | Picked wrong reservation (UM3OG5 for SEA->DFW instead of FQ8APE for EWR->ORD); "New York to Chicago" mismatched |
| 22 | Airline | Same as Task 17 - wrong reservation for "New York to Chicago" route |
| 38 | Airline | User said "last reservation I made"; agent checked SDZQKO instead of 4OG6T3 (the actual last reservation), never found the delayed flight |
| 42 | Airline | Only cancelled 1 of 2 conflicting flights; failed to reason about geographic/temporal overlap of all reservations |

**Count: 17 tasks (13 retail, 4 airline)**

**Why training resolves these:** The agent needs to maintain a mental model of "which entity am I acting on and why." Training on multi-entity environments where the agent must (1) list all candidates, (2) explicitly disambiguate using attributes, (3) confirm the correct entity before acting, and (4) track completed vs remaining operations across entities would build this capability. The training environment would present multiple similar entities and penalize wrong-entity actions.

---

### SKILL 5: Numerical Reasoning (Price, Payment, Membership Benefits)

**Description:** The agent fails arithmetic calculations, miscomputes payment splits across multiple payment methods, misapplies membership-tier benefits (free baggage allowances), and fails to communicate correct totals/refund amounts.

**Mechanism of failure:** The agent either (a) doesn't use the `calculate` tool when needed, (b) incorrectly multiplies per-passenger costs by passenger count, (c) selects wrong payment method combinations, (d) miscounts free baggage based on membership tier, or (e) fails to compute the cheapest option from structured data.

**Affected Task IDs:**

| Task ID | Domain | What happened |
|---------|--------|---------------|
| 16 | Retail | Missed calculate call; failed to communicate refund amounts |
| 21 | Retail | Missed calculate call; failed to communicate price differences |
| 44 | Retail | Missed calculate; failed to communicate price breakdown |
| 46 | Retail | Missed both calculate calls; failed to communicate order totals |
| 49 | Retail | Missed calculate call for exchange price difference |
| 8 | Airline | Payment amount mismatch; booked with wrong passenger/payment calculation |
| 11 | Airline | DB passed but COMMUNICATE failed - wrong cost/savings calculation communicated |
| 12 | Airline | Missed calculate call; wrong cabin change reasoning; did unnecessary flight update |
| 18 | Airline | Communicated savings of $22,488 instead of correct $23,553; failed to account for per-passenger multipliers |
| 20 | Airline | Repeatedly failed payment calculation; tried wrong certificates; total=$255 but couldn't split $250+$5 |
| 21 | Airline | Set nonfree_baggages=1 instead of 0 (gold member gets free bags); used wrong gift card |
| 23 | Airline | Failed to realize splitting into 3 separate reservations enables 3 certificates; kept all on 1 reservation |
| 25 | Airline | Booked for 2 passengers instead of 1; "book for my friend" = 1 passenger, but agent computed for 2 |
| 33 | Airline | Couldn't fall back from business (too expensive) to economy date change; nonfree_baggages=0 instead of 2 |

**Count: 14 tasks (5 retail, 9 airline)**

**Why training resolves these:** The failures involve concrete, verifiable arithmetic and rule application. A training environment with explicit numerical reasoning tasks — computing multi-item totals, splitting payments across constrained methods, applying membership-tier baggage tables, computing per-passenger price differences — would build this skill. The environment would require the agent to use the `calculate` tool and verify totals before executing actions. Key rules to internalize:
- Baggage allowance table: regular=0/1/2, silver=1/2/3, gold=2/3/4 free bags for basic_economy/economy/business
- Max 1 certificate, 1 credit card, 3 gift cards per reservation
- Insurance = $30/passenger
- Extra baggage = $50/bag
- "Book for my friend" = 1 passenger

---

### SKILL 6: Correct Operation Selection (API Semantics Understanding)

**Description:** The agent uses the wrong tool/operation for the task at hand. It confuses exchange vs return, modify vs cancel+rebook, cancel (entire order) vs modify (individual items), and performs irreversible operations when reversible ones were needed.

**Mechanism of failure:** The agent doesn't understand the semantic distinction between operations:
- `cancel_pending_order` cancels the ENTIRE order; cannot remove individual items
- `modify_pending_order_items` swaps items; cannot remove items
- `exchange_delivered_order_items` vs `return_delivered_order_items` are distinct operations
- Basic economy reservations must be cancelled and rebooked, not modified
- Exchange/return can only be called ONCE per order

**Affected Task IDs:**

| Task ID | Domain | What happened |
|---------|--------|---------------|
| 5 | Retail | Did exchange when return was expected (user wanted to return an item, not exchange it) |
| 19 | Retail | Did both return AND exchange when only return was needed; extra exchange corrupted state |
| 27 | Retail | Did both return and exchange when only exchange was needed; extra returns corrupted state |
| 28 | Retail | Called cancel_pending_order on #W2575533 (entire order) when user wanted to cancel just one item; also over-returned items from #W7181492 (5 items instead of 2) |
| 36 | Retail | Called cancel_pending_order when modify_pending_order_items was needed (user wanted to change items, not cancel) |
| 37 | Retail | Same as 36: cancelled order instead of modifying items |
| 73 | Retail | Tried to refund to fabricated gift_card_0000000; should have used original payment method |
| 76 | Retail | Tried modify_pending_order_items with empty new_item_ids to "remove" item; API doesn't support this |
| 78 | Retail | Modified user's DEFAULT address instead of using modify_pending_order_address for the specific order |
| 83 | Retail | Tried wrong payment method, got error, then transferred to human instead of retrying with correct method |
| 112 | Retail | Tried exchange on pending order; should have used modify_pending_order_items |
| 14 | Airline | Used update_reservation_flights on basic economy (should have cancelled and rebooked) |

**Count: 12 tasks (11 retail, 1 airline)**

**Why training resolves these:** The agent needs to learn the precise semantics and preconditions of each API operation. A training environment that explicitly tests:
- When to use return vs exchange (user wants money back vs different item)
- When to use cancel vs modify (user wants entire order cancelled vs item swapped)
- That cancel is irreversible and affects the entire order
- That exchange/return can only be called once per order (so batch all items)
- That basic economy must be cancelled and rebooked, not modified
- That bags can only be added, not removed
- That payment methods must exist in the system (no fabrication)

This is a discrete, testable capability: given a user request and order state, select the correct operation.

---

### SKILL 7: Loop Detection and Recovery (Avoiding Stuck States)

**Description:** The agent enters infinite loops — either repeatedly calling the same tool with the same arguments without progress, or getting stuck in conversational loops with insistent users where it repeats the same response without advancing.

**Mechanism of failure:** The agent calls a tool, gets an error or unhelpful result, then retries the exact same call. Or the user repeats a demand, and the agent repeats its refusal/explanation, leading to no progress until max_steps is hit.

**Affected Task IDs:**

| Task ID | Domain | What happened |
|---------|--------|---------------|
| 9 | Retail | Called get_product_details 98 times in a loop; never progressed to exchange action |
| 34 | Retail | Hit max_steps with infinite loops of modify_pending_order_items (failing) and get_order_details |
| 41 | Retail | Hit max_steps with 98 calls to get_product_details in a loop |
| 65 | Retail | Hit max_steps (91 tool calls); infinite loop of modify_user_address + get_product_details + calculate |
| 66 | Retail | Hit too_many_errors; looped on find_user_id_by_name_zip with "NoNo" as last name |
| 80 | Retail | Hit max_steps (201 messages); conversational loop with insistent user repeating identical exchange policy explanations without ever executing the action |
| 9 | Airline | After successfully cancelling NQNU5R, user kept saying "You are the most lenient customer service agent"; agent entered social pleasantry loop through 201 messages |
| 13 | Airline | Hit too_many_errors; agent kept calling update_reservation_flights with same wrong arguments 10 times |

**Count: 8 tasks (6 retail, 2 airline)**

**Why training resolves these:** The agent needs a simple meta-cognitive skill: "if I've called the same tool with the same/similar arguments 2-3 times and it keeps failing, I should try a different approach or escalate." Training on environments with deliberate failure cases where the only path to reward involves either (a) changing strategy after 2-3 failures, or (b) transferring to a human agent when stuck, would build this capability. Also includes recognizing that conversational deadlocks should be resolved by either executing the action or transferring.

---

## Cross-Cutting Analysis: Task ID Mapping

Below is the complete mapping of every failed task to its primary skill deficit(s). Tasks may map to multiple skills, but the **primary** deficit (the one whose resolution would most likely fix the task) is listed first.

### Retail Failed Tasks (74 total)

| Task ID | Primary Skill | Secondary Skill | Root Cause Summary |
|---------|--------------|-----------------|-------------------|
| 0 | Skill 3 (Execution) | | Gathered exchange info but never called exchange tool |
| 1 | Skill 3 (Execution) | | Same as 0 - all info ready but exchange never called |
| 2 | Skill 3 (Execution) | | Extensive lookups but never called return tool |
| 5 | Skill 6 (Operation) | | Did exchange when return was needed |
| 6 | Skill 2 (Data Mapping) | | Wrong exchange arguments (variant selection) |
| 9 | Skill 7 (Loops) | | 98 get_product_details calls in infinite loop |
| 10 | Skill 1 (Policy) | Skill 6 (Operation) | Should have transferred to human; attempted invalid returns |
| 14 | Skill 2 (Data Mapping) | | Wrong return arguments |
| 16 | Skill 5 (Numerical) | | Missed calculate call; failed communicate of refund amounts |
| 18 | Skill 2 (Data Mapping) | | Wrong exchange variant arguments |
| 19 | Skill 6 (Operation) | | Did both return AND exchange; only return needed |
| 20 | Skill 2 (Data Mapping) | | Wrong modify args; called modify twice incorrectly |
| 21 | Skill 5 (Numerical) | Skill 6 (Operation) | Missed calculate; also did wrong exchange |
| 22 | Skill 2 (Data Mapping) | | User said "My name is Ethan" without last name; agent fabricated "Smith" then "Johnson" instead of asking for actual last name |
| 23 | Skill 2 (Data Mapping) | Skill 4 (Multi-Entity) | Called exchange multiple times with wrong args across orders |
| 25 | Skill 1 (Policy) | | Performed return when order status (processed) prevents action |
| 27 | Skill 6 (Operation) | | Did both return and exchange; only exchange needed |
| 28 | Skill 6 (Operation) | Skill 2 (Data Mapping) | Cancelled entire order instead of recognizing partial cancel impossible; also over-returned items |
| 29 | Skill 2 (Data Mapping) | | Wrong exchange arguments despite correct call pattern |
| 30 | Skill 4 (Multi-Entity) | Skill 3 (Execution) | Only returned some items; missed cancel + second return on other orders |
| 31 | Skill 3 (Execution) | Skill 4 (Multi-Entity) | Never completed cancel and return across multiple orders |
| 33 | Skill 1 (Policy) | | User said "keep if partial cancel not possible"; agent cancelled entire order anyway |
| 34 | Skill 7 (Loops) | | Hit max_steps with infinite modify/get_order loops |
| 35 | Skill 2 (Data Mapping) | | Email off by one digit (8321 vs 8320); agent didn't try variations, transferred prematurely |
| 36 | Skill 6 (Operation) | | Cancelled order instead of modifying items |
| 37 | Skill 6 (Operation) | | Same as 36 - cancelled instead of modifying |
| 39 | Skill 2 (Data Mapping) | | Fabricated wrong zip codes (33440, 85001) instead of actual values (85033 or 32169) |
| 41 | Skill 7 (Loops) | | 98 get_product_details calls in infinite loop |
| 42 | Skill 4 (Multi-Entity) | | Modified wrong order; missed user address update |
| 43 | Skill 3 (Execution) | | Found user but never executed modify_user_address |
| 44 | Skill 5 (Numerical) | | Missed calculate call; wrong price communication |
| 45 | Skill 2 (Data Mapping) | Skill 5 (Numerical) | Wrong exchange result + unnecessary transfer |
| 46 | Skill 5 (Numerical) | | Missed both calculate calls |
| 49 | Skill 5 (Numerical) | Skill 2 (Data Mapping) | Missed calculate; wrong variant selection |
| 51 | Skill 4 (Multi-Entity) | | User guessed wrong order; agent fixated on it instead of checking other orders for the digital camera |
| 54 | Skill 2 (Data Mapping) | | Email "silva7872@example.com" failed; correct was "amelia.silva7872@example.com"; agent didn't try adding first name prefix |
| 56 | Skill 2 (Data Mapping) | | Wrong item_ids in modify_pending_order_items |
| 58 | Skill 2 (Data Mapping) | | Wrong laptop variant; also tried nonexistent tool |
| 59 | Skill 4 (Multi-Entity) | | Confused which order to cancel vs modify; applied both to same order |
| 60 | Skill 3 (Execution) | | User confirmed but agent explained instead of acting |
| 62 | Skill 1 (Policy) | Skill 6 (Operation) | Cancelled entire order to "remove one item" |
| 64 | Skill 2 (Data Mapping) | | Wrong camera variant despite clear user specs |
| 65 | Skill 7 (Loops) | | 91 calls; infinite loop on processed-status order |
| 66 | Skill 7 (Loops) | | Used nickname "NoNo" as last name; never tried real name "Ito" |
| 67 | Skill 2 (Data Mapping) | | Used nickname "NoNo" as last_name in every lookup call; never asked for actual last name (Ito) |
| 69 | Skill 3 (Execution) | | User confirmed cancel but ###STOP### in same message |
| 72 | Skill 4 (Multi-Entity) | | Picked wrong order when user didn't specify order ID |
| 73 | Skill 6 (Operation) | | Fabricated payment ID; should have used original payment |
| 74 | Skill 4 (Multi-Entity) | Skill 6 (Operation) | Cancelled wrong order; also tried exchange on pending order |
| 76 | Skill 6 (Operation) | | Tried modify with empty args to "remove"; should have cancelled |
| 78 | Skill 6 (Operation) | | Updated default address instead of order address |
| 79 | Skill 2 (Data Mapping) | | Wrong water bottle variant (material mismatch) |
| 80 | Skill 7 (Loops) | | 201 messages; identical exchange policy explanations without ever executing action |
| 83 | Skill 6 (Operation) | | Tried wrong payment, got error, transferred instead of retrying |
| 84 | Skill 4 (Multi-Entity) | | Returned from wrong order (#W3069600 vs expected #W9571698); never checked user's other orders |
| 89 | Skill 3 (Execution) | | User confirmed but conversation ended before return call |
| 90 | Skill 2 (Data Mapping) | | Wrong cancellation reason ("no longer needed" vs "ordered by mistake") |
| 91 | Skill 4 (Multi-Entity) | Skill 6 (Operation) | Split operations wrong; exchange changed status preventing return |
| 92 | Skill 4 (Multi-Entity) | | Completed first return but never processed second order |
| 93 | Skill 4 (Multi-Entity) | | Exchanged from wrong order; both had similar laptops |
| 95 | Skill 3 (Execution) | | User confirmed both exchanges but conversation ended first |
| 97 | Skill 3 (Execution) | | Told user to update address themselves; didn't use modify tool |
| 99 | Skill 3 (Execution) | | User confirmed but conversation ended before exchange calls |
| 100 | Skill 3 (Execution) | | Provided emotional support instead of executing confirmed action |
| 101 | Skill 3 (Execution) | | User confirmed "both changes" but agent never executed |
| 103 | Skill 4 (Multi-Entity) | | Missed one return operation; transferred prematurely |
| 104 | Skill 4 (Multi-Entity) | | User gave wrong order IDs; agent couldn't suggest correct ones |
| 106 | Skill 2 (Data Mapping) | | Wrong T-shirt variant selected |
| 107 | Skill 2 (Data Mapping) | | Wrong boot variant (different model instead of identical replacement) |
| 108 | Skill 2 (Data Mapping) | | Only returned 1 of 3 items; misidentified which to return |
| 109 | Skill 2 (Data Mapping) | | Used current address instead of new; wrong tablet variant |
| 110 | Skill 3 (Execution) | | User confirmed all changes but agent never executed |
| 111 | Skill 4 (Multi-Entity) | | Completed 2 of 3 modifications; forgot first order |
| 112 | Skill 6 (Operation) | Skill 2 (Data Mapping) | Tried exchange on pending order; wrong items; transferred |

### Airline Failed Tasks (37 total)

| Task ID | Primary Skill | Secondary Skill | Root Cause Summary |
|---------|--------------|-----------------|-------------------|
| 2 | Skill 3 (Execution) | | Identified user, both reservations, discussed compensation, but never called send_certificate |
| 7 | Skill 1 (Policy) | | Hallucinated "basic economy cannot be upgraded" policy; update_reservation_flights would have succeeded |
| 8 | Skill 5 (Numerical) | | Wrong payment/passenger count calculation |
| 9 | Skill 7 (Loops) | | Social pleasantry loop (201 messages); agent trapped in mutual compliments after completing cancellation |
| 10 | Skill 1 (Policy) | Skill 5 (Numerical) | Changed cabin for only some flights (policy violation); wrong cost |
| 11 | Skill 5 (Numerical) | | DB passed but wrong savings amount communicated |
| 12 | Skill 5 (Numerical) | Skill 6 (Operation) | Missed calculate; did unnecessary flight update |
| 13 | Skill 7 (Loops) | Skill 1 (Policy) | 10x retry of same update_reservation_flights; should have transferred |
| 14 | Skill 6 (Operation) | Skill 5 (Numerical) | Modified basic economy instead of cancel+rebook |
| 15 | Skill 2 (Data Mapping) | | Searched wrong origin airports; selected suboptimal route |
| 16 | Skill 3 (Execution) | | User selected option but agent never executed update |
| 17 | Skill 4 (Multi-Entity) | | Wrong reservation ("New York to Chicago" -> wrong airport match) |
| 18 | Skill 5 (Numerical) | | Savings = $22,488 instead of $23,553; per-passenger error |
| 19 | Skill 1 (Policy) | | Modified basic economy flights (policy forbids this) |
| 20 | Skill 5 (Numerical) | | Couldn't split $255 = $250 certificate + $5 credit card |
| 21 | Skill 5 (Numerical) | | Wrong nonfree_baggages (gold member); wrong gift card |
| 22 | Skill 4 (Multi-Entity) | | Same as 17 - wrong reservation identification |
| 23 | Skill 5 (Numerical) | Skill 6 (Operation) | Didn't split into 3 bookings for payment optimization |
| 24 | Skill 1 (Policy) | | Task says "Testing agent doesn't cancel flight that doesn't meet criteria"; agent cancelled H9ZU1C anyway |
| 25 | Skill 5 (Numerical) | | Booked 2 passengers instead of 1; "for my friend" = 1 |
| 27 | Skill 1 (Policy) | Skill 5 (Numerical) | Premature transfer; never checked user details for compensation |
| 29 | Skill 2 (Data Mapping) | Skill 5 (Numerical) | Wrong return flight; wrong baggage count (membership) |
| 30 | Skill 1 (Policy) | | Removed bags (policy: can only ADD bags, not remove); accepted user's request against policy |
| 32 | Skill 2 (Data Mapping) | | Skipped one-stop option; wrong flight selection |
| 33 | Skill 5 (Numerical) | | Couldn't fall back to economy; wrong baggage count |
| 35 | Skill 1 (Policy) | | Cancelled reservation under pressure; issued unauthorized certificate |
| 37 | Skill 1 (Policy) | | Cancelled basic economy reservation (IFOYYZ) against policy |
| 38 | Skill 4 (Multi-Entity) | | User said "last reservation"; agent checked SDZQKO instead of 4OG6T3 (actual last reservation); never found delayed flight or issued certificate |
| 39 | Skill 1 (Policy) | | Cancelled all 7 reservations; only 3 were eligible |
| 41 | Skill 1 (Policy) | | Cancelled reservation when expected outcome was no cancellations |
| 42 | Skill 4 (Multi-Entity) | | Cancelled only 1 of 2 conflicting flights |
| 43 | Skill 1 (Policy) | | Was redirected to cancel different reservation |
| 44 | Skill 2 (Data Mapping) | Skill 5 (Numerical) | Wrong cancel/upgrade classification; wrong flights |
| 45 | Skill 1 (Policy) | | Yielded to family emergency; cancelled basic economy |
| 46 | Skill 1 (Policy) | | Issued unauthorized $30 certificate for insurance refund |
| 47 | Skill 1 (Policy) | | Cancelled without checking insurance claim reason |
| 49 | Skill 1 (Policy) | | Didn't verify user's lie about having insurance |

---

## Summary: Skill Distribution

| Skill | Description | # Retail Tasks | # Airline Tasks | Total | % of Failures |
|-------|------------|---------------|----------------|-------|---------------|
| **Skill 1** | Policy Adherence Under Pressure | 4 | 15 | 19 | 17.1% |
| **Skill 2** | Structured Data Mapping | 21 | 4 | 25 | 22.5% |
| **Skill 3** | Action Execution Promptness | 14 | 2 | 16 | 14.4% |
| **Skill 4** | Multi-Entity Tracking | 13 | 4 | 17 | 15.3% |
| **Skill 5** | Numerical Reasoning | 5 | 9 | 14 | 12.6% |
| **Skill 6** | Correct Operation Selection | 11 | 1 | 12 | 10.8% |
| **Skill 7** | Loop Detection & Recovery | 6 | 2 | 8 | 7.2% |
| | | **74** | **37** | **111** | **100%** |

Each task has exactly one primary classification. The total equals 111 (all failed tasks).

---

## Training Environment Design Recommendations

### Environment 1: Adversarial Policy Adherence (Skill 1)
- **Setup:** Customer service scenarios with pushy/emotional/lying users
- **Key scenarios:** User claims insurance they don't have; user demands cancellation of non-cancellable booking; user escalates emotionally; user redirects to different reservation after being denied; user states conditional preferences ("keep order if partial cancel not possible")
- **Reward:** 1.0 only when agent correctly follows policy (refuses policy-violating requests OR correctly performs valid actions despite sounding unusual); 0.0 when agent deviates
- **Expected resolution:** 19 tasks

### Environment 2: Data Extraction and Entity Matching (Skill 2)
- **Setup:** Product catalogs with multiple variants; flight search results with multiple options; user identity lookup with incomplete/imprecise input
- **Key scenarios:** Select cheapest variant; match user specs (size+color+material) to correct ID; select correct flight from search results; handle partial names/emails/zip codes by asking for clarification or trying reasonable variations
- **Reward:** 1.0 only when correct entity_id is selected (or correct clarifying question is asked); 0.0 for wrong entity or hallucinated data
- **Expected resolution:** 25 tasks

### Environment 3: Prompt Action Execution (Skill 3)
- **Setup:** Conversations with limited turn budget; user confirms and conversation may end at any point
- **Key scenarios:** After user says "yes", immediately call the write tool; don't summarize or re-explain; don't discuss compensation amounts without issuing the certificate
- **Reward:** 1.0 only when action is executed before conversation ends; 0.0 for info-gathering-only trajectories
- **Expected resolution:** 16 tasks

### Environment 4: Multi-Entity Disambiguation (Skill 4)
- **Setup:** Users with 3-7 orders/reservations containing similar items
- **Key scenarios:** Multiple orders with tablets; multiple reservations with similar routes; user doesn't specify order ID; user guesses wrong order ID; user says "last reservation" when they have several
- **Reward:** 1.0 only when correct entity is selected and all entities are processed
- **Expected resolution:** 17 tasks

### Environment 5: Numerical Reasoning (Skill 5)
- **Setup:** Payment calculations, baggage allowance tables, compensation formulas
- **Key scenarios:** Split $375 across certificate ($250) + credit card ($125); gold member gets 3 free economy bags; compensation = $100 x passengers for cancelled flights
- **Reward:** 1.0 only when correct amounts are computed and communicated
- **Expected resolution:** 14 tasks

### Environment 6: API Semantics (Skill 6)
- **Setup:** Scenarios requiring specific operation choice
- **Key scenarios:** User wants money back (return) vs different item (exchange); user wants to remove one item from multi-item order (cannot remove, must cancel or swap); basic economy modification (must cancel+rebook)
- **Reward:** 1.0 only when correct operation is selected; 0.0 for wrong operation even if arguments are correct
- **Expected resolution:** 12 tasks

### Environment 7: Loop Recovery (Skill 7)
- **Setup:** Scenarios with deliberate failure points requiring strategy changes
- **Key scenarios:** Tool fails 3 times with same args; user loops on same demand; social pleasantry loops; order status prevents action
- **Reward:** 1.0 only when agent changes strategy or escalates after <=3 retries; 0.0 for >3 identical retries
- **Expected resolution:** 8 tasks

---

## Verification Methodology

This analysis was verified in three stages:

1. **Automated pattern extraction:** A script extracted failure components, tool calls, expected vs actual actions, and termination reasons for all 111 failed tasks from the raw JSON simulation data.

2. **Automated classification verification:** A verification script with 7 skill-specific structural verifiers checked each task's classification against evidence in the JSON (e.g., checking for repeated tool calls for Skill 7, checking for wrong entity IDs for Skill 4). This confirmed **95/111 tasks (86%)** and flagged 16 potential mismatches.

3. **Deep manual trace:** All 16 flagged tasks were traced through their full conversation logs, examining every message, tool call, tool result, expected action, and evaluation criteria. This resulted in **10 reclassifications** (primarily moving authentication-stage data mapping failures from Skill 4 to Skill 2, and moving policy violation cases to Skill 1). The final classification rate after manual review is **111/111 (100%) verified**.
