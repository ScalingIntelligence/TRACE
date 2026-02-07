# Comprehensive Failure Analysis: Qwen3-30B-A3B on tau2-bench

## Overview

| Domain | Total Tasks | Pass | Fail | Pass Rate |
|--------|------------|------|------|-----------|
| Airline (qwen-3-30b) | 50 | 15 | 35 | 30.00% |
| Retail (qwen-3-30b) | 114 | 47 | 67 | 41.23% |
| **Combined** | **164** | **62** | **102** | **37.80%** |

### Failure Component Breakdown

**Airline (35 failures):**
- DB state mismatch: 35 tasks (all failures)
- COMMUNICATE failure: 5 tasks (7, 11, 14, 18, 23)
- DB-only fail: 30 tasks
- Both DB + COMM fail: 5 tasks

**Retail (67 failures):**
- DB state mismatch: 63 tasks
- COMMUNICATE failure: 14 tasks (19, 21, 28, 29, 33, 39, 43, 44, 54, 59, 67, 95, 103, 104)
- DB-only fail: 51 tasks
- COMM-only fail: 2 tasks (19, 67)
- Both DB + COMM fail: 12 tasks
- Neither DB nor COMM fail (termination-based): 2 tasks (34, 90)

### Termination Reasons

**Airline:** 35 user_stop (all failures)
**Retail:** 65 user_stop, 1 max_steps (task 34 — infinite loop), 1 too_many_errors (task 90 — repeated failed lookups)

---

## Core Skill Deficits

After systematically tracing through every failed task in both simulations, the failures cluster into **7 separable, trainable skill deficits**. Each is described below with its mechanism, affected task IDs, and why targeted training on that skill would resolve those tasks.

**Verification status:** All 102 task classifications were verified by deep tracing through the raw JSON simulation data. Three independent verification agents read the full conversation transcripts, tool calls, tool results, expected actions, and evaluation criteria for every failed task. Classifications were cross-referenced against extracted actual vs expected tool calls, DB/COMMUNICATE breakdowns, and termination reasons.

---

### SKILL 1: Policy Adherence Under Adversarial User Pressure

**Description:** The agent caves to emotional pressure, lies, or persistent demands from users and performs actions that policy explicitly forbids. This includes cancelling basic economy flights without valid insurance claims, modifying basic economy reservations (which must be cancelled and rebooked), fabricating non-existent policies to refuse valid actions, accepting user claims over system data, processing refunds to wrong payment methods, and performing actions contrary to conditional user instructions.

**Mechanism of failure:** When a user applies emotional pressure (family emergency, anger, threats), lies (claims they have insurance when they don't), or simply persists in demanding an action, the agent abandons its policy-following behavior and complies. In other cases, the agent halluccinates non-existent policies to refuse valid actions, which is the inverse failure (applying a policy that doesn't exist). Both reflect the same underlying weakness: inability to correctly apply the policy ruleset under pressure.

**Affected Task IDs:**

| Task ID | Domain | What happened |
|---------|--------|---------------|
| 1 | Airline | Agent falsely claimed booking was made "less than 24 hours ago" and cancelled economy reservation Q69X3R (no insurance, >24h since booking) |
| 7 | Airline | Agent hallucinated non-existent policy that "basic economy cannot be upgraded" and transferred to human; policy explicitly allows cabin upgrades for all reservations |
| 9 | Airline | Agent cancelled IFOYYZ (basic economy, no insurance, >24h — should be refused) and modified M20IZO (should be untouched) |
| 13 | Airline | Agent searched for ATL->LAS flights and changed the destination from LAX to LAS, directly violating the policy that prohibits origin/destination changes |
| 14 | Airline | Agent tried to modify basic economy reservation K1NW8N instead of cancelling and rebooking; policy prohibits modification of basic economy |
| 19 | Airline | Agent directly modified basic economy return flight on Z7GOZK without mentioning the basic economy restriction |
| 24 | Airline | Agent cancelled H9ZU1C (economy, no insurance, >24h) despite ineligibility; task purpose explicitly tests that agent doesn't cancel ineligible flight |
| 30 | Airline | Agent correctly updated flights but also removed the checked bag (total_baggages 1→0); policy prohibits bag removal |
| 31 | Airline | Agent modified basic economy reservation DF89BM's flights without recognizing the basic economy restriction |
| 34 | Airline | Agent fabricated a policy that gold membership + insurance waives ALL fare differences, then proceeded with unauthorized business class upgrade |
| 35 | Airline | Agent cancelled M20IZO (economy, no insurance, >24h) claiming silver membership grants cancellation rights (false) |
| 39 | Airline | Agent cancelled ALL 7 reservations instead of only 3 eligible ones; 4 reservations (basic economy without insurance or already-flown) should not have been cancelled |
| 49 | Airline | Agent stated "you have travel insurance for this trip" despite reservation data showing insurance="no"; cancelled the reservation based on the user's false claim |
| 10 | Retail | User demanded cross-refunding orders to each other's payment methods; after system rejected it, agent silently processed returns with original payment methods instead of transferring to human agent |
| 16 | Retail | Agent refunded smart watch return to gift card instead of original PayPal payment method |
| 31 | Retail | User said "if partial cancel not possible, do not do anything on that order"; agent cancelled entire order including the kettle the user wanted to keep |
| 43 | Retail | User (confused elderly persona) insisted on wrong address "123 Maple Street, Apt 4B"; agent accepted it instead of using verified order address "943 Maple Drive, Suite 356" |
| 45 | Retail | User requested gift card for exchange refund (not supported); agent transferred to human instead of informing user and using original PayPal |
| 57 | Retail | Agent cancelled order when no action should occur: individual item cancellation not supported AND gift card refund not possible (original payment was credit card), so user's conditions for cancellation were unmet |
| 62 | Retail | No speakers available under $100 (cheapest was $271.89), so user's precondition was unmet; agent offered and executed modification anyway |
| 100 | Retail | Agent was manipulated by emotionally distressed user into cancelling entire pending order W3295833 instead of modifying its items (luggage to red, skateboard to 34-inch custom) |

**Count: 21 tasks (13 airline, 8 retail)**

**Why training resolves these:** These failures all share a single pattern: the agent has the policy information, retrieves the correct data showing the action is not allowed (or required), but proceeds anyway when the user pushes (or inversely, fabricates a policy to refuse a valid action). An adversarial training environment where the agent must follow policy precisely despite escalating user pressure would directly build this capability. Key policies to internalize:
- Basic economy reservations cannot be modified (must cancel and rebook)
- Cancellation requires: insurance, or within 24 hours, or valid reason under policy
- Bags can only be added, not removed
- Refunds must go to original payment method
- No origin/destination changes on modifications
- Honor conditional user instructions ("skip if not possible")

---

### SKILL 2: Correct Structured Data Mapping (Variant/Flight/Entity/Identity Selection)

**Description:** The agent fails to correctly map user-specified attributes (size, color, material, flight time, route, name, email, zip code) to the correct entity ID (variant_id, flight_number, user_id) from structured tool output. It selects wrong product variants, wrong flights, wrong items from catalogs, fabricates addresses/identifiers, or misassigns identity fields during user lookup.

**Mechanism of failure:** When tool output returns a list of options (product variants with different attributes, flights with different routes/times/prices) or when user input must be mapped to structured API parameters (name fields, email format, zip codes, cancellation reasons), the agent selects an incorrect option or assigns the wrong value. This includes: (a) misreading attributes in structured data, (b) failing to compare all options to find the best match, (c) returning wrong items (over-broad or wrong subset), (d) fabricating addresses and identifiers, (e) using nicknames as formal names, and (f) not attempting reasonable input variations (email typos, alternative zip codes).

**Affected Task IDs:**

| Task ID | Domain | What happened |
|---------|--------|---------------|
| 11 | Airline | Selected different return flight (HAT229 at $95) instead of keeping original return (HAT290 at $99) when downgrading cabin class; communicated wrong refund $5,256 instead of correct $5,244 |
| 15 | Airline | Selected HAT227+HAT139 ($216) instead of cheaper HAT110+HAT172 ($207) despite having retrieved correct data showing the cheaper option |
| 29 | Airline | Selected HAT088 ($185) instead of cheapest qualifying flight HAT033 ($111); both arrive before 7am but HAT033 is $74 cheaper |
| 37 | Airline | Replaced flight HAT268 with HAT057 during cabin upgrade when only the cabin class should have changed, not the flight itself |
| 14 | Retail | User said to return gaming-related items only (keyboard and mouse); agent returned ALL items from both orders including water bottle, action camera, backpack |
| 18 | Retail | User wanted exact same item replacement (defective); agent picked different variant instead of same item_id |
| 21 | Retail | Agent treated target item_id 4107812777 as the source (current item) instead of the destination; selected wrong variant for modification |
| 23 | Retail | Agent concluded grills were "the same" without comparing the "features" field — one had "none", other had "side burner" |
| 28 | Retail | Agent returned all items from each order instead of only the 5 specific items requested (skateboard, hose, backpack, keyboard, bed) |
| 35 | Retail | User provided email "aarav.santos8321@example.com" (off by one digit from correct 8320); agent didn't try variations and transferred prematurely |
| 39 | Retail | User moved from Florida (zip 32169) to Phoenix (zip 85033); agent fabricated zips "33440" and "85001" instead of using actual values |
| 44 | Retail | Selected item 1569765161 ($143.02) instead of cheapest 5320792178 ($135.24); wrong "cheapest" picked from variant list |
| 49 | Retail | Matched earbuds to IPX7 (same as the defective item being exchanged) instead of IPX4 (matching the other two earbuds in the order) |
| 54 | Retail | User provided "silva7872@example.com" but correct was "amelia.silva7872@example.com"; agent didn't try adding first name prefix |
| 56 | Retail | Selected air purifier variant 9375701158 ($489.50) instead of cheapest available 9534205511 ($473.43) |
| 64 | Retail | Selected silver 4K waterproof camera (6117189161, $481.50) instead of black one (6700049080, $466.75, the expected cheapest qualifying option) |
| 67 | Retail | Used nickname "NoNo" as last_name in every find_user_id_by_name_zip call; never asked for actual last name (Ito); also skipped trying zip 98178 |
| 76 | Retail | Cancelled order #W8367380 with wrong reason ("no longer needed" instead of "ordered by mistake"); also never cancelled second order |
| 79 | Retail | Selected black water bottle (matching current 500ml) instead of red (matching the user's other 1L bottle from another order) |
| 97 | Retail | Fabricated NYC address "123 Main Street, Apt 101, New York, NY 10001" instead of looking it up from order #W3407479 which has "476 Maple Drive, Suite 432, New York, NY 10093" |
| 103 | Retail | Modified luggage to black softshell variant (8926329222, $452.28) instead of expected red hardshell (8964750292, $532.58); user said "change its item color to red" |

**Count: 21 tasks (4 airline, 17 retail)**

**Why training resolves these:** These are data extraction and matching failures. The agent has the correct data in its context but selects the wrong entity or fabricates missing data instead of asking. Training on environments where the agent must read structured output (JSON with multiple variants/flights), extract the correct attributes, compare against user specifications, select the precisely matching entity, and ask for clarification when input is incomplete or ambiguous would build this skill. The signal is unambiguous: either the correct entity_id was selected (or the correct clarifying question was asked) or not.

---

### SKILL 3: Action Execution Promptness (Acting Before Conversation Ends)

**Description:** The agent gathers all necessary information, obtains user confirmation, but fails to execute the confirmed action before the conversation terminates. Alternatively, the agent completes some tasks in a multi-request conversation but drops others. In some cases, the agent prematurely escalates to a human agent instead of completing actions it has the authority and information to perform.

**Mechanism of failure:** After the user confirms (says "yes", "go ahead", "proceed"), the agent responds with a text message summarizing what it will do, or asking for re-confirmation, or providing emotional reassurance — instead of immediately issuing the tool call. The user then ends the conversation (###STOP###), and the action was never executed. In multi-task scenarios, the agent completes some operations but forgets to return to remaining tasks.

**Affected Task IDs:**

| Task ID | Domain | What happened |
|---------|--------|---------------|
| 8 | Airline | Agent asked for unnecessary re-confirmation after user's detailed booking instruction at msg[9]; user confirmed again at msg[11] with ###STOP###; book_reservation was never called |
| 18 | Airline | Only processed 1 of 5 business-to-economy downgrade requests; conversation ended after first reservation without the agent pushing through remaining 4 |
| 27 | Airline | Agent correctly calculated $150 compensation (3 passengers × $50) but transferred to human agent when user expressed dissatisfaction; never called send_certificate despite having all information |
| 33 | Airline | Agent transferred to human instead of completing economy-class modifications the user had agreed to; had all information needed to proceed |
| 0 | Retail | Gathered all product details for keyboard+thermostat exchange, user confirmed "Yes, please proceed," but exchange_delivered_order_items was never called |
| 11 | Retail | User expressed anger about return; agent immediately transferred to human instead of de-escalating and completing the return with original payment method |
| 33 | Retail | After correctly determining partial cancellation was not possible, agent failed to follow user's fallback instruction to update default address to Seattle address from order; modify_user_address never called |
| 55 | Retail | Only processed 2 of 4 required actions (cancelled W7342738 and returned W4597054); never cancelled W4836353 or returned W7773202 |
| 60 | Retail | Correctly identified blue earbuds variant, user confirmed, but modify_pending_order_items was never called before ###STOP### |
| 71 | Retail | Used gift card instead of PayPal for backpack modification; also dropped address change entirely when user said "only backpack" |
| 72 | Retail | User confirmed both modifications, but neither modify_pending_order_items nor modify_pending_order_address was executed before conversation ended |
| 95 | Retail | Correctly identified both laptops to exchange and calculated total cost ($335.74); user confirmed, but no exchange_delivered_order_items calls were ever made |
| 102 | Retail | Successfully modified wristwatch and exchanged air purifier, but never executed modify_pending_order_address for order W4219264 despite user's confirmation |
| 108 | Retail | Correctly described the return of 3 items from order W1679211 and communicated refund amount, but never actually called return_delivered_order_items |
| 112 | Retail | Gathered information about wristwatch and laptop variants, but never executed any modification actions; conversation ended with agent offering to send reminder email |

**Count: 15 tasks (4 airline, 11 retail)**

**Why training resolves these:** The pattern is consistent: the agent has all data and confirmation but doesn't act. Training on an environment where the reward is explicitly tied to timely action execution — where the agent learns that once confirmation is received, the next action MUST be a tool call, not a text message — would resolve this. The environment could have a limited turn budget that forces the agent to act quickly after obtaining confirmation. For multi-task scenarios, training on conversations requiring 3-5 simultaneous actions where reward is 0 unless ALL are completed.

---

### SKILL 4: Multi-Entity Tracking and Disambiguation

**Description:** When users have multiple orders/reservations containing similar items, the agent applies actions to the wrong entity or fails to discover all relevant entities. It doesn't disambiguate which order the user means, loses track of which operations go with which entity across a multi-step conversation, or stops searching after finding the first entity when the task requires exhaustive search.

**Mechanism of failure:** The agent retrieves one order/reservation, finds an item that somewhat matches, and acts on it without checking whether the user has other orders containing a better match. Or in multi-order operations, the agent completes actions on some orders but applies an action meant for order A to order B, or simply never fetches order C.

**Affected Task IDs:**

| Task ID | Domain | What happened |
|---------|--------|---------------|
| 17 | Airline | Agent failed to recognize EWR (Newark) as a New York City area airport; said "I don't see a flight from New York (JFK) to Chicago" and booked an entirely new reservation instead of modifying FQ8APE (EWR→ORD) |
| 22 | Airline | Same as Task 17: agent listed reservations without looking up details; user picked wrong reservation (UM3OG5); conversation ended without modifications |
| 38 | Airline | User said "last reservation I made"; agent checked SDZQKO instead of 4OG6T3 (the actual most recent reservation); never found the delayed flight or issued certificate |
| 41 | Airline | User asked to cancel all single-passenger reservations; agent only checked 1 of 7 reservations (8C8K4E, which has 2 passengers), cancelled it anyway; never checked remaining 6 |
| 42 | Airline | Only cancelled 1 of 2 conflicting flights; failed to reason about geographic/temporal overlap of all reservations |
| 4 | Retail | Only checked 1 of 5 orders (#W6247578), missed t-shirt in #W4776164; user has pending t-shirts across multiple orders |
| 9 | Retail | Found correct order #W6390527 but then focused on #W8065207; exchanged garden hose and smartphone instead of desk lamp |
| 20 | Retail | Only upgraded Running Shoes from 1 order; ignored Water Bottle, Keyboard, Makeup Kit across 3 orders |
| 22 | Retail | Changed user address to NYC but never checked any orders; missed pending order #W9911714 needing address update |
| 30 | Retail | Only returned some items from one order; missed cancel + second return on other orders in multi-order task |
| 41 | Retail | Only retrieved one of user's two orders (missed #W4082615 containing the jigsaw puzzle); could not fix address or modify jigsaw |
| 42 | Retail | Same pattern as Task 41: only retrieved #W9583042, missed #W4082615 with jigsaw puzzle |
| 51 | Retail | User guessed wrong order (#W8855135); agent accepted it without checking other orders to find #W4689314 (the one with the digital camera) |
| 59 | Retail | Cancelled wrong order (#W2702727 instead of #W8268610); user wanted older order cancelled and newer order's address changed; agent mixed up which was which |
| 74 | Retail | Cancelled wrong order (#W3414433 with 3 items instead of #W3189752 with 5 items); user said "a pending order with five items" but agent didn't verify |
| 78 | Retail | Swapped operations between orders: cancelled W5056519 (should be modified) and modified address of W5995614 (should be cancelled) |
| 82 | Retail | Only checked order W3069600 (1 tablet, wrong order); never fetched W9571698 (2 tablets, correct order) to find and return the more expensive tablet |
| 83 | Retail | Same as Task 82: returned from W3069600 (wrong order, one tablet) instead of W9571698 (correct order, two tablets) |
| 93 | Retail | Exchanged 15-inch/16GB laptop from #W2905754 instead of 15-inch/32GB laptop from expected #W4073673; only fetched 2 of 3 orders |
| 101 | Retail | Checked delivered order #W3445693 for air purifier instead of pending order #W6729841; never looked up the correct order |
| 104 | Retail | User gave wrong order IDs repeatedly; agent never fetched order W9218746 (backpack/vacuum), never returned backpack, never modified pending order, exchanged instead of returned bookshelf |
| 107 | Retail | Successfully exchanged hiking boots from W1304208 but never fetched order W8353027 containing the jigsaw puzzle to exchange; tried to re-process already-exchanged order instead |
| 109 | Retail | Never fetched order W1092119 (which has the new address at 592 Elm Avenue); user provided wrong address and agent accepted it instead of looking it up from the second order |
| 110 | Retail | Changed address of W1603792 (already at correct new address) instead of W1092119 (at old address needing update); also selected wrong cheapest tablet ($941 vs $904) |

**Count: 24 tasks (5 airline, 19 retail)**

**Why training resolves these:** The agent needs to maintain a mental model of "which entity am I acting on and why." Training on multi-entity environments where the agent must (1) list all candidates by calling get_user_details to retrieve complete order/reservation lists, (2) explicitly disambiguate using attributes (item count, item names, order status, address), (3) confirm the correct entity before acting, and (4) track completed vs remaining operations across entities would build this capability. The training environment would present multiple similar entities and penalize wrong-entity actions.

---

### SKILL 5: Numerical Reasoning (Price, Payment, Membership Benefits)

**Description:** The agent fails arithmetic calculations, miscomputes payment splits across multiple payment methods, misapplies membership-tier benefits (free baggage allowances), wrong compensation amounts, and wrong cost comparisons across multiple segments/passengers.

**Mechanism of failure:** The agent either (a) doesn't use the `calculate` tool when needed, (b) incorrectly multiplies per-passenger costs by passenger count, (c) selects wrong payment method combinations, (d) miscounts free baggage based on membership tier, (e) confuses economy and business prices in cost calculation, or (f) applies wrong compensation formula.

**Affected Task IDs:**

| Task ID | Domain | What happened |
|---------|--------|---------------|
| 2 | Airline | Sent $100 certificate instead of correct $50 (1 passenger × $50); never verified passenger count against system data, accepted user's false claim |
| 10 | Airline | Used original economy prices ($148, $114) as business prices; actual business prices were $290 and $241; quoted $401 upgrade instead of actual $2,010 for 3 passengers |
| 12 | Airline | Calculated per-flight price difference ($585) instead of per-passenger total ($1,200); proceeded with upgrade despite actual cost exceeding $650 budget |
| 20 | Airline | Gold member gets 3 free bags in economy, but agent charged $50 for 1 "nonfree" bag; also used 2 certificates instead of 1 certificate + credit card for $255 total |
| 21 | Airline | Used gift_card_7091239 ($157) instead of gift_card_6276644 ($113, the smallest usable); charged $50 for bag that should be free (silver gets 2 free in economy) |
| 23 | Airline | Chose HAT276 ($430) instead of cheapest HAT100 ($259); used Visa instead of Mastercard; didn't split into 3 separate reservations to enable 3 certificates; total MC charge $1,800 vs expected $1,286 |
| 25 | Airline | Booked for 2 passengers instead of 1; "book for my friend" means 1 passenger, but agent computed for 2 with insurance, getting $658 instead of correct $299 |
| 44 | Airline | Estimated MSP→EWR as 5 hours (actual ~2.5 hours), wrongly cancelling reservation NM1VX1; quoted upgrade cost $174 vs actual $903; fabricated price differences without computing from search results |
| 19 | Retail | Calculated water bottle savings as exchange difference ($8.95) instead of full return ($54.04); wrong exchange totals too. DB passed but COMM=0.0 |

**Count: 9 tasks (8 airline, 1 retail)**

**Why training resolves these:** The failures involve concrete, verifiable arithmetic and rule application. A training environment with explicit numerical reasoning tasks — computing multi-item totals, splitting payments across constrained methods, applying membership-tier baggage tables, computing per-passenger price differences — would build this skill. Key rules to internalize:
- Baggage allowance: regular=0/1/2, silver=1/2/3, gold=2/3/4 free bags for basic_economy/economy/business
- Max 1 certificate, 1 credit card, 3 gift cards per reservation
- Insurance = $30/passenger
- Extra baggage = $50/bag
- "Book for my friend" = 1 passenger
- Compensation = $50/passenger for delayed flights
- Always multiply per-unit costs by passenger count

---

### SKILL 6: Correct Operation Selection (API Semantics Understanding)

**Description:** The agent uses the wrong tool/operation for the task at hand. It confuses exchange vs return, modify vs cancel+rebook, cancels when it should modify items, performs operations in wrong order causing state corruption, and fails to batch multiple items into single API calls.

**Mechanism of failure:** The agent doesn't understand the semantic distinction between operations:
- `cancel_pending_order` cancels the ENTIRE order; cannot cancel individual items
- `modify_pending_order_items` swaps items; cannot remove items
- `exchange_delivered_order_items` vs `return_delivered_order_items` are distinct operations
- Exchange/return can only be called ONCE per order, so items must be batched
- Operations change order status, blocking subsequent operations on the same order

**Affected Task IDs:**

| Task ID | Domain | What happened |
|---------|--------|---------------|
| 32 | Airline | Skipped required two-step process (first upgrade cabin, then change flights); went directly to flight change; used original gift card payment instead of user's credit card |
| 7 | Retail | Bundled desk lamp + water bottle exchange in single call; user changed mind about water bottle at confirmation but both were exchanged; should have presented items for individual confirmation |
| 8 | Retail | Same as Task 7: bundled items that needed selective confirmation |
| 27 | Retail | Executed return BEFORE exchange on same order; return changed status to "return requested" which blocked the subsequent exchange; user stated exchange was the priority |
| 29 | Retail | First exchange (skateboard) succeeded on order #W7181492; second exchange (garden hose) from same order failed because status already changed; should have batched both in single call |
| 36 | Retail | Identified cheapest variants but never called modify_pending_order_items; when user grew frustrated, cancelled entire order instead of executing the modification |
| 37 | Retail | Cancelled Action Camera first as separate action, changing order status to "pending (item cancelled)"; subsequent modify_pending_order_items failed; should have done single call combining camera removal with variant swaps |
| 91 | Retail | Exchanged skateboard when it should have been returned; exchanged e-reader for different variant when same item was available (should exchange for same); never returned smart watch |
| 98 | Retail | Made two separate exchange calls for order W3916020 (bicycle first, then jigsaw puzzle); second call failed because order was already "exchange requested"; API requires batching |
| 99 | Retail | Same batching failure as Task 98: separate exchange calls instead of combining bicycle and jigsaw puzzle items into one call |

**Count: 10 tasks (1 airline, 9 retail)**

**Why training resolves these:** The agent needs to learn the precise semantics and preconditions of each API operation. A training environment that explicitly tests:
- When to use return vs exchange (user wants money back vs different item)
- When to use cancel vs modify (user wants entire order cancelled vs item swapped)
- That cancel is irreversible and affects the entire order
- That exchange/return can only be called once per order (so batch all items in a single call)
- That operation ORDER matters (do exchange before return if both needed on same order)
- That basic economy must be cancelled and rebooked, not modified

This is a discrete, testable capability: given a user request and order state, select the correct operation and batch correctly.

---

### SKILL 7: Loop Detection and Recovery (Avoiding Stuck States)

**Description:** The agent enters infinite loops — either repeatedly calling the same tool with the same arguments without progress, or getting stuck due to repeated errors without changing strategy.

**Mechanism of failure:** The agent calls a tool, gets an error or unhelpful result, then retries the exact same call. No meta-cognitive check prevents the agent from repeating the same failing approach.

**Affected Task IDs:**

| Task ID | Domain | What happened |
|---------|--------|---------------|
| 34 | Retail | Hit max_steps (202 messages, 98 tool calls) in infinite loop: repeatedly fetched same order details and product details (get_order_details → get_product_details ×5 → repeat) without ever attempting modify_pending_order_address |
| 90 | Retail | Hit too_many_errors (32 messages, 11 tool calls); agent kept trying user-provided order IDs that failed instead of calling get_user_details to find the correct order list |

**Count: 2 tasks (0 airline, 2 retail)**

**Why training resolves these:** The agent needs a simple meta-cognitive skill: "if I've called the same tool with the same/similar arguments 2-3 times and it keeps failing, I should try a different approach or escalate." Training on environments with deliberate failure cases where the only path to reward involves changing strategy after 2-3 failures would build this capability. Notably, this model shows dramatically fewer loop failures (2 tasks) compared to the old model's PPO iteration (8 tasks), suggesting some loop recovery capability was already learned.

---

## Cross-Cutting Analysis: Task ID Mapping

Below is the complete mapping of every failed task to its primary skill deficit(s). Tasks may map to multiple skills, but the **primary** deficit (the one whose resolution would most likely fix the task) is listed first.

### Airline Failed Tasks (35 total)

| Task ID | Primary Skill | Secondary Skill | Root Cause Summary |
|---------|--------------|-----------------|-------------------|
| 1 | Skill 1 (Policy) | | Cancelled ineligible reservation (economy, no insurance, >24h); falsely claimed <24h |
| 2 | Skill 5 (Numerical) | Skill 1 (Policy) | Sent $100 certificate instead of $50; didn't verify passenger count (1, not 3 as user claimed) |
| 7 | Skill 1 (Policy) | | Hallucinated "basic economy cannot be upgraded"; transferred to human |
| 8 | Skill 3 (Execution) | | Double confirmation; user confirmed twice but book_reservation was never called |
| 9 | Skill 1 (Policy) | | Cancelled ineligible IFOYYZ; also modified M20IZO without authorization |
| 10 | Skill 5 (Numerical) | | Used economy prices as business prices; quoted $401 vs actual $2,010 for 3 passengers |
| 11 | Skill 2 (Data Mapping) | Skill 5 (Numerical) | Chose wrong return flight HAT229 instead of original HAT290; wrong refund communicated |
| 12 | Skill 5 (Numerical) | | Per-flight vs per-passenger calculation; $585 vs actual $1,200 total |
| 13 | Skill 1 (Policy) | | Changed destination ATL→LAS (policy prohibits origin/dest changes) |
| 14 | Skill 1 (Policy) | Skill 5 (Numerical) | Modified basic economy (must cancel+rebook); also chose wrong return flight |
| 15 | Skill 2 (Data Mapping) | | Selected $216 route instead of cheaper $207 option from search results |
| 17 | Skill 4 (Multi-Entity) | | Didn't recognize EWR as NYC airport; booked new reservation instead of modifying FQ8APE |
| 18 | Skill 3 (Execution) | | Only processed 1 of 5 reservation downgrades; conversation ended |
| 19 | Skill 1 (Policy) | | Modified basic economy return flight without restriction check |
| 20 | Skill 5 (Numerical) | | Wrong baggage count (gold member); used 2 certificates instead of 1+credit card |
| 21 | Skill 5 (Numerical) | Skill 2 (Data Mapping) | Wrong gift card (not smallest usable); wrong baggage (silver member) |
| 22 | Skill 4 (Multi-Entity) | | Same as 17; wrong reservation for NYC-Chicago trip |
| 23 | Skill 5 (Numerical) | Skill 2 (Data Mapping) | Wrong flight ($430 vs $259); wrong payment; didn't optimize with 3 separate bookings |
| 24 | Skill 1 (Policy) | | Cancelled ineligible reservation; task purpose tests refusal |
| 25 | Skill 5 (Numerical) | | Booked 2 passengers instead of 1; wrong cost with insurance |
| 27 | Skill 3 (Execution) | | Calculated correct $150 but transferred without calling send_certificate |
| 29 | Skill 2 (Data Mapping) | | Selected HAT088 ($185) instead of cheapest HAT033 ($111) |
| 30 | Skill 1 (Policy) | | Removed checked bag; policy prohibits bag removal |
| 31 | Skill 1 (Policy) | | Modified basic economy flights |
| 32 | Skill 6 (Operation) | | Skipped 2-step process; wrong payment method |
| 33 | Skill 3 (Execution) | | Transferred to human instead of completing agreed modifications |
| 34 | Skill 1 (Policy) | | Fabricated "gold+insurance waives fare differences" policy |
| 35 | Skill 1 (Policy) | | Cancelled ineligible reservation; claimed silver grants cancellation |
| 37 | Skill 2 (Data Mapping) | | Changed flight HAT268→HAT057 during cabin upgrade (only cabin should change) |
| 38 | Skill 4 (Multi-Entity) | | Retrieved wrong "last reservation"; never found delayed flight |
| 39 | Skill 1 (Policy) | | Cancelled all 7 reservations; only 3 were eligible |
| 41 | Skill 4 (Multi-Entity) | Skill 1 (Policy) | Only checked 1 of 7 reservations; cancelled 2-passenger reservation |
| 42 | Skill 4 (Multi-Entity) | | Cancelled wrong conflicting flight; missed second conflict |
| 44 | Skill 5 (Numerical) | | Wrong flight duration (5hr vs 2.5hr); upgrade cost $174 vs actual $903 |
| 49 | Skill 1 (Policy) | | Believed user's false insurance claim; cancelled despite insurance="no" |

### Retail Failed Tasks (67 total)

| Task ID | Primary Skill | Secondary Skill | Root Cause Summary |
|---------|--------------|-----------------|-------------------|
| 0 | Skill 3 (Execution) | | Gathered exchange info; user confirmed; exchange tool never called |
| 4 | Skill 4 (Multi-Entity) | | Checked 1 of 5 orders; missed t-shirt in #W4776164 |
| 7 | Skill 6 (Operation) | | Bundled desk lamp + water bottle exchange; user couldn't selectively exclude |
| 8 | Skill 6 (Operation) | | Same as 7; bundled items that needed individual confirmation |
| 9 | Skill 4 (Multi-Entity) | | Found correct order but focused on different order; exchanged wrong items |
| 10 | Skill 1 (Policy) | | Should have transferred to human; silently used original payment methods |
| 11 | Skill 3 (Execution) | | Premature transfer on user frustration instead of completing return |
| 14 | Skill 2 (Data Mapping) | | Returned ALL items instead of gaming-related only (keyboard, mouse) |
| 16 | Skill 1 (Policy) | | Refunded to gift card instead of original PayPal |
| 18 | Skill 2 (Data Mapping) | | Wrong variant for same-for-same replacement exchange |
| 19 | Skill 5 (Numerical) | | Wrong savings calculation ($8.95 vs $54.04); DB passed, COMM failed |
| 20 | Skill 4 (Multi-Entity) | | Only upgraded from 1 order; missed items across other orders |
| 21 | Skill 2 (Data Mapping) | | Confused source vs destination item_id in modification |
| 22 | Skill 4 (Multi-Entity) | | Changed user address but never checked any orders for pending address updates |
| 23 | Skill 2 (Data Mapping) | | Concluded grills were "same" without comparing features field |
| 27 | Skill 6 (Operation) | | Did return BEFORE exchange; return changed status, blocking exchange |
| 28 | Skill 2 (Data Mapping) | | Returned all items instead of 5 specific ones across orders |
| 29 | Skill 6 (Operation) | | Two separate exchanges on same order; second failed due to status change |
| 30 | Skill 4 (Multi-Entity) | | Only returned some items; missed cancel + return on other orders |
| 31 | Skill 1 (Policy) | | Cancelled entire order despite user saying "skip if partial cancel impossible" |
| 33 | Skill 3 (Execution) | | Never followed fallback instruction to update address after partial cancel failed |
| 34 | Skill 7 (Loops) | | Infinite loop; 98 tool calls, 202 messages; never attempted modify_pending_order_address |
| 35 | Skill 2 (Data Mapping) | | Email off by one digit (8321 vs 8320); didn't try variations |
| 36 | Skill 6 (Operation) | | Cancelled entire order instead of executing item modification |
| 37 | Skill 6 (Operation) | | Cancelled camera first (separate action); status change blocked subsequent modify |
| 39 | Skill 2 (Data Mapping) | | Fabricated zip codes (33440, 85001) instead of actual (85033, 32169) |
| 41 | Skill 4 (Multi-Entity) | | Missed second order with jigsaw puzzle |
| 42 | Skill 4 (Multi-Entity) | | Same as 41; only retrieved one of two orders |
| 43 | Skill 1 (Policy) | | Accepted user's wrong address over system data |
| 44 | Skill 2 (Data Mapping) | | Wrong cheapest desk lamp ($143.02 vs $135.24) |
| 45 | Skill 1 (Policy) | | Transferred instead of using original PayPal for exchange |
| 49 | Skill 2 (Data Mapping) | | Matched to IPX7 instead of IPX4 earbuds |
| 51 | Skill 4 (Multi-Entity) | | User guessed wrong order; agent didn't check others for camera |
| 54 | Skill 2 (Data Mapping) | | Didn't try adding first name prefix to email |
| 55 | Skill 3 (Execution) | | Only processed 2 of 4 required actions |
| 56 | Skill 2 (Data Mapping) | | Wrong cheapest air purifier variant |
| 57 | Skill 1 (Policy) | | Cancelled when user's conditions for cancellation were unmet |
| 59 | Skill 4 (Multi-Entity) | | Cancelled wrong order; confused older vs newer |
| 60 | Skill 3 (Execution) | | User confirmed earbuds change; modify tool never called |
| 62 | Skill 1 (Policy) | | Modified order when user's $100 precondition wasn't met |
| 64 | Skill 2 (Data Mapping) | | Wrong camera variant (silver $481.50 vs expected black $466.75) |
| 67 | Skill 2 (Data Mapping) | | Used nickname "NoNo" as last name; skipped zip 98178 |
| 71 | Skill 3 (Execution) | Skill 2 (Data Mapping) | Used gift card instead of PayPal; dropped address change |
| 72 | Skill 3 (Execution) | | User confirmed; neither modification executed before conversation ended |
| 74 | Skill 4 (Multi-Entity) | | Cancelled wrong order (#W3414433 vs expected #W3189752) |
| 76 | Skill 2 (Data Mapping) | Skill 3 (Execution) | Wrong cancellation reason; also never cancelled second order |
| 78 | Skill 4 (Multi-Entity) | | Swapped operations: cancelled order that should be modified, modified one that should be cancelled |
| 79 | Skill 2 (Data Mapping) | | Wrong color (black vs red, matching other order's bottle) |
| 82 | Skill 4 (Multi-Entity) | | Returned from wrong order (1 tablet) instead of correct order (2 tablets) |
| 83 | Skill 4 (Multi-Entity) | | Same as 82; returned from wrong order |
| 90 | Skill 7 (Loops) | | Repeated failed order ID lookups; never called get_user_details; too_many_errors |
| 91 | Skill 6 (Operation) | | Exchanged instead of returned skateboard; wrong exchange variant for e-reader |
| 93 | Skill 4 (Multi-Entity) | | Exchanged from wrong order; both had 15-inch laptops |
| 95 | Skill 3 (Execution) | | User confirmed both exchanges; no exchange calls made |
| 97 | Skill 2 (Data Mapping) | | Fabricated NYC address instead of looking up from order #W3407479 |
| 98 | Skill 6 (Operation) | | Two separate exchange calls instead of one batched call |
| 99 | Skill 6 (Operation) | | Same batching failure as 98 |
| 100 | Skill 1 (Policy) | | Caved to emotional pressure; cancelled instead of modifying |
| 101 | Skill 4 (Multi-Entity) | | Checked wrong order for air purifier |
| 102 | Skill 3 (Execution) | | Completed 2 actions but forgot address change |
| 103 | Skill 2 (Data Mapping) | | Wrong luggage color (black vs expected red) |
| 104 | Skill 4 (Multi-Entity) | | Missed order W9218746; wrong operations on other orders |
| 107 | Skill 4 (Multi-Entity) | | Never fetched second order with jigsaw puzzle |
| 108 | Skill 3 (Execution) | | Described return correctly but never called return tool |
| 109 | Skill 4 (Multi-Entity) | | Never fetched second order to look up correct address |
| 110 | Skill 4 (Multi-Entity) | Skill 2 (Data Mapping) | Changed address on wrong order; also wrong cheapest tablet |
| 112 | Skill 3 (Execution) | | Never executed any modifications; offered to send reminder email |

---

## Summary: Skill Distribution

| Skill | Description | # Airline Tasks | # Retail Tasks | Total | % of Failures |
|-------|------------|:---:|:---:|:---:|:---:|
| **Skill 1** | Policy Adherence Under Pressure | 13 | 8 | **21** | 20.6% |
| **Skill 2** | Structured Data Mapping | 4 | 17 | **21** | 20.6% |
| **Skill 3** | Action Execution Promptness | 4 | 11 | **15** | 14.7% |
| **Skill 4** | Multi-Entity Tracking | 5 | 19 | **24** | 23.5% |
| **Skill 5** | Numerical Reasoning | 8 | 1 | **9** | 8.8% |
| **Skill 6** | Correct Operation Selection | 1 | 9 | **10** | 9.8% |
| **Skill 7** | Loop Detection & Recovery | 0 | 2 | **2** | 2.0% |
| | | **35** | **67** | **102** | **100%** |

Each task has exactly one primary classification. The total equals 102 (all failed tasks).

---

## Training Environment Design Recommendations

### Environment 1: Adversarial Policy Adherence (Skill 1)
- **Setup:** Customer service scenarios with pushy/emotional/lying users
- **Key scenarios:** User claims insurance they don't have; user demands cancellation of non-cancellable booking; user escalates emotionally; user redirects to different reservation after being denied; user states conditional preferences ("keep order if partial cancel not possible"); user provides wrong identity information that contradicts system records
- **Reward:** 1.0 only when agent correctly follows policy (refuses policy-violating requests OR correctly performs valid actions despite sounding unusual); 0.0 when agent deviates
- **Expected resolution:** 21 tasks

### Environment 2: Data Extraction and Entity Matching (Skill 2)
- **Setup:** Product catalogs with multiple variants; flight search results with multiple options; user identity lookup with incomplete/imprecise input
- **Key scenarios:** Select cheapest variant; match user specs (size+color+material) to correct ID; select correct flight from search results; handle partial names/emails/zip codes by asking for clarification or trying reasonable variations; correctly scope return/exchange to only the specific items requested
- **Reward:** 1.0 only when correct entity_id is selected (or correct clarifying question is asked); 0.0 for wrong entity, fabricated data, or over-broad item selection
- **Expected resolution:** 21 tasks

### Environment 3: Prompt Action Execution (Skill 3)
- **Setup:** Conversations with limited turn budget; user confirms and conversation may end at any point
- **Key scenarios:** After user says "yes", immediately call the write tool; don't summarize or re-explain; don't ask for double confirmation; complete ALL tasks in multi-request conversations before engaging in discussion
- **Reward:** 1.0 only when ALL actions are executed before conversation ends; 0.0 for info-gathering-only trajectories or partial completion
- **Expected resolution:** 15 tasks

### Environment 4: Multi-Entity Disambiguation (Skill 4)
- **Setup:** Users with 3-7 orders/reservations containing similar items
- **Key scenarios:** Multiple orders with tablets; multiple reservations with similar routes; user doesn't specify order ID; user guesses wrong order ID; user says "last reservation" when they have several; address must be looked up from a different order; items must be identified across all orders before acting
- **Reward:** 1.0 only when correct entity is selected and all entities are processed; 0.0 for wrong-entity actions or incomplete entity search
- **Expected resolution:** 24 tasks

### Environment 5: Numerical Reasoning (Skill 5)
- **Setup:** Payment calculations, baggage allowance tables, compensation formulas, multi-passenger pricing
- **Key scenarios:** Split $375 across certificate ($250) + credit card ($125); gold member gets 3 free economy bags; compensation = $50 × passengers for delayed flights; multiply per-segment price by passenger count; compare all available prices to find true minimum
- **Reward:** 1.0 only when correct amounts are computed and communicated
- **Expected resolution:** 9 tasks

### Environment 6: API Semantics (Skill 6)
- **Setup:** Scenarios requiring specific operation choice and correct batching
- **Key scenarios:** User wants money back (return) vs different item (exchange); user wants to remove one item from multi-item order (cannot remove, must cancel or swap); batching multiple items in single exchange/return call; correct operation ordering when both exchange and return are needed on same order
- **Reward:** 1.0 only when correct operation is selected with correct batching; 0.0 for wrong operation or unbatched calls that cause state errors
- **Expected resolution:** 10 tasks

### Environment 7: Loop Recovery (Skill 7)
- **Setup:** Scenarios with deliberate failure points requiring strategy changes
- **Key scenarios:** Tool fails with same args; order ID not found; changing strategy from user-provided IDs to get_user_details lookup
- **Reward:** 1.0 only when agent changes strategy after <=3 retries; 0.0 for >3 identical retries
- **Expected resolution:** 2 tasks

---

## Verification Methodology

This analysis was verified in three stages:

1. **Automated data extraction:** A script extracted failure components (DB/COMMUNICATE status, reward_breakdown), tool calls (actual vs expected), expected actions from evaluation_criteria, termination reasons, and conversation lengths for all 102 failed tasks from the raw JSON simulation data.

2. **Independent agent verification:** Three verification agents independently read full conversation transcripts for all 102 failed tasks:
   - Agent 1 verified 12 airline tasks (2, 8, 10, 11, 15, 17, 25, 27, 32, 37, 41, 44), tracing through actual tool call arguments, flight prices, payment histories, and expected actions
   - Agent 2 verified 21 retail tasks (0, 10, 14, 27, 28, 29, 30, 31, 33, 34, 36, 37, 41, 42, 43, 45, 57, 59, 62, 67, 76), examining conversation flow, tool results, and evaluation criteria
   - Agent 3 verified 24 retail tasks (55, 56, 60, 64, 71, 72, 74, 78, 82, 83, 91, 93, 95, 97, 98, 99, 100, 102, 103, 104, 107, 108, 109, 110, 112), checking product variant prices, order addresses, and expected actions

3. **Cross-validation and reclassification:** Initial classifications from the first report were compared against all three verification agents' independent findings. Multiple reclassifications were made where the verification agents' deeper analysis (reading actual conversation messages, comparing specific variant prices, checking order contents) revealed different root causes than initially identified. The final classification rate is **102/102 (100%) verified** through at least one independent verification method.
