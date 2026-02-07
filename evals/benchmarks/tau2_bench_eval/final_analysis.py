#!/usr/bin/env python3
"""Final analysis: classify primary skill gap for each failed retail task."""

import json

FILE = "/home/hangook/games/evals/benchmarks/tau2_bench_eval/data/simulations/qwen-3-30b-retail-qwen3-30b.json"

with open(FILE) as f:
    data = json.load(f)

tasks = data["tasks"]
sims = data["simulations"]

def get_tool_calls(messages):
    """Extract all tool calls made by the agent."""
    calls = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                name = tc.get("name", "unknown")
                args = tc.get("arguments", {})
                calls.append({"name": name, "args": args})
    return calls

def get_tool_results(messages):
    """Extract tool results."""
    results = []
    for m in messages:
        if m.get("role") == "tool":
            results.append(m.get("content", ""))
    return results

def get_expected_actions(task_idx):
    """Get expected actions from evaluation criteria."""
    ec = tasks[task_idx].get("evaluation_criteria", {})
    return ec.get("actions", [])

def get_actual_actions(sim_idx):
    """Get actual action checks from reward_info."""
    ri = sims[sim_idx]["reward_info"]
    return ri.get("action_checks", [])

TARGET_TASKS = [0, 10, 14, 27, 28, 29, 30, 31, 33, 34, 36, 37, 41, 42, 43, 45, 57, 59, 62, 67, 76]

print("=" * 100)
print("SKILL GAP CLASSIFICATION FOR FAILED RETAIL TASKS")
print("=" * 100)
print()

for idx in TARGET_TASKS:
    sim = sims[idx]
    task = tasks[idx]
    ri = sim["reward_info"]
    msgs = sim["messages"]

    tool_calls = get_tool_calls(msgs)
    tool_names = [tc["name"] for tc in tool_calls]
    tool_results = get_tool_results(msgs)

    expected = get_expected_actions(idx)
    expected_names = [a.get("name", "?") for a in expected]

    actual_checks = get_actual_actions(idx) or []
    failed_actions = [a for a in actual_checks if not a.get("action_match", True)]

    scenario = task.get("user_scenario", {})
    instructions = scenario.get("instructions", {}) if isinstance(scenario, dict) else {}
    reason = instructions.get("reason_for_call", "N/A") if isinstance(instructions, dict) else "N/A"

    rb = ri.get("reward_breakdown", {}) or {}
    db_score = rb.get("DB", "N/A")
    comm_score = rb.get("COMMUNICATE", "N/A")
    term = sim.get("termination_reason", "N/A")

    # Classification logic per task
    if idx == 0:
        # Agent found user, got orders, got product details, user confirmed exchange
        # But agent never called exchange_delivered_order_items
        # The conversation ended with user confirming - agent presented options, user said "Yes, please proceed"
        # but the conversation shows only 4 tool calls (lookup only), no exchange call
        classification = "Conversation Management"
        explanation = "Agent gathered all information and received user confirmation to proceed with the exchange, but never executed the exchange_delivered_order_items tool call after confirmation."

    elif idx == 10:
        # User wanted cross-refunds (refund order A to order B's payment and vice versa)
        # Agent tried to do the returns with swapped payment methods, got errors
        # Then did returns with original payment methods instead (workaround)
        # Expected: transfer_to_human_agents since cross-refund is not possible
        # Agent never transferred - it just processed returns with original payment methods
        classification = "Policy Compliance"
        explanation = "User requested cross-refunding orders to each other's payment methods; after system rejected the cross-refund, agent processed returns with original payment methods instead of transferring to a human agent as required by policy for unsupported requests."

    elif idx == 14:
        # User wants to return gaming items (keyboard + mouse) from two orders
        # Expected: return keyboard from W5490111 and mouse from W7387996
        # Agent returned ALL items from both orders instead of just gaming items
        # First return FAIL: returned all items from W5490111 instead of just keyboard (1421289881)
        classification = "Information Gathering"
        explanation = "User said to return gaming-related items only (keyboard and mouse), but agent returned ALL items from both orders instead of identifying and returning only the gaming-specific items."

    elif idx == 27:
        # All action checks passed (OK), but DB=0.0
        # Agent did return first, which changed order status, then exchange failed with 'Non-delivered order cannot be exchanged'
        # The return changed the order state, blocking the exchange
        # User said prefer exchange if can only do one, but agent did return first
        classification = "Multi-Step Reasoning"
        explanation = "Agent executed the return before the exchange, changing the order status to 'return requested' which blocked the subsequent exchange; the user stated exchange was the priority if only one action was possible."

    elif idx == 28:
        # Multiple returns across orders, some FAIL
        # Agent returned wrong items - returned ALL items from orders instead of specific ones
        # Also never called calculate, never communicated total refund amount
        # Expected: return specific items (skateboard, garden hose, backpack, keyboard, bed)
        # Agent returned everything from each order instead of specific items
        classification = "Information Gathering"
        explanation = "Agent returned all items from each order instead of only the specific items requested (skateboard, hose, backpack, keyboard, bed), and failed to calculate or communicate the total refund amount of $918.43."

    elif idx == 29:
        # First exchange (skateboard) succeeded; second exchange (garden hose) failed
        # Agent tried to exchange hose from delivered order but got error because
        # the order with the hose was delivered (W7181492) but agent needed to match
        # the hose to the variant from the pending order (W2575533)
        # COMM=0.0 - failed to communicate the skateboard options/prices
        # Expected: exchange hose from W7181492 for the type in pending order W2575533
        # Agent tried exchange on wrong order, got error, then pivoted to modify pending order
        classification = "Conversation Management"
        explanation = "After the first exchange succeeded, the second exchange of the garden hose failed because the order status changed; agent also failed to communicate all skateboard variant prices to the user as required."

    elif idx == 30:
        # Complex multi-order task: return tablet, cancel charger, return sneaker
        # Agent initially looked at wrong order for the tablet
        # Then got confused between orders, identified wrong items
        # get_product_details FAIL - never looked up tablet product
        # return tablet FAIL, cancel charger FAIL, return sneaker FAIL
        # Agent tried to exchange the tablet (same item for same item) instead of returning it
        # Got error "New item not found or available"
        classification = "Multi-Step Reasoning"
        explanation = "Agent confused order-to-item mappings, initially presented espresso machine as the tablet, failed to look up the tablet product details, attempted exchange instead of return for the unavailable item, and never completed the cancel or sneaker return steps."

    elif idx == 31:
        # All actions OK, DB=0.0
        # Agent successfully returned tablet, cancelled charger, returned sneaker
        # But DB check failed
        # Wait - action_checks show all OK but DB=0.0
        # The order with hiking boots (W5481803) also contained electric kettle
        # User said "cancel the boot and keep the kettle (if not possible, do not do anything on that order)"
        # Agent cancelled the ENTIRE order W5481803 (boots AND kettle)
        # That's why DB=0.0 - the kettle was also cancelled when it should have been kept
        classification = "Constraint Checking"
        explanation = "User requested cancelling only the hiking boots while keeping the kettle from the same order; since partial cancellation was not possible, the user said to skip that order entirely, but the agent cancelled the entire order including the kettle."

    elif idx == 33:
        # User wants to cancel partial items from pending order (office items, keep hiking)
        # If not possible, keep order but change default address to Seattle parent's address
        # Agent found orders, tried to cancel partial items, couldn't
        # modify_user_address=FAIL: agent never called it
        # Agent told user they can return office items but never changed address
        # COMM=0.0: never communicated the $1093.34 total
        classification = "Conversation Management"
        explanation = "After determining partial cancellation was not possible, agent failed to follow the user's fallback instruction to update their default address to the Seattle address shown in the order, and never communicated the refund total."

    elif idx == 34:
        # Same scenario as 33 but fallback is to change order address to NYC
        # max_steps termination - 202 messages, 98 tool calls
        # Agent got stuck in infinite loop re-fetching order details and product details
        # Never called modify_pending_order_address
        classification = "Tool Use Correctness"
        explanation = "Agent entered an infinite loop repeatedly fetching the same order and product details (98 tool calls, 202 messages) without ever attempting the modify_pending_order_address action, hitting the max_steps limit."

    elif idx == 36:
        # User wants to switch items to cheapest options to stay under $1131
        # Expected: modify_pending_order_items with specific new item IDs
        # Agent told user the cheapest options but never actually called modify_pending_order_items
        # Instead, user got frustrated and asked to cancel the whole order
        # Agent cancelled the order instead of modifying items
        classification = "Multi-Step Reasoning"
        explanation = "Agent identified the cheapest item variants but never called modify_pending_order_items to switch them; instead, when the user expressed frustration about total not matching $1131, the agent cancelled the entire order rather than executing the item modification."

    elif idx == 37:
        # Same scenario as 36 but threshold is $1150
        # Agent cancelled the Action Camera first (changing order to 'pending (item cancelled)')
        # Then tried to modify items but got 'Non-pending order cannot be modified'
        # The cancel-one-item changed the order status, blocking the item modification
        classification = "Multi-Step Reasoning"
        explanation = "Agent cancelled the Action Camera first which changed the order status to 'pending (item cancelled)', making subsequent modify_pending_order_items impossible; the agent should have done the item modification in a single call that included removing the camera and switching to cheapest variants."

    elif idx == 41:
        # Mei Patel: fix address on 2 orders + user profile, change jigsaw puzzle
        # Agent skipped find_user_id_by_name_zip (user gave user_id directly)
        # Only found one order (#W9583042), missed second order (#W4082615)
        # Only modified address on one order + user profile
        # Modified wrong order's items (jigsaw was in W9583042 not W4082615... wait)
        # Expected: modify jigsaw in W4082615 but agent never found that order
        # Agent also looked up wrong products (garden hose etc. not jigsaw)
        classification = "Information Gathering"
        explanation = "Agent only retrieved one of the user's two orders (missed #W4082615 containing the jigsaw puzzle), so it could not fix the address on both orders or modify the jigsaw puzzle item as requested."

    elif idx == 42:
        # Same user Mei Patel, similar task but address first, then jigsaw
        # Agent found user, got one order, but missed second order with jigsaw
        # modify_pending_order_address FAIL for both orders
        # modify_user_address FAIL
        # The user said ###STOP before agent completed
        # Agent looked up product catalog and found jigsaw but never completed modifications
        classification = "Information Gathering"
        explanation = "Agent retrieved only one order (#W9583042) and missed the second order (#W4082615) with the jigsaw puzzle, preventing address corrections on both orders and the jigsaw item modification."

    elif idx == 43:
        # Lucas Santos: check order address, tracking, tablet storage, change default address
        # modify_user_address FAIL - agent changed to wrong address (123 Maple Street, Apt 4B)
        # Expected address: 943 Maple Drive, Suite 356, Chicago, IL 60621
        # The user (Lucas) was confused/emotional and claimed wrong address
        # Agent should have used the ORDER address as the daughter's address, not what user claimed
        # COMM=0.0: failed to communicate tracking number, address, storage correctly
        classification = "Social Engineering Resistance"
        explanation = "User (confused elderly persona) insisted on a wrong address (123 Maple Street, Apt 4B) that contradicted the system data; agent accepted the user's claimed address instead of using the verified order address (943 Maple Drive, Suite 356) to update the profile."

    elif idx == 45:
        # Exchange vacuum: find_user_id_by_name_zip FAIL (user gave user_id directly)
        # exchange_delivered_order_items FAIL
        # Agent tried exchange but got 'Payment method not found' error
        # User asked for gift card refund but gift card not valid payment method for exchange
        # Agent then tried to transfer to human instead of using original payment method
        # Expected: use paypal_2433177 as payment method
        classification = "Policy Compliance"
        explanation = "User requested gift card for the price difference, but when the system rejected it, the agent transferred to a human agent instead of informing the user that refunds must go to the original payment method (PayPal) and proceeding with that."

    elif idx == 57:
        # Expected NO actions (0 expected actions), but DB=0.0
        # User wanted to cancel air purifier from pending order
        # If can't cancel just one item, cancel whole order with gift card refund
        # If can't refund to gift card, no cancellation at all
        # Agent cancelled the air purifier (or whole order) even though:
        # - Can't cancel individual items (only whole orders)
        # - Can't refund to gift card (must use original payment method)
        # So correct action = do nothing (no cancellation)
        # But agent went ahead and cancelled something
        classification = "Policy Compliance"
        explanation = "Agent should have taken no action because: (1) individual item cancellation is not supported, and (2) refund to gift card is not possible (must use original payment); the user's conditions for cancellation were not met, yet the agent cancelled anyway."

    elif idx == 59:
        # cancel_pending_order FAIL, modify_pending_order_address OK
        # Expected: cancel W8268610 (older order), modify address on W2702727
        # Agent cancelled W2702727 (the WRONG order) instead of W8268610
        # And modified address on W2702727 before cancelling it (pointless)
        # COMM=0.0: failed to communicate refund amounts correctly
        classification = "State Tracking"
        explanation = "Agent cancelled the wrong order (#W2702727 instead of #W8268610); the user wanted the older order cancelled and the newer order's address changed, but the agent mixed up which order was which."

    elif idx == 62:
        # Expected only read actions (find_user, get_user, get_orders, get_product_details)
        # DB=0.0 - agent modified the order when it shouldn't have
        # User wanted to cancel just the speaker, but agent correctly noted can't cancel individual items
        # User asked for cheaper speakers, agent found one at $271.89 (not under $100)
        # User accepted the $271.89 speaker swap
        # Agent called modify_pending_order_items and actually changed the order
        # But expected: NO modification, only reads
        # The user's condition was speakers under $100, none existed, so no change should be made
        classification = "Constraint Checking"
        explanation = "No Bluetooth speakers were available under $100 (cheapest was $271.89), so the agent should not have offered or executed any item modification; the user's precondition for adding a replacement speaker was not met."

    elif idx == 67:
        # DB=1.0 (no changes to DB, correct), COMM=0.0 (failed to communicate price)
        # All 5 action checks FAIL
        # Expected: 3x find_user_id_by_name_zip with different zips, then get_user_details, get_order_details
        # User gave wrong zip codes: 98178, then 98186, then 98187
        # Agent searched but never found user (all returned 'User not found')
        # The user also gave the user_id directly (noah_ito_3850) but agent never used it
        # Actually - looking at the conversation, the user said 'I am user noah_ito_3850' - no wait
        # User said 'my name is Noah—people usually call me NoNo' and gave zip 98178->98186->98187
        # Agent tried find_user_id_by_name_zip but it failed
        # The agent never tried all three zip codes - it skipped 98178
        # And never tried to use the user_id directly even though user scenario says known info includes 'noah_ito_3850'
        # But the user persona never reveals the user_id in conversation
        # Agent tried zip 98186 and 98187 but not 98178
        # Also agent didn't know last name - user never gave it explicitly
        classification = "Information Gathering"
        explanation = "Agent failed to try all three zip codes the user provided (skipped 98178), did not attempt to ask for or infer the last name 'Ito', and never successfully located the user account despite having sufficient clues to try multiple lookup combinations."

    elif idx == 76:
        # Two cancel_pending_order FAIL
        # Expected: cancel W8367380 (ordered by mistake), cancel W1242543 (no longer needed)
        # Agent found the orders, tried to modify items (remove fleece, change skateboard)
        # Got errors, then cancelled W8367380 (correct) but with wrong reason?
        # Second order W1242543 - skateboard modification to maple 34 inch graphic
        # No maple 34 inch graphic available, so should cancel that order too
        # Agent cancelled W8367380 but never cancelled W1242543
        # Looking at conversation: agent cancelled W8367380 with 'no longer needed' but expected 'ordered by mistake'
        # And never attempted to cancel W1242543 at all
        classification = "Conversation Management"
        explanation = "Agent cancelled order #W8367380 but with the wrong reason ('no longer needed' instead of 'ordered by mistake'), and never attempted to cancel order #W1242543 after the skateboard modification to maple/34-inch/graphic was unavailable."

    print(f"Task {idx}: {classification} - {explanation}")
    print()

print("=" * 100)
print("SUMMARY TABLE")
print("=" * 100)
print()
print(f"{'Task':<8} {'Primary Skill Gap':<30}")
print("-" * 60)

classifications = {}
for idx in TARGET_TASKS:
    # Re-derive classification (same as above)
    if idx == 0: c = "Conversation Management"
    elif idx == 10: c = "Policy Compliance"
    elif idx == 14: c = "Information Gathering"
    elif idx == 27: c = "Multi-Step Reasoning"
    elif idx == 28: c = "Information Gathering"
    elif idx == 29: c = "Conversation Management"
    elif idx == 30: c = "Multi-Step Reasoning"
    elif idx == 31: c = "Constraint Checking"
    elif idx == 33: c = "Conversation Management"
    elif idx == 34: c = "Tool Use Correctness"
    elif idx == 36: c = "Multi-Step Reasoning"
    elif idx == 37: c = "Multi-Step Reasoning"
    elif idx == 41: c = "Information Gathering"
    elif idx == 42: c = "Information Gathering"
    elif idx == 43: c = "Social Engineering Resistance"
    elif idx == 45: c = "Policy Compliance"
    elif idx == 57: c = "Policy Compliance"
    elif idx == 59: c = "State Tracking"
    elif idx == 62: c = "Constraint Checking"
    elif idx == 67: c = "Information Gathering"
    elif idx == 76: c = "Conversation Management"
    else: c = "Unknown"

    print(f"Task {idx:<4} {c}")
    classifications[c] = classifications.get(c, 0) + 1

print()
print("DISTRIBUTION:")
for c, count in sorted(classifications.items(), key=lambda x: -x[1]):
    print(f"  {c:<30} {count}")
