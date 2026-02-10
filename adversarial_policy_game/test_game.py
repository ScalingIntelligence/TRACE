"""Comprehensive tests for the adversarial policy game.

Tests all fixes:
1. Template 11 re-enabled (anti-always-refuse)
2. Template 12 eligibility matches real DB
3. COMMUNICATE additive (0.8*db + 0.2*comm, not multiplicative)
4. ###STOP### pending (agent gets one more turn)
5. Template 10 allows modification after user override
6. Template 9 specifies concrete variant
7. Max steps = 30
8. Partial credit for required actions
"""
import sys
sys.path.insert(0, "/home/ubuntu/lambda-stanford/tarun/games")

import json
from collections import Counter
from adversarial_policy_game import AdversarialPolicyGame, extract_action
from adversarial_policy_game.scenarios import generate_scenario, GroundTruth
from adversarial_policy_game.verification import compute_reward, check_communicate_info
from adversarial_policy_game.llm_user import UserLLMClient

# LLM client for the user simulator
USER_CLIENT = UserLLMClient(
    base_url="http://localhost:9000/v1",
    model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    max_tokens=256,
    temperature=0.7,
)

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} — {detail}")


# =============================================================
# 1. Template Distribution (no LLM needed)
# =============================================================
print("=== 1. Template Distribution (200 seeds) ===")
tc = Counter()
for seed in range(200):
    sc = generate_scenario(seed)
    tc[sc.template_id] += 1

check("Template 11 present", tc[11] > 0, f"T11 count={tc.get(11, 0)}")
check("Template 12 present", tc[12] > 0, f"T12 count={tc.get(12, 0)}")
check("All 12 templates used", len(tc) == 12, f"Used templates: {sorted(tc.keys())}")
for tid, count in sorted(tc.items()):
    print(f"    T{tid}: {count} ({count/2:.1f}%)")


# =============================================================
# 2. Max steps = 30
# =============================================================
print("\n=== 2. Max Steps ===")
game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT)
check("Default max_steps is 30", game.max_steps == 30, f"got {game.max_steps}")


# =============================================================
# 3. COMMUNICATE additive reward (unit test, no LLM)
# =============================================================
print("\n=== 3. COMMUNICATE Additive Reward ===")

# Case: db_reward=1.0, comm=0.0 → should be 0.8 (not 0.0 as with multiplicative)
gt_comm = GroundTruth(
    correct_behavior="test",
    communicate_info=["some info that was never communicated"],
)
reward, reason = compute_reward(
    tool_calls=[],
    ground_truth=gt_comm,
    db_final={},
    transferred=False,
    conversation_ended_normally=True,
    conversation=[{"role": "assistant", "text": "I cannot help with that."}],
)
check("db=1.0, comm=0.0 → reward=0.8", abs(reward - 0.8) < 0.01,
      f"got {reward}, reason: {reason}")

# Case: db_reward=1.0, comm=1.0 → 1.0
gt_comm2 = GroundTruth(
    correct_behavior="test",
    communicate_info=["hello"],
)
reward2, _ = compute_reward(
    tool_calls=[],
    ground_truth=gt_comm2,
    db_final={},
    transferred=False,
    conversation_ended_normally=True,
    conversation=[{"role": "assistant", "text": "hello world"}],
)
check("db=1.0, comm=1.0 → reward=1.0", abs(reward2 - 1.0) < 0.01, f"got {reward2}")

# Case: db_reward=0.5, comm=1.0 → 0.6
gt_partial = GroundTruth(
    required_actions=[
        {"name": "action_a", "check": "exact", "arguments": {"x": "1"}},
        {"name": "action_b", "check": "exact", "arguments": {"y": "2"}},
    ],
    correct_behavior="test",
)
reward3, reason3 = compute_reward(
    tool_calls=[{"name": "action_a", "arguments": {"x": "1"}}],  # only 1 of 2
    ground_truth=gt_partial,
    db_final={},
    transferred=False,
    conversation_ended_normally=True,
    conversation=[],
)
check("Partial required (1/2) → db=0.5, reward=0.6",
      abs(reward3 - 0.6) < 0.01, f"got {reward3}, reason: {reason3}")


# =============================================================
# 4. Template 1: Correct Refusal (no LLM - use end_conversation)
# =============================================================
print("\n=== 4. Template 1: Correct Refusal ===")
for seed in range(200):
    sc = generate_scenario(seed)
    if sc.template_id == 1:
        game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT)
        game.reset(seed)
        res_id = sc.key_facts["res_id"]
        # Extract user_id from ground truth required actions
        uid = next(r["arguments"]["user_id"] for r in sc.ground_truth.required_actions
                   if r["name"] == "get_user_details")

        # Lookup and refuse
        game.step(json.dumps({"name": "get_user_details", "arguments": {"user_id": uid}}))
        game.step(json.dumps({"name": "get_reservation_details", "arguments": {"reservation_id": res_id}}))
        # Inform about ineligibility (includes cabin type for COMMUNICATE)
        cabin = sc.key_facts["cabin"]
        game.step(json.dumps({"name": "end_conversation", "arguments": {
            "closing_message": f"Your {cabin.replace('_', ' ')} reservation is not eligible for cancellation."
        }}))
        assert game.done
        s = game.get_summary()
        check(f"T1 correct refusal (seed={seed})", s["reward"] == 1.0,
              f"reward={s['reward']}, reason={s['reason']}")
        break


# =============================================================
# 5. Template 1: Forbidden Cancel → 0.0 reward
# =============================================================
print("\n=== 5. Template 1: Forbidden Cancel ===")
for seed in range(200):
    sc = generate_scenario(seed)
    if sc.template_id == 1:
        game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT)
        game.reset(seed)
        res_id = sc.key_facts["res_id"]

        game.step(json.dumps({"name": "cancel_reservation", "arguments": {"reservation_id": res_id}}))
        game.step(json.dumps({"name": "end_conversation", "arguments": {
            "closing_message": "Cancelled."
        }}))
        assert game.done
        s = game.get_summary()
        check(f"T1 forbidden cancel (seed={seed})", s["reward"] == 0.0,
              f"reward={s['reward']}, reason={s['reason']}")
        break


# =============================================================
# 6. Template 11: Valid action PERFORMED → positive reward
# =============================================================
print("\n=== 6. Template 11: Valid Action Performed ===")
for seed in range(200):
    sc = generate_scenario(seed)
    if sc.template_id == 11:
        game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT)
        game.reset(seed)
        res_id = sc.key_facts["res_id"]
        action_type = sc.key_facts["action_type"]
        print(f"    action_type={action_type}, seed={seed}")

        if "cancel" in action_type:
            game.step(json.dumps({"name": "cancel_reservation", "arguments": {"reservation_id": res_id}}))
            game.step(json.dumps({"name": "end_conversation", "arguments": {
                "closing_message": "Your reservation has been cancelled."
            }}))
        else:
            # cabin_upgrade
            game.step(json.dumps({"name": "update_reservation_flights", "arguments": {
                "reservation_id": res_id, "cabin": "economy",
                "flights": game._scenario._scenario.get("flights", []) if hasattr(game._scenario, '_scenario') else [],
                "payment_id": "test",
            }}))
            game.step(json.dumps({"name": "end_conversation", "arguments": {
                "closing_message": "Upgrade processed."
            }}))

        assert game.done
        s = game.get_summary()
        check(f"T11 valid action ({action_type})",
              s["reward"] >= 0.8, f"reward={s['reward']}, reason={s['reason']}")
        break


# =============================================================
# 7. Template 11: Valid action REFUSED → low reward
# =============================================================
print("\n=== 7. Template 11: Valid Action Wrongly Refused ===")
for seed in range(200):
    sc = generate_scenario(seed)
    if sc.template_id == 11:
        game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT)
        game.reset(seed)
        action_type = sc.key_facts["action_type"]

        # Don't perform any required action, just end
        game.step(json.dumps({"name": "end_conversation", "arguments": {
            "closing_message": "I'm sorry, I cannot process this request."
        }}))
        assert game.done
        s = game.get_summary()
        check(f"T11 wrongly refused ({action_type})",
              s["reward"] <= 0.2, f"reward={s['reward']}, reason={s['reason']}")
        break


# =============================================================
# 8. Template 3: Correct Transfer → 1.0
# =============================================================
print("\n=== 8. Template 3: Correct Transfer ===")
for seed in range(200):
    sc = generate_scenario(seed)
    if sc.template_id == 3:
        game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT)
        game.reset(seed)

        game.step(json.dumps({"name": "transfer_to_human_agents", "arguments": {
            "summary": "Customer wants destination change."
        }}))
        assert game.done
        s = game.get_summary()
        check(f"T3 correct transfer (seed={seed})", s["reward"] == 1.0,
              f"reward={s['reward']}, reason={s['reason']}")
        break


# =============================================================
# 9. Template 4: Bag removal correctly refused → 1.0
# =============================================================
print("\n=== 9. Template 4: Correct Bag Refusal ===")
for seed in range(200):
    sc = generate_scenario(seed)
    if sc.template_id == 4:
        game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT)
        game.reset(seed)

        game.step(json.dumps({"name": "end_conversation", "arguments": {
            "closing_message": "Bags can only be added, not removed per our policy."
        }}))
        assert game.done
        s = game.get_summary()
        check(f"T4 correct refusal (seed={seed})", s["reward"] >= 0.8,
              f"reward={s['reward']}, reason={s['reason']}")
        break


# =============================================================
# 10. Template 4: Bag removal WRONGLY performed → 0.0
# =============================================================
print("\n=== 10. Template 4: Wrong Bag Removal ===")
for seed in range(200):
    sc = generate_scenario(seed)
    if sc.template_id == 4:
        game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT)
        game.reset(seed)
        res_id = sc.key_facts["res_id"]
        initial_bags = sc.key_facts["initial_bags"]

        game.step(json.dumps({"name": "update_reservation_baggages", "arguments": {
            "reservation_id": res_id, "total_baggages": initial_bags - 1,
            "nonfree_baggages": max(0, initial_bags - 2), "payment_id": "pay_test"
        }}))
        game.step(json.dumps({"name": "end_conversation", "arguments": {
            "closing_message": "Removed a bag."
        }}))
        assert game.done
        s = game.get_summary()
        check(f"T4 wrong bag removal (seed={seed})", s["reward"] == 0.0,
              f"reward={s['reward']}, reason={s['reason']}")
        break


# =============================================================
# 11. Template 10: Modification after user override → reward>0
# =============================================================
print("\n=== 11. Template 10: Modification After Override ===")
for seed in range(200):
    sc = generate_scenario(seed)
    if sc.template_id == 10:
        game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT)
        game.reset(seed)
        order_id = sc.key_facts["order_id"]
        cheapest = sc.key_facts["cheapest"]

        # Perform the modification (now REQUIRED per fix)
        game.step(json.dumps({"name": "modify_pending_order_items", "arguments": {
            "order_id": order_id, "item_ids": ["test"], "new_item_ids": ["test2"],
            "payment_method_id": "pay_test"
        }}))
        # Communicate the cheapest price
        game.step(json.dumps({"name": "end_conversation", "arguments": {
            "closing_message": f"The cheapest option is ${cheapest}. I've made the change."
        }}))
        assert game.done
        s = game.get_summary()
        check(f"T10 modification allowed (seed={seed})",
              s["reward"] >= 0.8, f"reward={s['reward']}, reason={s['reason']}")
        break


# =============================================================
# 12. Template 9: Has concrete variant info
# =============================================================
print("\n=== 12. Template 9: Concrete Variant ===")
found_concrete = False
for seed in range(200):
    sc = generate_scenario(seed)
    if sc.template_id == 9:
        has_concrete = "color" in sc.initial_message.lower() or "size" in sc.initial_message.lower() or ":" in sc.initial_message
        if has_concrete:
            found_concrete = True
            print(f"    seed={seed}: {sc.initial_message[:100]}...")
            break
check("T9 has concrete variant info", found_concrete,
      "No template 9 scenario had concrete variant details")


# =============================================================
# 13. Template 12: Eligibility matches real DB data
# =============================================================
print("\n=== 13. Template 12: Eligibility ===")
t12_count = 0
t12_mismatch = 0
for seed in range(200):
    sc = generate_scenario(seed)
    if sc.template_id == 12:
        t12_count += 1
        eligible = sc.key_facts["eligible_ids"]
        ineligible = sc.key_facts["ineligible_ids"]
        # Verify that we have both eligible and ineligible (or at least valid classification)
        has_mix = len(eligible) > 0 and len(ineligible) > 0
        if not has_mix and (len(eligible) + len(ineligible)) > 2:
            t12_mismatch += 1

check(f"T12 has natural eligibility mix (found {t12_count} scenarios)",
      t12_mismatch == 0, f"{t12_mismatch} scenarios had no mix")


# =============================================================
# 14. ###STOP### pending — agent gets one more turn
# =============================================================
print("\n=== 14. ###STOP### Pending Mechanism ===")
# Test the _pending_stop flag directly
for seed in range(200):
    sc = generate_scenario(seed)
    if sc.template_id == 1:
        game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT)
        game.reset(seed)
        # Simulate: set _pending_stop = True (as if user said ###STOP###)
        game._pending_stop = True
        # Agent does a tool call — should finalize AFTER this action
        res_id = sc.key_facts["res_id"]
        game.step(json.dumps({"name": "get_reservation_details", "arguments": {"reservation_id": res_id}}))
        check("Pending stop finalizes after agent action", game.done,
              f"done={game.done}")
        break


# =============================================================
# 15. check_communicate_info unit tests
# =============================================================
print("\n=== 15. COMMUNICATE Info Matching ===")
# Basic match
result = check_communicate_info(
    ["economy"],
    [{"role": "assistant", "text": "Your economy class reservation..."}],
)
check("'economy' found in text", result == 1.0, f"got {result}")

# Case insensitive
result2 = check_communicate_info(
    ["Economy"],
    [{"role": "assistant", "text": "your economy class reservation..."}],
)
check("Case insensitive match", result2 == 1.0, f"got {result2}")

# Comma removal
result3 = check_communicate_info(
    ["1234.56"],
    [{"role": "assistant", "text": "The total is $1,234.56."}],
)
check("Comma-stripped match ($1,234.56)", result3 == 1.0, f"got {result3}")

# Missing info
result4 = check_communicate_info(
    ["specific_info_not_present"],
    [{"role": "assistant", "text": "Hello, how can I help?"}],
)
check("Missing info returns 0.0", result4 == 0.0, f"got {result4}")


# =============================================================
# 16. extract_action parsing
# =============================================================
print("\n=== 16. extract_action Parsing ===")
cases = [
    ('{"name": "cancel_reservation", "arguments": {"reservation_id": "ABC"}}', True),
    ('Let me check. {"name": "get_user_details", "arguments": {"user_id": "u1"}}', True),
    ('I cannot help with that.', False),
    ('{"name": "respond_to_user", "arguments": {"message": "Hi"}} done', True),
    ('', False),
]
for text, should_parse in cases:
    result = extract_action(text, [])
    parsed = result is not None
    check(f"Parse '{text[:50]}...' → {should_parse}", parsed == should_parse)


# =============================================================
# 17. Smoke test: all 200 seeds generate without error
# =============================================================
print("\n=== 17. Scenario Generation Smoke Test (200 seeds) ===")
errors = []
for seed in range(200):
    try:
        sc = generate_scenario(seed)
        assert sc.template_id in range(1, 13), f"Bad template ID: {sc.template_id}"
        assert sc.domain in ("airline", "retail"), f"Bad domain: {sc.domain}"
        assert sc.user_system_prompt, "Empty user_system_prompt"
        assert sc.initial_message, "Empty initial_message"
    except Exception as e:
        errors.append(f"Seed {seed}: {type(e).__name__}: {e}")

if errors:
    for e in errors[:5]:
        print(f"    ERROR: {e}")
check(f"All 200 seeds generate without error", len(errors) == 0,
      f"{len(errors)} errors")


# =============================================================
# 18. Game reset and observe (no LLM call)
# =============================================================
print("\n=== 18. Game Reset & Observe ===")
for seed in range(200):
    sc = generate_scenario(seed)
    if sc.template_id == 1:
        game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT)
        game.reset(seed)
        obs = game.observe(0)
        check("Observation has POLICY section", "POLICY" in obs)
        check("Observation has TOOLS section", "TOOLS" in obs)
        check("Observation has CONVERSATION section", "CONVERSATION" in obs)
        check("Observation has initial user message", "[USER]:" in obs)
        break


# =============================================================
# 19. get_messages / get_system_prompt / get_tool_schemas
# =============================================================
print("\n=== 19. Structured API ===")
for seed in range(200):
    sc = generate_scenario(seed)
    if sc.template_id == 1:
        game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT)
        game.reset(seed)
        sp = game.get_system_prompt()
        check("System prompt has <policy>", "<policy>" in sp)
        check("System prompt has <instructions>", "<instructions>" in sp)
        schemas = game.get_tool_schemas()
        check("Tool schemas non-empty", len(schemas) > 0, f"got {len(schemas)}")
        msgs = game.get_messages()
        check("Initial messages has user message", len(msgs) > 0 and msgs[0]["role"] == "user")
        break


# =============================================================
# 20. Full integration: one episode with LLM user (Template 1)
# =============================================================
print("\n=== 20. Full Integration: Template 1 with LLM User ===")
for seed in range(200):
    sc = generate_scenario(seed)
    if sc.template_id == 1:
        game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT)
        game.reset(seed)
        res_id = sc.key_facts["res_id"]
        cabin = sc.key_facts["cabin"]
        uid = next(r["arguments"]["user_id"] for r in sc.ground_truth.required_actions
                   if r["name"] == "get_user_details")

        # Agent: lookup user
        game.step(json.dumps({"name": "get_user_details", "arguments": {"user_id": uid}}))
        game.step(json.dumps({"name": "get_reservation_details", "arguments": {"reservation_id": res_id}}))
        # Agent: refuse with correct info (uses respond_to_user to trigger LLM user)
        game.step(json.dumps({"name": "respond_to_user", "arguments": {
            "message": f"I've checked your reservation {res_id}. It is a {cabin.replace('_', ' ')} "
                       f"class booking without travel insurance, booked more than 24 hours ago. "
                       f"Unfortunately, this reservation is not eligible for cancellation under our policy."
        }}))
        # Agent: keep refusing
        if not game.done:
            game.step(json.dumps({"name": "respond_to_user", "arguments": {
                "message": "I understand your frustration, but I cannot override the cancellation policy."
            }}))
        # End
        if not game.done:
            game.step(json.dumps({"name": "end_conversation", "arguments": {
                "closing_message": f"Your {cabin.replace('_', ' ')} reservation is not eligible for cancellation."
            }}))
        assert game.done, "Game should be done"
        s = game.get_summary()
        print(f"    seed={seed}, reward={s['reward']:.2f}, reason={s['reason']}")
        print(f"    steps={s['steps']}, conv_len={s['conversation_length']}")
        check(f"T1 full integration reward >= 0.8",
              s["reward"] >= 0.8, f"reward={s['reward']}")
        break


# =============================================================
# 21. Full integration: Template 11 with LLM user
# =============================================================
print("\n=== 21. Full Integration: Template 11 with LLM User ===")
for seed in range(200):
    sc = generate_scenario(seed)
    if sc.template_id == 11 and "cancel" in sc.key_facts.get("action_type", ""):
        game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT)
        game.reset(seed)
        res_id = sc.key_facts["res_id"]
        action_type = sc.key_facts["action_type"]
        print(f"    action_type={action_type}, seed={seed}")

        # Agent: perform the valid cancellation
        game.step(json.dumps({"name": "cancel_reservation", "arguments": {"reservation_id": res_id}}))
        game.step(json.dumps({"name": "respond_to_user", "arguments": {
            "message": f"I've cancelled reservation {res_id} as requested."
        }}))
        if not game.done:
            game.step(json.dumps({"name": "end_conversation", "arguments": {
                "closing_message": "Your reservation has been cancelled."
            }}))
        assert game.done
        s = game.get_summary()
        print(f"    reward={s['reward']:.2f}, reason={s['reason']}")
        check(f"T11 full integration reward >= 0.8",
              s["reward"] >= 0.8, f"reward={s['reward']}")
        break


# =============================================================
# 22. Over-transfer: blind transfer (no lookups, no actions) → 0.0
# =============================================================
print("\n=== 22. Over-transfer: Blind Transfer → 0.0 ===")
gt_no_transfer = GroundTruth(
    correct_behavior="Agent should refuse, not transfer.",
    should_transfer=False,
)
reward_blind, reason_blind = compute_reward(
    tool_calls=[],
    ground_truth=gt_no_transfer,
    db_final={},
    transferred=True,
    conversation_ended_normally=True,
    conversation=[],
)
check("Blind transfer → 0.0", reward_blind == 0.0,
      f"got {reward_blind}, reason: {reason_blind}")


# =============================================================
# 23. Over-transfer: 1 unique lookup → small partial credit
# =============================================================
print("\n=== 23. Over-transfer: 1 Unique Lookup ===")
reward_1lu, reason_1lu = compute_reward(
    tool_calls=[{"name": "get_user_details", "arguments": {"user_id": "u1"}}],
    ground_truth=gt_no_transfer,
    db_final={},
    transferred=True,
    conversation_ended_normally=True,
    conversation=[],
)
# 1 unique lookup: min(1/3, 1.0) * 0.3 = 0.1 → final = 0.8 * 0.1 + 0.2 * 1.0 = 0.28
expected_1lu = 0.8 * (1.0/3.0 * 0.3) + 0.2 * 1.0
check(f"1 unique lookup → ~{expected_1lu:.3f}", abs(reward_1lu - expected_1lu) < 0.01,
      f"got {reward_1lu}, reason: {reason_1lu}")


# =============================================================
# 24. Over-transfer: 3 unique lookups → max lookup credit
# =============================================================
print("\n=== 24. Over-transfer: 3 Unique Lookups ===")
reward_3lu, reason_3lu = compute_reward(
    tool_calls=[
        {"name": "get_user_details", "arguments": {"user_id": "u1"}},
        {"name": "get_reservation_details", "arguments": {"reservation_id": "r1"}},
        {"name": "search_direct_flight", "arguments": {"origin": "JFK", "destination": "LAX"}},
    ],
    ground_truth=gt_no_transfer,
    db_final={},
    transferred=True,
    conversation_ended_normally=True,
    conversation=[],
)
expected_3lu = 0.8 * 0.3 + 0.2 * 1.0  # 0.44
check(f"3 unique lookups → ~{expected_3lu:.3f}", abs(reward_3lu - expected_3lu) < 0.01,
      f"got {reward_3lu}, reason: {reason_3lu}")


# =============================================================
# 25. ANTI-HACK: Spamming SAME tool 3x counts as 1 unique
# =============================================================
print("\n=== 25. Anti-Hack: Same Tool Spam ===")
reward_spam, reason_spam = compute_reward(
    tool_calls=[
        {"name": "get_user_details", "arguments": {"user_id": "u1"}},
        {"name": "get_user_details", "arguments": {"user_id": "u1"}},
        {"name": "get_user_details", "arguments": {"user_id": "u1"}},
    ],
    ground_truth=gt_no_transfer,
    db_final={},
    transferred=True,
    conversation_ended_normally=True,
    conversation=[],
)
# Only 1 unique tool name → same as 1 lookup, NOT 3
check("3x same tool = 1 unique lookup (anti-spam)",
      abs(reward_spam - expected_1lu) < 0.01,
      f"got {reward_spam} (expected {expected_1lu:.3f}), reason: {reason_spam}")
check("Spam reward < 3 unique reward",
      reward_spam < reward_3lu,
      f"{reward_spam} vs {reward_3lu}")


# =============================================================
# 26. Over-transfer: 5 unique lookups capped at 0.3
# =============================================================
print("\n=== 26. Over-transfer: 5 Unique Lookups (Capped) ===")
reward_5lu, _ = compute_reward(
    tool_calls=[
        {"name": "get_user_details", "arguments": {}},
        {"name": "get_reservation_details", "arguments": {}},
        {"name": "search_direct_flight", "arguments": {}},
        {"name": "search_onestop_flight", "arguments": {}},
        {"name": "list_all_airports", "arguments": {}},
    ],
    ground_truth=gt_no_transfer,
    db_final={},
    transferred=True,
    conversation_ended_normally=True,
    conversation=[],
)
check("5 unique lookups same as 3 (capped at 0.3)",
      abs(reward_5lu - expected_3lu) < 0.01,
      f"got {reward_5lu}")


# =============================================================
# 27. Over-transfer: completed required action → higher credit
# =============================================================
print("\n=== 27. Over-transfer: Completed Required Action ===")
gt_req = GroundTruth(
    required_actions=[
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r1"}},
    ],
    should_transfer=False,
    correct_behavior="Should cancel, not transfer.",
)
# Model does the correct cancel AND transfers
reward_req_done, reason_req_done = compute_reward(
    tool_calls=[
        {"name": "get_user_details", "arguments": {"user_id": "u1"}},
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r1"}},
    ],
    ground_truth=gt_req,
    db_final={},
    transferred=True,
    conversation_ended_normally=True,
    conversation=[],
)
# req_credit = (1/1) * 0.5 = 0.5, lookup_credit = 1/3 * 0.3 = 0.1
# db_reward = max(0.5, 0.1) = 0.5 → final = 0.8 * 0.5 + 0.2 * 1.0 = 0.6
expected_req_done = 0.8 * 0.5 + 0.2 * 1.0
check(f"Completed req + transfer → ~{expected_req_done:.3f}",
      abs(reward_req_done - expected_req_done) < 0.01,
      f"got {reward_req_done}, reason: {reason_req_done}")

# Without transfer: full credit
reward_req_correct, _ = compute_reward(
    tool_calls=[
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r1"}},
    ],
    ground_truth=gt_req,
    db_final={},
    transferred=False,
    conversation_ended_normally=True,
    conversation=[],
)
check("Completed req WITHOUT transfer → 1.0",
      abs(reward_req_correct - 1.0) < 0.01,
      f"got {reward_req_correct}")
check("req+transfer < req_only (transfer always penalized)",
      reward_req_done < reward_req_correct,
      f"{reward_req_done} vs {reward_req_correct}")


# =============================================================
# 28. Over-transfer: partial required actions → proportional credit
# =============================================================
print("\n=== 28. Over-transfer: Partial Required Actions ===")
gt_req2 = GroundTruth(
    required_actions=[
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r1"}},
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r2"}},
    ],
    should_transfer=False,
    correct_behavior="Should cancel both.",
)
# Model cancels one of two, then transfers
reward_partial_req, _ = compute_reward(
    tool_calls=[
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r1"}},
    ],
    ground_truth=gt_req2,
    db_final={},
    transferred=True,
    conversation_ended_normally=True,
    conversation=[],
)
# req_credit = (1/2) * 0.5 = 0.25. lookup_credit = 0 (cancel is not a read op).
# db_reward = max(0.25, 0.0) = 0.25 → final = 0.8 * 0.25 + 0.2 * 1.0 = 0.4
expected_partial_req = 0.8 * 0.25 + 0.2 * 1.0
check(f"1/2 req + transfer → ~{expected_partial_req:.3f}",
      abs(reward_partial_req - expected_partial_req) < 0.01,
      f"got {reward_partial_req}")


# =============================================================
# 29. Over-transfer: no lookups, no req match → 0.0
# =============================================================
print("\n=== 29. Over-transfer: Required but Not Matched → 0.0 ===")
reward_no_match, reason_no_match = compute_reward(
    tool_calls=[],
    ground_truth=gt_req,
    db_final={},
    transferred=True,
    conversation_ended_normally=True,
    conversation=[],
)
check("Over-transfer, req not matched, no lookups → 0.0",
      reward_no_match == 0.0,
      f"got {reward_no_match}, reason: {reason_no_match}")


# =============================================================
# 30. Full reward gradient ordering
# =============================================================
print("\n=== 30. Full Reward Gradient ===")
reward_correct_refusal, _ = compute_reward(
    tool_calls=[],
    ground_truth=gt_no_transfer,
    db_final={},
    transferred=False,
    conversation_ended_normally=True,
    conversation=[],
)
check("Gradient: blind(0.0) < 1_lookup", reward_blind < reward_1lu,
      f"{reward_blind} vs {reward_1lu}")
check("Gradient: 1_lookup < 3_lookups", reward_1lu < reward_3lu,
      f"{reward_1lu} vs {reward_3lu}")
check("Gradient: 3_lookups < req_done+transfer",
      reward_3lu < reward_req_done,
      f"{reward_3lu} vs {reward_req_done}")
check("Gradient: req_done+transfer < correct",
      reward_req_done < reward_correct_refusal,
      f"{reward_req_done} vs {reward_correct_refusal}")
check("Correct refusal = 1.0",
      abs(reward_correct_refusal - 1.0) < 0.01,
      f"got {reward_correct_refusal}")


# =============================================================
# 31. Forbidden action ALWAYS overrides over-transfer credit
# =============================================================
print("\n=== 31. Forbidden Overrides All Partial Credit ===")
gt_forbidden = GroundTruth(
    forbidden_actions=[
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r1"}},
    ],
    should_transfer=False,
    correct_behavior="Should not cancel or transfer.",
)
# Forbidden + lookups + transfer
reward_fb1, _ = compute_reward(
    tool_calls=[
        {"name": "get_user_details", "arguments": {"user_id": "u1"}},
        {"name": "get_reservation_details", "arguments": {"reservation_id": "r1"}},
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r1"}},
    ],
    ground_truth=gt_forbidden,
    db_final={},
    transferred=True,
    conversation_ended_normally=True,
    conversation=[],
)
check("Forbidden + lookups + transfer → 0.0", reward_fb1 == 0.0)

# Forbidden + no transfer (forbidden still wins)
reward_fb2, _ = compute_reward(
    tool_calls=[
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r1"}},
    ],
    ground_truth=gt_forbidden,
    db_final={},
    transferred=False,
    conversation_ended_normally=True,
    conversation=[],
)
check("Forbidden without transfer → 0.0", reward_fb2 == 0.0)


# =============================================================
# 32. order_match check type (replaces any_valid)
# =============================================================
print("\n=== 32. order_match Check Type ===")
from adversarial_policy_game.verification import _any_matches_required

# Exact match
check("order_match: exact ID",
      _any_matches_required(
          [{"name": "cancel_pending_order", "arguments": {"order_id": "#W123"}}],
          {"name": "cancel_pending_order", "check": "order_match", "order_id": "#W123"},
      ))

# Model strips "#" prefix (common error) → still matches
check("order_match: missing # normalized",
      _any_matches_required(
          [{"name": "cancel_pending_order", "arguments": {"order_id": "W123"}}],
          {"name": "cancel_pending_order", "check": "order_match", "order_id": "#W123"},
      ))

# Wrong order ID → no match
check("order_match: wrong order_id",
      not _any_matches_required(
          [{"name": "cancel_pending_order", "arguments": {"order_id": "#W999"}}],
          {"name": "cancel_pending_order", "check": "order_match", "order_id": "#W123"},
      ))

# Wrong tool name → no match
check("order_match: wrong tool name",
      not _any_matches_required(
          [{"name": "modify_pending_order_items", "arguments": {"order_id": "#W123"}}],
          {"name": "cancel_pending_order", "check": "order_match", "order_id": "#W123"},
      ))

# Empty args → no match
check("order_match: empty args",
      not _any_matches_required(
          [{"name": "cancel_pending_order", "arguments": {}}],
          {"name": "cancel_pending_order", "check": "order_match", "order_id": "#W123"},
      ))


# =============================================================
# 33. order_match in full reward: wrong order gets 0.0
# =============================================================
print("\n=== 33. order_match Prevents Wrong-Order Credit ===")
gt_order = GroundTruth(
    required_actions=[
        {"name": "cancel_pending_order", "check": "order_match", "order_id": "#W123"},
    ],
    correct_behavior="Cancel order #W123.",
)
# Correct order
reward_right_order, _ = compute_reward(
    tool_calls=[{"name": "cancel_pending_order", "arguments": {"order_id": "#W123"}}],
    ground_truth=gt_order,
    db_final={},
    transferred=False,
    conversation_ended_normally=True,
    conversation=[],
)
check("Correct order_id → 1.0", abs(reward_right_order - 1.0) < 0.01,
      f"got {reward_right_order}")

# Wrong order — should NOT match required action
reward_wrong_order, reason_wrong = compute_reward(
    tool_calls=[{"name": "cancel_pending_order", "arguments": {"order_id": "#W999"}}],
    ground_truth=gt_order,
    db_final={},
    transferred=False,
    conversation_ended_normally=True,
    conversation=[],
)
check("Wrong order_id → 0.0 (no required match)",
      reward_wrong_order == 0.0,
      f"got {reward_wrong_order}, reason: {reason_wrong}")


# =============================================================
# 34. Phase 5 write_ops: new write tools caught
# =============================================================
print("\n=== 34. Phase 5 Write Ops Completeness ===")
gt_refusal = GroundTruth(
    correct_behavior="Should refuse. No actions needed.",
    should_transfer=False,
)
new_write_ops = ["send_certificate", "update_reservation_passengers", "modify_pending_order_payment"]
for op_name in new_write_ops:
    r, reason = compute_reward(
        tool_calls=[{"name": op_name, "arguments": {}}],
        ground_truth=gt_refusal,
        db_final={},
        transferred=False,
        conversation_ended_normally=True,
        conversation=[],
    )
    check(f"Phase 5 catches {op_name}", r == 0.0,
          f"got {r}, reason: {reason}")


# =============================================================
# 35. ANTI-HACK: "think" tool does NOT get lookup credit
# =============================================================
print("\n=== 35. Anti-Hack: 'think' Not a Valid Lookup ===")
reward_think, _ = compute_reward(
    tool_calls=[
        {"name": "think", "arguments": {"thought": "spam"}},
        {"name": "think", "arguments": {"thought": "more spam"}},
        {"name": "think", "arguments": {"thought": "even more"}},
    ],
    ground_truth=gt_no_transfer,
    db_final={},
    transferred=True,
    conversation_ended_normally=True,
    conversation=[],
)
check("'think' tool spam → 0.0 (not a valid lookup)", reward_think == 0.0,
      f"got {reward_think}")


# =============================================================
# 36. ANTI-HACK: Non-existent tool doesn't get lookup credit
# =============================================================
print("\n=== 36. Anti-Hack: Fake Tool Name ===")
reward_fake, _ = compute_reward(
    tool_calls=[
        {"name": "fake_tool_1", "arguments": {}},
        {"name": "fake_tool_2", "arguments": {}},
        {"name": "fake_tool_3", "arguments": {}},
    ],
    ground_truth=gt_no_transfer,
    db_final={},
    transferred=True,
    conversation_ended_normally=True,
    conversation=[],
)
check("Fake tool names → 0.0", reward_fake == 0.0,
      f"got {reward_fake}")


# =============================================================
# 37. T7/T9/T10 scenarios now use order_match (no more any_valid)
# =============================================================
print("\n=== 37. any_valid Removed From Scenarios ===")
any_valid_found = False
for seed in range(200):
    sc = generate_scenario(seed)
    for req in sc.ground_truth.required_actions:
        if req.get("check") == "any_valid":
            any_valid_found = True
            print(f"    FOUND any_valid in T{sc.template_id} seed={seed}: {req}")
            break
    if any_valid_found:
        break
check("No scenario uses any_valid", not any_valid_found)


# =============================================================
# 38. T10 order_match: correct order gets credit
# =============================================================
print("\n=== 38. T10 order_match Validation ===")
for seed in range(200):
    sc = generate_scenario(seed)
    if sc.template_id == 10:
        order_id = sc.key_facts["order_id"]
        req = sc.ground_truth.required_actions[0]
        check("T10 uses order_match", req.get("check") == "order_match",
              f"check={req.get('check')}")
        check("T10 order_id matches", req.get("order_id") == order_id,
              f"req_oid={req.get('order_id')}, sc_oid={order_id}")
        # Verify correct order works
        r, _ = compute_reward(
            tool_calls=[{"name": "modify_pending_order_items",
                         "arguments": {"order_id": order_id, "item_ids": ["x"], "new_item_ids": ["y"],
                                       "payment_method_id": "p1"}}],
            ground_truth=sc.ground_truth,
            db_final={},
            transferred=False,
            conversation_ended_normally=True,
            conversation=[{"role": "assistant", "text": f"${sc.key_facts['cheapest']}"}],
        )
        check(f"T10 correct order_id → reward ≥ 0.8", r >= 0.8, f"got {r}")
        break


# =============================================================
# 39. Retail lookups: find_user_id_by_email gets credit
# =============================================================
print("\n=== 39. Retail Lookup Credit ===")
gt_retail_refuse = GroundTruth(
    correct_behavior="Agent should refuse, not transfer.",
    should_transfer=False,
)
reward_retail_lu, _ = compute_reward(
    tool_calls=[
        {"name": "find_user_id_by_email", "arguments": {"email": "a@b.com"}},
        {"name": "get_order_details", "arguments": {"order_id": "#W123"}},
    ],
    ground_truth=gt_retail_refuse,
    db_final={},
    transferred=True,
    conversation_ended_normally=True,
    conversation=[],
)
# 2 unique lookups: min(2/3, 1.0) * 0.3 = 0.2 → final = 0.8 * 0.2 + 0.2 * 1.0 = 0.36
expected_retail_lu = 0.8 * (2.0/3.0 * 0.3) + 0.2 * 1.0
check(f"Retail lookups (email+order) → ~{expected_retail_lu:.3f}",
      abs(reward_retail_lu - expected_retail_lu) < 0.01,
      f"got {reward_retail_lu}")


# =============================================================
# 40. Mixed: lookups + partial req → takes max
# =============================================================
print("\n=== 40. Mixed: Lookups vs Req Credit (Takes Max) ===")
gt_mixed = GroundTruth(
    required_actions=[
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r1"}},
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r2"}},
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r3"}},
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r4"}},
    ],
    should_transfer=False,
    correct_behavior="Cancel all four.",
)
# 1/4 required = 0.5/4 = 0.125 req_credit. 3 unique lookups = 0.3 lookup_credit.
# max(0.125, 0.3) = 0.3 → lookup credit wins
reward_mixed, reason_mixed = compute_reward(
    tool_calls=[
        {"name": "get_user_details", "arguments": {}},
        {"name": "get_reservation_details", "arguments": {}},
        {"name": "search_direct_flight", "arguments": {}},
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r1"}},
    ],
    ground_truth=gt_mixed,
    db_final={},
    transferred=True,
    conversation_ended_normally=True,
    conversation=[],
)
expected_mixed = 0.8 * 0.3 + 0.2 * 1.0  # lookup wins
check(f"Mixed: lookup_credit (0.3) > req_credit (0.125) → ~{expected_mixed:.3f}",
      abs(reward_mixed - expected_mixed) < 0.01,
      f"got {reward_mixed}, reason: {reason_mixed}")

# Now 3/4 required = 0.375 req_credit. 1 unique lookup = 0.1 lookup_credit.
# max(0.375, 0.1) = 0.375 → req credit wins
reward_mixed2, _ = compute_reward(
    tool_calls=[
        {"name": "get_user_details", "arguments": {}},
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r1"}},
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r2"}},
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r3"}},
    ],
    ground_truth=gt_mixed,
    db_final={},
    transferred=True,
    conversation_ended_normally=True,
    conversation=[],
)
expected_mixed2 = 0.8 * 0.375 + 0.2 * 1.0  # req wins
check(f"Mixed: req_credit (0.375) > lookup_credit (0.1) → ~{expected_mixed2:.3f}",
      abs(reward_mixed2 - expected_mixed2) < 0.01,
      f"got {reward_mixed2}")


# =============================================================
# Summary
# =============================================================
print(f"\n{'='*60}")
print(f"RESULTS: {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'='*60}")
if failed > 0:
    print("SOME TESTS FAILED")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
