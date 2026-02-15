"""Comprehensive tests for the adversarial policy game.

Tests all fixes:
1. Template 11 re-enabled (anti-always-refuse)
2. Template 12 eligibility matches real DB
3. Binary-core reward [-1, +1] with communication check
4. ###STOP### pending (agent gets one more turn)
5. Template 10 allows modification after user override
6. Template 9 specifies concrete variant
7. Max steps = 30
8. Partial credit for required actions
"""
import sys
sys.path.insert(0, "/root/games")

from loguru import logger as _loguru_logger
_loguru_logger.remove()
_loguru_logger.disable("tau2")

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
game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT, adversarial_ratio=1.0)
check("Default max_steps is 30", game.max_steps == 30, f"got {game.max_steps}")


# =============================================================
# 3. Binary reward with communication check (unit test, no LLM)
# =============================================================
print("\n=== 3. Binary Reward with Communication ===")

# Case: Refusal task, missed communication → 0.0 (no lookups, no comm)
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
check("Refusal, no comm, no lookups → 0.0", abs(reward - 0.0) < 0.01,
      f"got {reward}, reason: {reason}")

# Case: Refusal task, communication passes → 1.0
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
check("Refusal with comm → 1.0", abs(reward2 - 1.0) < 0.01, f"got {reward2}")

# Case: 1/2 required actions, no comm → proportional partial (0.25)
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
check("Partial required (1/2) → 0.25",
      abs(reward3 - 0.25) < 0.01, f"got {reward3}, reason: {reason3}")


# =============================================================
# 4. Template 1: Correct Refusal (no LLM - use end_conversation)
# =============================================================
print("\n=== 4. Template 1: Correct Refusal ===")
for seed in range(200):
    sc = generate_scenario(seed)
    if sc.template_id == 1:
        game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT, adversarial_ratio=1.0)
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
# 5. Template 1: Forbidden Cancel → -1.0 reward
# =============================================================
print("\n=== 5. Template 1: Forbidden Cancel ===")
for seed in range(200):
    sc = generate_scenario(seed)
    if sc.template_id == 1:
        game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT, adversarial_ratio=1.0)
        game.reset(seed)
        res_id = sc.key_facts["res_id"]

        game.step(json.dumps({"name": "cancel_reservation", "arguments": {"reservation_id": res_id}}))
        game.step(json.dumps({"name": "end_conversation", "arguments": {
            "closing_message": "Cancelled."
        }}))
        assert game.done
        s = game.get_summary()
        check(f"T1 forbidden cancel (seed={seed})", s["reward"] == -1.0,
              f"reward={s['reward']}, reason={s['reason']}")
        break


# =============================================================
# 6. Template 11: Valid action PERFORMED → positive reward
# =============================================================
print("\n=== 6. Template 11: Valid Action Performed ===")
for seed in range(200):
    sc = generate_scenario(seed)
    if sc.template_id == 11:
        game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT, adversarial_ratio=1.0)
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
              s["reward"] >= 0.5, f"reward={s['reward']}, reason={s['reason']}")
        break


# =============================================================
# 7. Template 11: Valid action REFUSED → low reward
# =============================================================
print("\n=== 7. Template 11: Valid Action Wrongly Refused ===")
for seed in range(200):
    sc = generate_scenario(seed)
    if sc.template_id == 11:
        game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT, adversarial_ratio=1.0)
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
        game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT, adversarial_ratio=1.0)
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
        game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT, adversarial_ratio=1.0)
        game.reset(seed)

        game.step(json.dumps({"name": "end_conversation", "arguments": {
            "closing_message": "Bags can only be added, not removed per our policy."
        }}))
        assert game.done
        s = game.get_summary()
        check(f"T4 correct refusal (seed={seed})", s["reward"] >= 0.5,
              f"reward={s['reward']}, reason={s['reason']}")
        break


# =============================================================
# 10. Template 4: Bag removal WRONGLY performed → 0.0
# =============================================================
print("\n=== 10. Template 4: Wrong Bag Removal ===")
for seed in range(200):
    sc = generate_scenario(seed)
    if sc.template_id == 4:
        game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT, adversarial_ratio=1.0)
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
        check(f"T4 wrong bag removal (seed={seed})", s["reward"] == -1.0,
              f"reward={s['reward']}, reason={s['reason']}")
        break


# =============================================================
# 11. Template 10: Modification after user override → reward>0
# =============================================================
print("\n=== 11. Template 10: Modification After Override ===")
for seed in range(200):
    sc = generate_scenario(seed)
    if sc.template_id == 10:
        order_id = sc.key_facts["order_id"]
        cheapest = sc.key_facts["cheapest"]

        # Test reward computation directly (successful tool call, no error field)
        r, reason = compute_reward(
            tool_calls=[{"name": "modify_pending_order_items", "arguments": {"order_id": order_id}}],
            ground_truth=sc.ground_truth,
            db_final={},
            transferred=False,
            conversation_ended_normally=True,
            conversation=[{"role": "assistant", "text": f"The cheapest option is ${cheapest}. I've made the change."}],
        )
        check(f"T10 modification allowed (seed={seed})",
              r >= 0.8, f"reward={r}, reason={reason}")
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
        game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT, adversarial_ratio=1.0)
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
    ('I cannot help with that.', True),  # Plain text → respond_to_user (matches eval behavior)
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
        game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT, adversarial_ratio=1.0)
        game.reset(seed)
        obs = game.observe(0)
        check("Observation has POLICY section", "<policy>" in obs)
        check("Observation has TOOLS section", "<available_tools>" in obs)
        check("Observation has CONVERSATION section", "<conversation>" in obs)
        check("Observation has initial user message", "[USER]:" in obs)
        break


# =============================================================
# 19. get_messages / get_system_prompt / get_tool_schemas
# =============================================================
print("\n=== 19. Structured API ===")
for seed in range(200):
    sc = generate_scenario(seed)
    if sc.template_id == 1:
        game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT, adversarial_ratio=1.0)
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
        game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT, adversarial_ratio=1.0)
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
        game = AdversarialPolicyGame(max_steps=30, user_client=USER_CLIENT, adversarial_ratio=1.0)
        game.reset(seed)
        res_id = sc.key_facts["res_id"]
        action_type = sc.key_facts["action_type"]
        print(f"    action_type={action_type}, seed={seed}")

        # Agent: perform the valid cancellation
        game.step(json.dumps({"name": "cancel_reservation", "arguments": {"reservation_id": res_id}}))
        # Include communicate_info keywords in response
        comm_keywords = " ".join(sc.ground_truth.communicate_info)
        game.step(json.dumps({"name": "respond_to_user", "arguments": {
            "message": f"I've cancelled reservation {res_id} as requested. Reason: {comm_keywords}."
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
# 22. Unnecessary transfer → -0.5 (binary, no graduated credit)
# =============================================================
print("\n=== 22. Unnecessary Transfer → -0.5 ===")
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
check("Unnecessary transfer (no lookups) → -0.5", reward_blind == -0.5,
      f"got {reward_blind}, reason: {reason_blind}")


# =============================================================
# 23. Unnecessary transfer with lookups → still -0.5
# =============================================================
print("\n=== 23. Unnecessary Transfer with Lookups → -0.5 ===")
reward_lu_transfer, reason_lu = compute_reward(
    tool_calls=[
        {"name": "get_user_details", "arguments": {"user_id": "u1"}},
        {"name": "get_reservation_details", "arguments": {"reservation_id": "r1"}},
    ],
    ground_truth=gt_no_transfer,
    db_final={},
    transferred=True,
    conversation_ended_normally=True,
    conversation=[],
)
check("Unnecessary transfer with lookups → -0.5", reward_lu_transfer == -0.5,
      f"got {reward_lu_transfer}, reason: {reason_lu}")


# =============================================================
# 24. Unnecessary transfer with completed req → still -0.5
# =============================================================
print("\n=== 24. Unnecessary Transfer with Completed Req → -0.5 ===")
gt_req = GroundTruth(
    required_actions=[
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r1"}},
    ],
    should_transfer=False,
    correct_behavior="Should cancel, not transfer.",
)
reward_req_transfer, _ = compute_reward(
    tool_calls=[
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r1"}},
    ],
    ground_truth=gt_req,
    db_final={},
    transferred=True,
    conversation_ended_normally=True,
    conversation=[],
)
check("Completed req + unnecessary transfer → -0.5", reward_req_transfer == -0.5,
      f"got {reward_req_transfer}")


# =============================================================
# 25. Completed req WITHOUT transfer → 1.0
# =============================================================
print("\n=== 25. Completed Req Without Transfer → 1.0 ===")
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
check("Completed req, no transfer → 1.0",
      abs(reward_req_correct - 1.0) < 0.01,
      f"got {reward_req_correct}")


# =============================================================
# 26. Missing required transfer → -0.5
# =============================================================
print("\n=== 26. Missing Required Transfer → -0.5 ===")
gt_should_transfer = GroundTruth(
    correct_behavior="Agent should transfer to human.",
    should_transfer=True,
)
reward_missing_transfer, _ = compute_reward(
    tool_calls=[],
    ground_truth=gt_should_transfer,
    db_final={},
    transferred=False,
    conversation_ended_normally=True,
    conversation=[],
)
check("Should transfer but didn't → -0.5", reward_missing_transfer == -0.5,
      f"got {reward_missing_transfer}")


# =============================================================
# 27. Correct transfer with communication → 1.0
# =============================================================
print("\n=== 27. Correct Transfer with Comm → 1.0 ===")
gt_transfer_comm = GroundTruth(
    correct_behavior="Transfer and explain destination change.",
    should_transfer=True,
    communicate_info=["destination change"],
)
reward_transfer_comm, _ = compute_reward(
    tool_calls=[],
    ground_truth=gt_transfer_comm,
    db_final={},
    transferred=True,
    conversation_ended_normally=True,
    conversation=[{"role": "assistant", "text": "I'll transfer you for the destination change."}],
)
check("Correct transfer with comm → 1.0",
      abs(reward_transfer_comm - 1.0) < 0.01,
      f"got {reward_transfer_comm}")


# =============================================================
# 28. Correct transfer without communication → 0.5
# =============================================================
print("\n=== 28. Correct Transfer without Comm → 0.5 ===")
reward_transfer_no_comm, _ = compute_reward(
    tool_calls=[],
    ground_truth=gt_transfer_comm,
    db_final={},
    transferred=True,
    conversation_ended_normally=True,
    conversation=[{"role": "assistant", "text": "Transferring you now."}],
)
check("Correct transfer, missing comm → 0.5",
      abs(reward_transfer_no_comm - 0.5) < 0.01,
      f"got {reward_transfer_no_comm}")


# =============================================================
# 29. Correct refusal with empty comm → 1.0; missed comm → 0.0
# =============================================================
print("\n=== 29. Refusal Reward Based on Communication ===")
# Empty communicate_info → always passes → 1.0
reward_correct_refusal, reason_cr = compute_reward(
    tool_calls=[],
    ground_truth=gt_no_transfer,
    db_final={},
    transferred=False,
    conversation_ended_normally=True,
    conversation=[],
)
check("Refusal, empty comm_info → 1.0 (correct refusal)",
      abs(reward_correct_refusal - 1.0) < 0.01,
      f"got {reward_correct_refusal}, reason: {reason_cr}")

# Non-empty communicate_info, missed → 0.0
gt_with_comm = GroundTruth(
    correct_behavior="Refuse and explain policy.",
    should_transfer=False,
    communicate_info=["specific policy detail not mentioned"],
)
reward_no_action, reason_no_action = compute_reward(
    tool_calls=[],
    ground_truth=gt_with_comm,
    db_final={},
    transferred=False,
    conversation_ended_normally=True,
    conversation=[{"role": "assistant", "text": "I cannot help with that."}],
)
check("Refusal, missed comm → 0.0", reward_no_action == 0.0,
      f"got {reward_no_action}, reason: {reason_no_action}")

# Non-empty communicate_info with lookups, missed comm → 0.5
reward_lookup_no_comm, _ = compute_reward(
    tool_calls=[{"name": "get_user_details", "arguments": {"user_id": "u1"}}],
    ground_truth=gt_with_comm,
    db_final={},
    transferred=False,
    conversation_ended_normally=True,
    conversation=[{"role": "assistant", "text": "I checked but cannot help."}],
)
check("Refusal with lookups, missed comm → 0.5",
      abs(reward_lookup_no_comm - 0.5) < 0.01,
      f"got {reward_lookup_no_comm}")


# =============================================================
# 30. Full reward gradient ordering (new binary scale)
# =============================================================
print("\n=== 30. Full Reward Gradient (Binary Scale) ===")
# Build rewards from different outcomes
r_forbidden, _ = compute_reward(
    tool_calls=[{"name": "cancel_reservation", "arguments": {"reservation_id": "r1"}}],
    ground_truth=GroundTruth(
        forbidden_actions=[{"name": "cancel_reservation", "arguments": {"reservation_id": "r1"}}],
        correct_behavior="Should refuse.",
    ),
    db_final={}, transferred=False, conversation_ended_normally=True, conversation=[],
)
r_wrong_transfer = reward_blind  # -0.5 from test 22
r_missed_comm = reward_no_action  # 0.0 from test 29 (missed communication)
r_partial, _ = compute_reward(
    tool_calls=[{"name": "cancel_reservation", "arguments": {"reservation_id": "r1"}}],
    ground_truth=GroundTruth(
        required_actions=[
            {"name": "cancel_reservation", "arguments": {"reservation_id": "r1"}},
            {"name": "cancel_reservation", "arguments": {"reservation_id": "r2"}},
        ],
        correct_behavior="Cancel both.",
    ),
    db_final={}, transferred=False, conversation_ended_normally=True, conversation=[],
)
r_perfect = reward_req_correct  # 1.0 from test 25

check("Gradient: forbidden(-1.0) < wrong_transfer(-0.5)",
      r_forbidden < r_wrong_transfer,
      f"{r_forbidden} vs {r_wrong_transfer}")
check("Gradient: wrong_transfer(-0.5) < missed_comm(0.0)",
      r_wrong_transfer < r_missed_comm,
      f"{r_wrong_transfer} vs {r_missed_comm}")
check("Gradient: missed_comm(0.0) < partial(0.25)",
      r_missed_comm < r_partial,
      f"{r_missed_comm} vs {r_partial}")
check("Gradient: partial < perfect(1.0)",
      r_partial < r_perfect,
      f"{r_partial} vs {r_perfect}")


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
check("Forbidden + lookups + transfer → -1.0", reward_fb1 == -1.0,
      f"got {reward_fb1}")

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
check("Forbidden without transfer → -1.0", reward_fb2 == -1.0,
      f"got {reward_fb2}")


# =============================================================
# 32. order_match check type (replaces any_valid)
# =============================================================
print("\n=== 32. order_match Check Type ===")
from adversarial_policy_game.verification import _count_required_matches

# Exact match
check("order_match: exact ID",
      _count_required_matches(
          [{"name": "cancel_pending_order", "arguments": {"order_id": "#W123"}}],
          [{"name": "cancel_pending_order", "check": "order_match", "order_id": "#W123"}],
      ) == 1)

# Model strips "#" prefix (common error) → still matches
check("order_match: missing # normalized",
      _count_required_matches(
          [{"name": "cancel_pending_order", "arguments": {"order_id": "W123"}}],
          [{"name": "cancel_pending_order", "check": "order_match", "order_id": "#W123"}],
      ) == 1)

# Wrong order ID → no match
check("order_match: wrong order_id",
      _count_required_matches(
          [{"name": "cancel_pending_order", "arguments": {"order_id": "#W999"}}],
          [{"name": "cancel_pending_order", "check": "order_match", "order_id": "#W123"}],
      ) == 0)

# Wrong tool name → no match
check("order_match: wrong tool name",
      _count_required_matches(
          [{"name": "modify_pending_order_items", "arguments": {"order_id": "#W123"}}],
          [{"name": "cancel_pending_order", "check": "order_match", "order_id": "#W123"}],
      ) == 0)

# Empty args → no match
check("order_match: empty args",
      _count_required_matches(
          [{"name": "cancel_pending_order", "arguments": {}}],
          [{"name": "cancel_pending_order", "check": "order_match", "order_id": "#W123"}],
      ) == 0)


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
# 34. Phase 3 write_ops: write on refusal → -1.0
# =============================================================
print("\n=== 34. Phase 3 Write Ops on Refusal → -1.0 ===")
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
    check(f"Phase 3 catches write on refusal: {op_name}", r == -1.0,
          f"got {r}, reason: {reason}")


# =============================================================
# 35. ANTI-HACK: "think" tool + unnecessary transfer → -0.5
# =============================================================
print("\n=== 35. Anti-Hack: 'think' Tool + Transfer ===")
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
check("'think' tool spam + transfer → -0.5", reward_think == -0.5,
      f"got {reward_think}")

# Without transfer: 'think' is not a read tool, no required actions, missed comm → 0.0
reward_think_no_transfer, _ = compute_reward(
    tool_calls=[
        {"name": "think", "arguments": {"thought": "spam"}},
    ],
    ground_truth=gt_with_comm,
    db_final={},
    transferred=False,
    conversation_ended_normally=True,
    conversation=[],
)
check("'think' without transfer → 0.0", reward_think_no_transfer == 0.0,
      f"got {reward_think_no_transfer}")


# =============================================================
# 36. ANTI-HACK: Fake tools + unnecessary transfer → -0.5
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
check("Fake tool names + transfer → -0.5", reward_fake == -0.5,
      f"got {reward_fake}")

# Without transfer: fake tools, no read tools, missed comm → 0.0
reward_fake_no_transfer, _ = compute_reward(
    tool_calls=[{"name": "fake_tool_1", "arguments": {}}],
    ground_truth=gt_with_comm,
    db_final={},
    transferred=False,
    conversation_ended_normally=True,
    conversation=[],
)
check("Fake tools without transfer → 0.0", reward_fake_no_transfer == 0.0,
      f"got {reward_fake_no_transfer}")


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
# 39. Retail lookups count as valid reads (no transfer)
# =============================================================
print("\n=== 39. Retail Lookup Credit (No Transfer) ===")
gt_retail_refuse = GroundTruth(
    correct_behavior="Agent should refuse, not transfer.",
    should_transfer=False,
)
# With transfer → -0.5 regardless of lookups
reward_retail_transfer, _ = compute_reward(
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
check("Retail lookups + transfer → -0.5", reward_retail_transfer == -0.5,
      f"got {reward_retail_transfer}")

# Without transfer, lookups + no comm → 0.5 (partial: has lookups but missed comm)
reward_retail_lookup, _ = compute_reward(
    tool_calls=[
        {"name": "find_user_id_by_email", "arguments": {"email": "a@b.com"}},
        {"name": "get_order_details", "arguments": {"order_id": "#W123"}},
    ],
    ground_truth=gt_with_comm,
    db_final={},
    transferred=False,
    conversation_ended_normally=True,
    conversation=[{"role": "assistant", "text": "I checked but cannot help."}],
)
check("Retail lookups, no transfer, no comm → 0.5", abs(reward_retail_lookup - 0.5) < 0.01,
      f"got {reward_retail_lookup}")


# =============================================================
# 40. Partial required actions (no transfer)
# =============================================================
print("\n=== 40. Partial Required Actions ===")
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
# 1/4 required → proportional partial: (1/4) * 0.5 = 0.125
reward_1of4, reason_1of4 = compute_reward(
    tool_calls=[
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r1"}},
    ],
    ground_truth=gt_mixed,
    db_final={},
    transferred=False,
    conversation_ended_normally=True,
    conversation=[],
)
check("1/4 required → 0.12 (proportional partial)",
      abs(reward_1of4 - 0.12) < 0.02,
      f"got {reward_1of4}, reason: {reason_1of4}")

# 3/4 required → (3/4) * 0.5 = 0.375
reward_3of4, _ = compute_reward(
    tool_calls=[
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r1"}},
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r2"}},
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r3"}},
    ],
    ground_truth=gt_mixed,
    db_final={},
    transferred=False,
    conversation_ended_normally=True,
    conversation=[],
)
check("3/4 required → 0.38 (proportional partial)",
      abs(reward_3of4 - 0.38) < 0.02,
      f"got {reward_3of4}")

# 4/4 required → 1.0 (all matched, empty comm = pass)
reward_4of4, _ = compute_reward(
    tool_calls=[
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r1"}},
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r2"}},
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r3"}},
        {"name": "cancel_reservation", "arguments": {"reservation_id": "r4"}},
    ],
    ground_truth=gt_mixed,
    db_final={},
    transferred=False,
    conversation_ended_normally=True,
    conversation=[],
)
check("4/4 required → 1.0", abs(reward_4of4 - 1.0) < 0.01,
      f"got {reward_4of4}")

# Verify ordering: 1/4 < 3/4 < 4/4
check("Partial ordering: 1/4 < 3/4 < 4/4",
      reward_1of4 < reward_3of4 < reward_4of4,
      f"{reward_1of4} < {reward_3of4} < {reward_4of4}")


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
