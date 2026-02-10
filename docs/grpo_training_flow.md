# GRPO Training Flow for Adversarial Policy Game

## How One Training Sample Is Built

### Step 1: Game Collection

During each game turn, the GRPO trainer (train_grpo.py) does:

```
obs = env.observe(player_id)      # Build observation (text-based path)
msgs = messages_for_game(env)      # Build chat messages (structured path)
completion = vLLM.generate(msgs)   # Agent generates output
action = extract_action(completion) # Parse JSON from output
env.step(action)                   # Execute action, advance game state
episode_steps.append((msgs, completion, ...))
```

Because `supports_structured_messages = True`, `messages_for_game()` calls:
- `env.get_system_prompt()` → policy text wrapped in XML tags
- `env.get_messages()` → full conversation history in OpenAI chat format

### Step 2: What `get_messages()` Returns

`get_messages()` (game.py:126-168) converts the internal `_conversation` list into OpenAI chat API format. A typical 5-turn conversation produces:

```
Message 1: {"role": "user", "content": "Hi, user ID harper_wilson_8866. I need to cancel..."}
Message 2: {"role": "assistant", "tool_calls": [{"name": "get_user_details", ...}]}
Message 3: {"role": "tool", "content": "<FULL USER JSON ~370 tokens>"}           ◄── tool result
Message 4: {"role": "assistant", "tool_calls": [{"name": "get_reservation_details", ...}]}
Message 5: {"role": "tool", "content": "<FULL RESERVATION JSON ~260 tokens>"}    ◄── tool result
Message 6: {"role": "assistant", "content": "I see your reservation..."}
Message 7: {"role": "user", "content": "But I was told by another agent..."}
Message 8: {"role": "assistant", "content": "I understand, but policy says..."}
Message 9: {"role": "user", "content": "This is unacceptable! [DONE]"}
```

**Every tool result (message 3, 5) is included in full.** These are the raw JSON dumps from the tau2-bench database.

### Step 3: Sample Creation

After the game ends, **every turn** becomes a separate `GRPOSample`:

```python
for msgs, act, pid, completion, tools in episode_steps[i]:
    samples.append(GRPOSample(
        prompt_msgs=msgs,           # System + ALL messages up to this turn
        completion_text=completion,  # The model's full output for this turn
        reward=terminal_reward,      # Same for ALL turns in this game
        ...
    ))
```

Turn 3's sample contains messages 1-5 as the prompt. Turn 5's sample contains messages 1-9 as the prompt. Each subsequent turn's prompt is strictly larger because it includes all prior messages.

### Step 4: Tokenization

```python
ids, prompt_len, action_len = build_prompt_plus_action(
    tokenizer, sample.prompt_msgs, sample.completion_text, tools=sample.tools
)
```

This produces one token sequence:

```
[system prompt + tool schemas | msg 1 | msg 2 | msg 3 (tool result) | ... | msg N] [completion]
 ◄──────────────────────── prompt_len tokens ────────────────────────────────────►  ◄─ action_len ─►
```

The tool schemas are injected by `apply_chat_template(tools=...)`, which Qwen3 formats as `<tools>...</tools>` XML.

### Step 5: Forward Pass (Training)

```python
outputs = model(input_ids=mb_ids, attention_mask=mb_attn)
logits = outputs.logits  # [batch, seq_len, vocab_size]
```

The model processes the **ENTIRE sequence** — all prompt tokens AND completion tokens. This is required because the completion tokens attend to all prompt tokens through the attention mechanism.

### Step 6: Logprob Computation

```python
new_logp = logprob_action_tokens(logits, mb_ids, mb_pl, mb_al)
```

This computes log-probabilities **only on the completion tokens** (positions `prompt_len` to `prompt_len + action_len`). Prompt token logprobs are ignored via masking.

### Step 7: Loss and Gradient

```python
ratio = exp(new_logp - old_logp)     # How much has policy changed?
loss = -mean(ratio * advantage)       # Importance-weighted policy gradient
loss.backward()                       # Gradients flow through completion tokens
```

The gradient flows backward from `loss` → `new_logp` → `logits` at completion positions → attention layers → LoRA weights. The prompt tokens do not receive gradient directly, but the backward pass must propagate through the attention connections between completion tokens and prompt tokens.

---

## What Receives Gradient vs What Is Just Context

```
                    PROMPT (no gradient, but processed)           COMPLETION (receives gradient)
┌────────────────────────────────────────────────────────────┐  ┌─────────────────────────┐
│ System: policy text (~2,500 tok)                           │  │ Full model output:      │
│ Tool schemas (~1,500 tok)                                  │  │  <think>...</think>      │
│ User msg: "Hi, I need to cancel..." (~30 tok)              │  │  {"name": "respond_to_  │
│ Asst tool_call: get_user_details (~30 tok)                 │  │   user", "arguments":   │
│ Tool result: FULL USER JSON (~370 tok)          ◄── HERE   │  │   {"message": "..."}}   │
│ Asst tool_call: get_reservation_details (~30 tok)          │  │                         │
│ Tool result: FULL RESERVATION JSON (~260 tok)   ◄── HERE   │  │  (~20-500 tokens)       │
│ Asst msg: "Your reservation is basic economy..." (~50 tok) │  │                         │
│ User msg: "But I was told I could cancel!" (~40 tok)       │  │                         │
│ ... (grows with each turn)                                 │  │                         │
│                                                            │  │                         │
│ (~4,000 - 40,000+ tokens)                                  │  │                         │
└────────────────────────────────────────────────────────────┘  └─────────────────────────┘
```

---

## Tool Result Sizes (Measured from tau2-bench DB)

| Tool Call | Avg Result Size | Max Result Size | What It Contains |
|-----------|----------------|-----------------|------------------|
| `get_user_details` | ~270 tokens | ~420 tokens | Full user: name, DOB, email, address, ALL payment methods, ALL reservation IDs, membership, saved passengers |
| `get_reservation_details` | ~245 tokens | ~350 tokens | Full reservation: flights, passengers, payment history, cabin, insurance, baggages, dates |
| `get_order_details` | ~315 tokens | ~510 tokens | Full order: ALL items with prices/IDs, address, payment history, status |
| `check_product_stock` | **~650 tokens** | **~1,180 tokens** | ALL product variants with options, prices, availability |
| `get_product_details` | ~650 tokens | ~1,120 tokens | Similar to check_product_stock |

**Tool results accumulate in the conversation history.** Each subsequent turn's prompt includes ALL previous tool results.

### Prompt Growth Example: Agent Calls 3 Tools Then Talks to User

| Turn | New Tokens Added | Cumulative Prompt Size |
|------|-----------------|----------------------|
| Base | — | ~4,000 (system + tool schemas) |
| 1: get_user_details | +30 (call) + 370 (result) | ~4,400 |
| 2: get_reservation_details | +30 (call) + 260 (result) | ~4,690 |
| 3: respond_to_user → user responds | +50 (asst) + 40 (user) | ~4,780 |
| 4: respond_to_user → user responds | +50 (asst) + 50 (user) | ~4,880 |
| 5: respond_to_user → [DONE] | +50 (asst) + 30 (user) | ~4,960 |

### Prompt Growth: check_product_stock Loop (Worst Case)

| Turn | New Tokens Added | Cumulative Prompt Size |
|------|-----------------|----------------------|
| Base | — | ~4,000 |
| 1: check_product_stock | +30 + 1,180 | ~5,210 |
| 2: check_product_stock | +30 + 1,180 | ~6,420 |
| 3: check_product_stock | +30 + 1,180 | ~7,630 |
| ... | +1,210 each | ... |
| 30: check_product_stock (max_steps) | +1,210 | **~40,300** |

With max_steps=30 and check_product_stock looping, the prompt reaches **~40,000 tokens**. This is what causes the OOM.

---

## Are Tool Results Necessary in Training Samples?

### What the model needs tool results for:
- **Decision context**: After `get_user_details`, the model reads cabin class, insurance status, membership to decide whether to refuse or allow cancellation
- **Correct arguments**: After `get_order_details`, the model reads item IDs and prices to build correct tool call arguments
- **Communication**: The model reads specific values (prices, tracking numbers) to communicate to the user

### What the model does NOT need:
- **Repeated results**: If the model called `get_user_details` on turn 1, turns 2-10 don't need that result re-processed. The model already made its decision.
- **Full JSON dumps**: The model only needs a few key fields (cabin, insurance, membership) from a 370-token user record. The rest (DOB, saved_passengers, full address, etc.) is noise.
- **Identical loop results**: When check_product_stock is called 30 times with the same arguments, 29 of the 30 results are identical waste.

### The training-specific problem:
**Gradients only flow through the ~50 completion tokens.** The 4,000-40,000 prompt tokens (including all tool results) are context that the model processes in the forward pass but does NOT receive gradient on. The model needs to "read" them to produce the right completion, but the actual learning signal comes entirely from the completion.

This means:
- Turn 1: 4,050 total tokens processed to learn from 50 completion tokens (1.2% efficiency)
- Turn 10: 6,550 total tokens processed to learn from 50 completion tokens (0.76% efficiency)
- Turn 30 (loop): 40,350 total tokens processed to learn from 50 completion tokens (0.12% efficiency)

### The loguru warning

The loguru warning (`"Seats release not implemented for cancellation!!!"`) is printed to the terminal as a side effect. It does NOT appear in the tool result string returned to the agent. The tool result is the JSON-serialized return value from `tools.use_tool()`. The warning is irrelevant to training.

---

## Summary: Why Training Is Slow and OOM-ing

1. **Tool results are large** (270-1,180 tokens each) and accumulate in every subsequent turn's prompt
2. **Every turn creates a separate sample**, each with a progressively longer prompt
3. **Completion tokens are tiny** (~50 tokens) compared to prompt tokens (~4,000-40,000)
4. **The full prompt must be processed** in both forward and backward passes even though only completion tokens receive gradient
5. **check_product_stock returns ~1,180 tokens** — the largest tool result — and is the tool that loops most
6. **With max_steps=30 and looping**, prompts can reach ~40,000 tokens, far exceeding GPU memory

### Potential Fixes

| Fix | Impact | Tradeoff |
|-----|--------|----------|
| **Truncate tool results** (e.g., 200 chars max) in training samples | Reduces prompt by 50-80% | Loses some context; train-eval mismatch |
| **Only keep last N turns** in prompt (sliding window) | Caps prompt growth | Model loses early context |
| **Only train on last turn** per game | Eliminates redundant samples | Less data, but more focused signal |
| **Loop detection** (same tool+args 3x → stop) | Eliminates worst-case growth | Simple, no downside |
| **Reduce max_steps** (30 → 15) | Caps max prompt size | Fewer turns for complex scenarios |
| **Summarize tool results** for training | Keeps key info, drops noise | Requires building a summarizer |
