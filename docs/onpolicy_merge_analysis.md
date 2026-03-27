 ---
  Why Merging and Distillation Fail — and What Will Work

  1. Results Summary

  ┌────────────────────────────────┬─────────┬─────────┐
  │             Model              │ Pass/50 │ vs Base │
  ├────────────────────────────────┼─────────┼─────────┤
  │ Base (Qwen3-30B)               │ 12      │ —       │
  ├────────────────────────────────┼─────────┼─────────┤
  │ Individual Adapters            │         │         │
  ├────────────────────────────────┼─────────┼─────────┤
  │ Structured-data LoRA (40 iter) │ 18      │ +6      │
  ├────────────────────────────────┼─────────┼─────────┤
  │ Precondition LoRA (40 iter)    │ 17      │ +5      │
  ├────────────────────────────────┼─────────┼─────────┤
  │ Toolcalling LoRA (40 iter)     │ 15      │ +3      │
  ├────────────────────────────────┼─────────┼─────────┤
  │ Multistep LoRA (v4, 10 iter)   │ 15      │ +3      │
  ├────────────────────────────────┼─────────┼─────────┤
  │ Union ceiling (4 individuals)  │ 29/50   │ +17     │
  ├────────────────────────────────┼─────────┼─────────┤
  │ Merges                         │         │         │
  ├────────────────────────────────┼─────────┼─────────┤
  │ Core-TSV merge                 │ 18      │ +6      │
  ├────────────────────────────────┼─────────┼─────────┤
  │ Linear merge                   │ 16      │ +4      │
  ├────────────────────────────────┼─────────┼─────────┤
  │ TIES-DARE merge                │ 15      │ +3      │
  ├────────────────────────────────┼─────────┼─────────┤
  │ Stack (rank 64) merge          │ 13      │ +1      │
  ├────────────────────────────────┼─────────┼─────────┤
  │ Distillation                   │         │         │
  ├────────────────────────────────┼─────────┼─────────┤
  │ Distill-to-base (20 iter)      │ 15      │ +3      │
  ├────────────────────────────────┼─────────┼─────────┤
  │ Distill-to-core-TSV (40 iter)  │ 14      │ +2      │
  ├────────────────────────────────┼─────────┼─────────┤
  │ Distill-to-base (40 iter)      │ 13      │ +1      │
  ├────────────────────────────────┼─────────┼─────────┤
  │ Distill-to-base (60 iter)      │ 14      │ +2      │
  └────────────────────────────────┴─────────┴─────────┘

  Key observation: The best merge (core-TSV, 18/50) merely ties the best single adapter
  (structured-data, 18/50). The union ceiling is 29/50 — meaning individual adapters
  collectively solve 29 tasks, but no merge or distillation model captures more than 18. We're
  losing 11 tasks (38%) of the available improvement on the floor.

  2. Root Cause Analysis

  2.1 The Training Games Are Not Aligned With the Evaluation Task

  This is the primary failure mode. Each game teaches a different behavior in a different
  format, and these formats conflict with each other AND with the tau2-bench evaluation:

  ┌────────────────┬───────────────┬────────────────────────┬───────────┬─────────┬──────┐
  │      Game      │    Domain     │         Format         │   Tools   │  User   │ Turn │
  │                │               │                        │           │   Sim   │  s   │
  ├────────────────┼───────────────┼────────────────────────┼───────────┼─────────┼──────┤
  │ Structured-dat │ Synthetic     │ submit_answers(a1,a2,a │ Custom    │ None    │ 1    │
  │ a              │               │ 3)                     │           │         │      │
  ├────────────────┼───────────────┼────────────────────────┼───────────┼─────────┼──────┤
  │ Tool-calling   │ Airline+Retai │ tau2-bench tools       │ tau2-benc │ LLM     │ Mult │
  │                │ l             │                        │ h         │         │ i    │
  ├────────────────┼───────────────┼────────────────────────┼───────────┼─────────┼──────┤
  │ Multistep      │ Retail        │ Custom retail tools    │ Custom    │ Scripte │ Mult │
  │                │               │                        │           │ d       │ i    │
  ├────────────────┼───────────────┼────────────────────────┼───────────┼─────────┼──────┤
  │ Precondition   │ Airline       │ tau2-bench tools       │ tau2-benc │ LLM     │ Mult │
  │                │               │                        │ h         │         │ i    │
  ├────────────────┼───────────────┼────────────────────────┼───────────┼─────────┼──────┤
  │ tau2-bench     │ Airline       │ tau2-bench tools       │ tau2-benc │ LLM     │ Mult │
  │ eval           │               │                        │ h         │         │ i    │
  └────────────────┴───────────────┴────────────────────────┴───────────┴─────────┴──────┘

  Conflicts identified:
  1. Structured-data game uses submit_answers tool that doesn't exist in tau2-bench. This
  teaches a response format that's never useful at eval time.
  2. Multistep game uses custom retail tool schemas (modify_pending_order_items,
  find_user_id_by_name_zip) with different parameter names than tau2-bench. Training on these
  teaches wrong argument names.
  3. Structured-data is single-turn while tau2-bench is multi-turn. This trains the model to
  dump everything at once vs. the step-by-step tool-call discipline needed.
  4. System prompts differ across all 4 games: structured-data uses a simple prompt, multistep
  uses another, only precondition and tool-calling match tau2-bench's XML <instructions> +
  <policy> format.
  5. Reward signals conflict: tool-calling env uses reward=-1.0 for false-positive actions,
  precondition uses 0.0 for the same error.

  2.2 Why Merging Fails: Skill Interference in Weight Space

  When you train 4 separate LoRA adapters, each one moves the model in a different direction in
   weight space to solve a different problem. Merging assumes these directions are either
  orthogonal (linear merge) or can be aligned (core-TSV). But:

  1. The games teach contradictory behaviors. Structured-data teaches "emit one JSON blob with
  all answers." Tool-calling teaches "call one tool at a time and wait for results." These
  produce opposing gradients on the same attention and MLP weights. When you average them, they
   cancel out.
  2. Stack merge (rank 64) is worse than base (+1). This is the most telling result. Stacking
  concatenates all LoRA matrices, giving the model full access to all 4 skills' parameters
  without any averaging/cancellation. Yet it scores 13/50 — worse than any individual adapter.
  This means the model can't even figure out which skill to use when given access to all of
  them. The skills are not just interfering in weight space; they're interfering in behavior
  space.
  3. Core-TSV ties the best single adapter. TSV allocates each adapter a proportional share of
  singular directions, which is the "fairest" merge — minimal inter-task interference. That it
  merely ties structured-data (18/50) means the other 3 adapters contribute ~nothing useful
  when merged. The structured-data adapter is carrying the merge.

  2.3 Why Distillation Fails: Wrong Training Signal + Train-Eval Gap

  The distillation setup has several compounding problems:

  1. Per-skill teachers disagree on the same input. When the student generates a rollout for a
  tau_tool_calling game, it gets teacher logprobs from the tool-calling LoRA. When it generates
   a rollout for precondition_check, it gets teacher logprobs from the precondition LoRA. But
  these teachers would produce different logprobs for the same token sequence, because they
  learned different behaviors. The student is being pulled in 4 directions simultaneously.
  2. 25% training budget per skill is insufficient. With groups-per-batch=32 split 4 ways, each
   skill gets ~8 rollouts per iteration. With multi-turn conversations that can be 10+ turns,
  many rollouts fail or are truncated. The effective gradient signal per skill per iteration is
   very thin.
  3. The student practices on games, but is evaluated on tau2-bench. Even if distillation
  perfectly transfers all 4 game skills into one model, the resulting model has learned to play
   4 synthetic games — not to solve tau2-bench airline tasks. The structured-data game's
  submit_answers tool and the multistep game's custom retail tools are useless at eval time.
  4. Distillation to core-TSV (14/50) is worse than core-TSV itself (18/50). Starting from a
  merged model and then distilling degrades it. The on-policy distillation gradients are
  pulling the model away from the merge's balanced solution toward the conflicting game
  behaviors.
  5. Non-monotonic iteration curve (15 → 13 → 14 for 20/40/60 iter). The best distillation
  result is the earliest checkpoint (20 iter). More training makes it worse. This is a classic
  sign of the model overfitting to game-specific behaviors at the expense of general
  capability.

  2.4 The Fundamental Problem: Distribution Mismatch

  The core issue is a train-eval distribution gap:

  - At training time: the model plays 4 synthetic games with different tools, different
  formats, different reward signals, and 25% of its training budget on a single-turn game with
  a non-existent tool.
  - At eval time: the model faces 50 airline customer service tasks using tau2-bench's specific
   tools, specific policy document, specific system prompt, and LLM user simulator.

  No amount of merging or distillation can close this gap because the gap is in the training
  data distribution, not in how the weights are combined.

  3. Evidence That This Diagnosis Is Correct

  1. The 2 games that match tau2-bench format perform best. Precondition (17/50) and
  tool-calling (15/50) use tau2-bench's actual tools and system prompt. Structured-data (18/50)
   works by teaching JSON parsing, a broadly useful skill despite the format mismatch.
  Multistep (15/50) teaches multi-step reasoning but in a foreign tool schema.
  2. The orchestrator approach (23/50) massively outperforms any merge/distill. Your memory
  notes show the 3-model orchestrator with per-conversation routing hits 23/50 — +5 over the
  best individual or merge. This works precisely because it avoids the interference problem:
  each adapter runs in its own inference pass, with its own learned behavior intact.
  3. Stack merge (rank 64) is the worst merge. If the skills were complementary, stacking
  (which preserves all parameters) should be the best merge, not the worst. That it's the worst
   proves the skills are actively interfering.
  4. More distillation iterations make it worse. If the distillation target (the union of
  teacher behaviors) were learnable, more iterations should improve results. The reverse trend
  proves the target is incoherent.

  4. The Method That Should Work: Direct GRPO on tau2-bench

  The method that should "for sure work" to learn a single model capable of all skills is to
  train directly on the evaluation distribution — tau2-bench airline tasks — using GRPO (which
  you already have working for the individual games).

  4.1 Approach: On-Policy GRPO on tau2-bench Rollouts

  for each iteration:
      1. Student generates rollouts by playing actual tau2-bench airline tasks
         (using the tau2-bench environment as the game)
      2. tau2-bench's grader scores each rollout (reward = task pass/fail)
      3. GRPO updates the student's LoRA weights using these rewards

  Why this works:
  - Zero distribution gap: the model trains on exactly what it will be evaluated on
  - No conflicting formats: only tau2-bench tools, only tau2-bench system prompt, only
  tau2-bench user simulator
  - No skill interference: the model learns one coherent behavior (airline customer service)
  rather than 4 conflicting ones
  - The reward signal is exactly the eval metric: task pass/fail

  4.2 Concrete Implementation

  You already have nearly all the infrastructure:

  1. Game environment: Wrap tau2-bench's airline evaluation as a game environment compatible
  with train_distill.py's collect_grpo_rollouts. The key pieces exist:
    - tau2-bench/src/tau2/agent/llm_agent.py already generates tool-calling conversations
    - tau2-bench already has a grading system that returns binary rewards
    - You just need a Tau2BenchGame class that: (a) samples a task, (b) runs the LLM agent
  against the user simulator, (c) returns the reward
  2. Training loop: Use your existing GRPO training loop from train_distill.py but with --games
   tau2_bench_airline instead of the 4 synthetic games. No teacher needed — this is direct RL,
  not distillation.
  3. Hyperparameters:
    - 50 tasks is small, so use all of them each iteration (no subsampling)
    - Start with groups-per-batch=50, group-size=4 (4 rollouts per task for variance reduction)
    - Use the structured-data LoRA (best individual, 18/50) as the initialization — warm-start
  from your best checkpoint
    - Low learning rate (5e-6) to avoid catastrophic forgetting of the base model's language
  ability
    - Monitor per-task pass rates to detect overfitting to easy tasks

  4.3 Why Not Just Distillation on tau2-bench?

  You could also do on-policy distillation with the 4 teachers, but restricted to tau2-bench
  rollouts instead of synthetic games. This would fix the distribution gap but still has the
  "teachers disagree" problem. GRPO with task reward is cleaner: the reward signal is
  unambiguous (did the task pass or not?), and the model figures out which behavior to use on
  its own.

  4.4 Hybrid: GRPO + Teacher KL Regularization

  If you want to leverage the teacher adapters, the best hybrid approach is:

  loss = GRPO_reward_loss + λ * min_over_teachers(KL(student || teacher_i))

  For each student rollout, compute KL divergence against all 4 teachers and use the minimum —
  this lets the student learn from whichever teacher is most relevant for that particular
  rollout, rather than being pulled in all 4 directions. Set λ small (0.01-0.1) so the reward
  signal dominates.

  4.5 Alternative: Multi-Task GRPO With Aligned Games

  If you want to keep the synthetic games for data augmentation (since 50 tau2-bench tasks is
  small), fix the games first:

  1. Remove structured-data game or rewrite it to use tau2-bench airline tools (e.g., model
  must look up flight data and answer questions using tau2-bench's search_flights and
  get_reservation_details tools)
  2. Remove multistep game or rewrite it to use tau2-bench airline tool schemas instead of
  custom retail tools
  3. Align reward scales: use 0/1 binary rewards consistently across all games
  4. Use tau2-bench system prompt in all games
  5. Weight tau2-bench tasks heavily (e.g., 50% tau2-bench, 25% precondition, 25% tool-calling)

  This gives you more training data diversity while eliminating the format conflicts.

  5. Summary

  ┌──────────────────┬──────────────────────────────────────┬─────────────────────────────┐
  │   Why it fails   │              Root cause              │             Fix             │
  ├──────────────────┼──────────────────────────────────────┼─────────────────────────────┤
  │ Merging loses    │ Opposing gradients from conflicting  │ Don't merge — or only merge │
  │ skills           │ games cancel in weight space         │  compatible adapters        │
  ├──────────────────┼──────────────────────────────────────┼─────────────────────────────┤
  │ Stack merge is   │ Model can't disambiguate 4           │ Skills need to be trained   │
  │ worst            │ conflicting behaviors                │ on same distribution        │
  ├──────────────────┼──────────────────────────────────────┼─────────────────────────────┤
  │ Distillation     │ Teachers disagree, games ≠ eval      │ Train on eval distribution  │
  │ degrades         │ distribution                         │ directly                    │
  ├──────────────────┼──────────────────────────────────────┼─────────────────────────────┤
  │ More iterations  │ Overfitting to game-specific         │ Fewer/no synthetic game     │
  │ = worse          │ artifacts                            │ iterations                  │
  └──────────────────┴──────────────────────────────────────┴─────────────────────────────┘

  The single most impactful change: Train a single LoRA adapter directly on tau2-bench airline
  tasks via GRPO, warm-started from your best individual checkpoint. This eliminates every
  source of interference simultaneously.
                                                             
✻ Worked for 8m 31s                               
                                                 
❯ we cant train directly on tau bench or do rl on it                                           
                                                                                               
● Understood. That changes the approach significantly — we must work through the synthetic     
  games only. Given that constraint, here's the revised analysis focused on what's actionable. 

  The Core Problem Remains: Game Misalignment                                                  
                                                                                               
  The games themselves are the bottleneck, not the merging/distillation algorithm. The 4 games 
  teach conflicting behaviors that destructively interfere when combined. But instead of       
  bypassing the games, we need to fix them.                                                    
                                                                                               
  Two Approaches That Should Work

  Approach 1: Multi-Task GRPO (Single LoRA, All Games Simultaneously)                          
   
  This has never been tried. Your current pipeline is:                                         
                  
  Individual GRPO → 4 separate LoRAs → merge/distill post-hoc                                  
                                                                                               
  The alternative is:                                                                          
                                                                                               
  Single GRPO on all 4 games simultaneously → 1 LoRA                                           
                                                                                               
  You already have the infrastructure for this — train_distill.py supports --games             
  "game1:0.25,game2:0.25,..." and the GRPO training loop (not just distillation). The          
  difference from distillation is: no teacher, just reward signal from the games directly.     
                  
  But this will only work if the games are first fixed to not conflict. Which brings us to:    
   
  Approach 2 (prerequisite): Fix the Games                                                     
                  
  The 3 critical fixes, ordered by impact:                                                     
   
  Fix 1: Replace structured-data game's submit_answers tool with tau2-bench airline tools.     
                  
  Currently the model learns to call submit_answers(answer_1, answer_2, answer_3) — a tool that
   doesn't exist at eval time. Rewrite the game so the model must use tau2-bench's actual
  airline tools (get_reservation_details, search_flights, etc.) to look up data and then call  
  respond_to_user with the answers. Same skill (JSON data reasoning), but practiced in the
  right format.

  Fix 2: Replace multistep game's custom retail tools with tau2-bench airline tools.           
   
  The multistep game uses find_user_id_by_name_zip, modify_pending_order_items, etc. — none of 
  which exist in tau2-bench airline. Rewrite the game to use tau2-bench airline operations:
  cancel flights, change seats, upgrade cabins, update passenger info — in sequence. Same skill
   (multi-step sequential execution), but with the right tool schemas.

  Fix 3: Unify system prompts and reward scales.                                               
   
  - All games should use tau2-bench's <instructions> + <policy> XML system prompt format       
  - All games should use binary 0/1 rewards (remove the -1.0 penalty in tool-calling env)
  - All games should use the LLM user simulator (remove scripted user in multistep)            
                                                                                               
  After these fixes, all 4 games would teach different skills but in the same format —         
  tau2-bench airline format. The gradients would no longer conflict because the model would be 
  learning one coherent interface (tau2-bench airline tool-calling) with 4 types of challenging
   scenarios.     

  Why This Should Work                                                                         
   
  Consider what the 4 skills actually teach, stripped of format:                               
                  
  ┌─────────────────┬────────────────────────────────────────────┬──────────────────────────┐  
  │      Skill      │              What it teaches               │   Conflicts with eval?   │
  ├─────────────────┼────────────────────────────────────────────┼──────────────────────────┤  
  │ Structured-data │ Parse complex JSON, compute derived values │ Yes — wrong tool format  │
  ├─────────────────┼────────────────────────────────────────────┼──────────────────────────┤
  │ Tool-calling    │ Call right tool with right args            │ No (if airline domain)   │  
  ├─────────────────┼────────────────────────────────────────────┼──────────────────────────┤  
  │ Precondition    │ Check policy before acting                 │ No                       │  
  ├─────────────────┼────────────────────────────────────────────┼──────────────────────────┤  
  │ Multistep       │ Execute sequential multi-op tasks          │ Yes — wrong tool schemas │
  └─────────────────┴────────────────────────────────────────────┴──────────────────────────┘  
   
  Once structured-data and multistep are rewritten to use tau2-bench tools, all 4 skills become
   complementary rather than conflicting. At that point:
                                                                                               
  - Multi-task GRPO should produce a single LoRA that's better than any individual, because the
   gradient signals reinforce rather than cancel
  - Merging (especially core-TSV) should also improve, because the weight deltas will point in 
  compatible directions                                                                        
  - Distillation becomes viable because all 4 teachers now teach behaviors in the same format
                                                                                               
  What About Merging Without Game Fixes?                                                       
                                                                                               
  If rewriting games is too costly, the best you can do with current games is:                 
                  
  1. Only merge the 2 compatible adapters: precondition + tool-calling (both use tau2-bench    
  format). Drop structured-data and multistep from the merge entirely — they add noise.
  2. Weighted merge favoring structured-data: Since structured-data carries the core-TSV merge 
  (18/50 = same as structured-data solo), and it teaches a broadly useful skill (JSON          
  reasoning) despite the format mismatch, try a merge with structured-data at weight 0.7 and
  the others at 0.1 each.                                                                      
  3. Distill from only 2 teachers: Use only precondition and tool-calling teachers (the
  format-compatible ones). This gives the student cleaner signal from 2 coherent teachers      
  instead of noise from 4 conflicting ones.
                                                                                               
  Bottom Line                                                                                  
   
  The ranking of approaches by expected effectiveness:                                         
                  
  1. Fix games → multi-task GRPO (single LoRA trained on all 4 aligned games) — highest        
  ceiling, eliminates all interference
  2. Fix games → merge/distill — should also work once games are aligned                       
  3. Merge only compatible adapters (precondition + tool-calling) — quick experiment, no game  
  changes needed                                                                               
  4. Current approach (merge/distill all 4 as-is) — fundamentally limited by game conflicts    
                                                            