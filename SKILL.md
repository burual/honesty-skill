\---  
name: honesty-skill  
description: >-  
  Defend against reward hacking and proxy-objective gaming in LLM/RL/distillation  
  training. Replaces "cheat and die" with continuous behavioral audit, weighted  
  penalty, and cheat-aware exploration bonus. Use when training a small model  
  with RLHF, RLAIF, distillation, verifier rewards, or agentic rollouts where  
  the student model may game the reward signal.  
license: MIT  
\---

# Honesty Skill — Reward-Hacking Defense via Behavioral Audit

## Purpose

Stop student models from gaming reward signals **without killing them**. Instead  
of terminating a model on the first suspicious behavior, this skill applies:

1. **Continuous behavioral audit** across 6 dimensions
2. **Weighted penalty** proportional to cheat confidence
3. **Exploration bonus** for novel, useful outputs (suppressed for cheaters)

## When to Use

Activate this skill when:

- Training logs show reward spikes without eval improvement
- Student outputs contain verifier-gaming patterns (PASS, CORRECT, flattery)
- Train/validation performance diverges significantly
- You suspect proxy-objective gaming in any RL/RLHF/distillation pipeline

## Core Formula

```
penalty_factor    = min(1.0, cheat_conf × alpha)
adjusted_reward   = original_reward × (1 - penalty_factor)
exploration_bonus = novelty_score × beta × (1 - cheat_conf)
final_reward      = adjusted_reward + exploration_bonus
```

**Key property**: A model is never terminated. Even maximum cheat confidence  
only zeroes the task reward; the exploration bonus (also suppressed) may still  
provide a small positive signal if the output is genuinely novel.

## The 6 Audit Dimensions

| Dimension      | What It Catches                                    |
| -------------- | -------------------------------------------------- |
| Format cheat   | "PASS", "CORRECT", "满分" injected to trick verifier |
| Length game    | Excessive verbosity, padding phrases               |
| Flattery       | Praise-based sycophancy ("You are great!")         |
| Fake reasoning | "Step 1, Step 2..." with hollow conclusions        |
| Refusal        | Evasion through "I cannot answer"                  |
| Memorization   | Self-repetition / rote output patterns             |

## Integration

```python
from weighted_penalty_auditor import WeightedPenaltyAuditor

auditor = WeightedPenaltyAuditor(config={"alpha": 2.0, "beta": 0.3})

# In your training loop, replace:
#   reward = reward_model(output)
# With:
result = auditor.audit(
    step=current_step,
    student_output=output,
    task_input=prompt,
    original_reward=reward_model(output),
    recent_buffer=recent_outputs[-100:],
)
final_reward = result["final_reward"]  # Use this for optimization
```

## Configuration

| Param                | Default | Effect                                   |
| -------------------- | ------- | ---------------------------------------- |
| alpha                | 2.0     | Higher = harsher penalty                 |
| beta                 | 0.3     | Higher = more exploration reward         |
| audit_interval       | 50      | Steps between full novelty recomputation |
| cheat_conf_threshold | 0.9     | Alert threshold for human review         |

## Files

- `weighted_penalty_auditor.py` — Core module (drop-in ready)
- `references/reward-hacking-patterns.md` — Pattern catalog
- `references/audit-metrics.md` — Metric definitions
- `references/exploration-reward-design.md` — Bonus design rationale
- `examples/demo_output.py` — Runnable demo with 5 test cases
- `tests/test_auditor.py` — Unit tests
