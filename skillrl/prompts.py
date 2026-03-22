"""Prompt templates for SkillRL skill generation, evolution, and SFT trace creation."""


SKILL_SECTION_TEMPLATE = """\
<skills>
The following skills are distilled from prior experience. Apply relevant skills when handling the user's request.

## General Skills
{general_skills}

## Task-Specific Skills
{category_skills}
</skills>"""


SKILL_ENTRY_TEMPLATE = """\
[{skill_id}] {title}
  Principle: {principle}
  When to apply: {when_to_apply}"""


SKILL_EVOLUTION_PROMPT = """\
You are analyzing failed customer service interactions to generate new skills for an AI agent.

## Current Domain: {domain}
## Current SkillBank
{current_skills}

## Failed Trajectories
The following interactions failed (the agent did not achieve the task goal):

{failed_trajectories}

## Task
Analyze the failures above. For each failure, identify what the agent did wrong and what skill or principle would have prevented the mistake.

Then generate 1-{max_new} NEW skills that are NOT already covered by the existing SkillBank. Each skill should be:
- Concise (1-2 sentences for the principle)
- Actionable (tells the agent exactly what to check or do)
- General enough to apply across similar tasks

Output as JSON array:
```json
[
  {{
    "skill_id": "{next_id_prefix}N",
    "title": "3-5 word title",
    "principle": "1-2 sentence actionable principle",
    "when_to_apply": "Brief description of when this skill is relevant",
    "category": "general OR specific category name"
  }}
]
```

Only output the JSON array, no other text."""


SFT_TRACE_SYSTEM_PROMPT = """\
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.

<policy>
{policy}
</policy>

<skills>
The following skills are distilled from prior experience. Apply relevant skills when handling the user's request.

## General Skills
{general_skills}

## Task-Specific Skills
{category_skills}
</skills>"""
