"""Adversarial Policy Adherence Game.

Trains the skill of policy adherence under adversarial user pressure.
Targets Skill 1 from tau2-bench failure analysis (21 tasks, 20.6% of failures).

12 scenario templates covering: ineligible cancellation, basic economy
modification, destination changes, bag removal, false policy claims,
wrong payment methods, individual item cancellation, system vs user claims,
emotional manipulation, unmet preconditions, valid action refusal prevention,
and multi-reservation selective actions.
"""

from .game import AdversarialPolicyGame, extract_action, SYSTEM_PROMPT
from .llm_user import UserLLMClient

__all__ = ["AdversarialPolicyGame", "extract_action", "SYSTEM_PROMPT", "UserLLMClient"]
