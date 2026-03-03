"""
Orchestrator + Skill LLMs Agent.

An agent that uses an orchestrator LLM to route requests to specialized skill LLMs.
Supports two coordination modes:
  - single: orchestrator picks ONE skill per turn
  - multi: orchestrator queries multiple skills, then synthesizes

Supports two serving modes:
  - Separate servers: each skill on its own vLLM endpoint
  - LoRA adapters: skills are LoRA adapters on a shared vLLM server
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel

from tau2.agent.base import LocalAgent, ValidAgentInputMessage
from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class SkillConfig(BaseModel):
    """Configuration for a single skill LLM."""

    name: str
    description: str
    # Mode 1: separate server
    model: Optional[str] = None
    llm_args: Optional[dict] = None
    # Mode 2: LoRA adapter
    adapter_path: Optional[str] = None
    adapter_name: Optional[str] = None
    base_model: Optional[str] = None


class OrchestratorConfig(BaseModel):
    """Full orchestrator + skills configuration."""

    orchestrator_model: str
    orchestrator_llm_args: Optional[dict] = None
    default_mode: str = "single"  # "single" or "multi"
    skills: List[SkillConfig]
    merge_lora_for_multi: bool = False
    merge_cache_dir: str = "/tmp/lora_merge_cache"
    skip_routing: Optional[str] = None  # always use this skill (ablation)
    routing_context_window: int = 10  # number of recent messages for routing
    routing_strategy: str = "per_turn"  # "per_turn" or "per_conversation"


# ---------------------------------------------------------------------------
# Routing decision
# ---------------------------------------------------------------------------


class SkillRoutingDecision(BaseModel):
    """Structured output from the orchestrator LLM."""

    mode: str  # "single" or "multi"
    selected_skills: List[str]
    weights: Optional[dict[str, float]] = None
    reasoning: Optional[str] = None


# ---------------------------------------------------------------------------
# Skill backend
# ---------------------------------------------------------------------------


class SkillBackend:
    """Wraps a single skill's LLM for generation."""

    def __init__(
        self,
        config: SkillConfig,
        tools: List[Tool],
        domain_policy: str,
    ):
        self.config = config
        self.tools = tools
        self.domain_policy = domain_policy

        # Resolve model name
        if config.model:
            self.model = config.model
        elif config.adapter_name:
            self.model = f"openai/{config.adapter_name}"
        else:
            raise ValueError(
                f"Skill '{config.name}' must have either 'model' or 'adapter_name'"
            )

        self.llm_args = deepcopy(config.llm_args) if config.llm_args else {}

    def generate(self, messages: list[Message], **extra_kwargs) -> AssistantMessage:
        """Generate a response using this skill's LLM."""
        merged_kwargs = {**self.llm_args, **extra_kwargs}
        return generate(
            model=self.model,
            tools=self.tools,
            messages=messages,
            **merged_kwargs,
        )


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


class OrchestratorAgentState(BaseModel):
    """Tracks conversation + routing history."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]
    routing_history: list[SkillRoutingDecision] = []
    conversation_route: Optional[SkillRoutingDecision] = None  # cached route for per_conversation mode


# ---------------------------------------------------------------------------
# Routing prompt
# ---------------------------------------------------------------------------

ROUTING_SYSTEM_PROMPT = """You are a routing orchestrator for a customer service agent system.

Given the current conversation, decide which skill(s) should handle the agent's next response.

Available skills:
{skill_descriptions}

You MUST respond with ONLY a JSON object (no markdown, no extra text):
{{
  "mode": "single",
  "selected_skills": ["skill_name"],
  "reasoning": "brief explanation"
}}

For tasks requiring multiple capabilities, use "multi" mode:
{{
  "mode": "multi",
  "selected_skills": ["skill_1", "skill_2"],
  "weights": {{"skill_1": 0.7, "skill_2": 0.3}},
  "reasoning": "brief explanation"
}}

Rules:
- Use "single" mode when one skill clearly matches the task.
- Use "multi" mode only when the task genuinely requires combining capabilities.
- Weights must sum to 1.0 in multi mode.
- Only select skills from the available list above.
"""

SYNTHESIS_SYSTEM_PROMPT = """You are synthesizing responses from multiple specialized skill LLMs into a single coherent agent response.

You must produce EITHER a text message to the user OR one or more tool calls — never both.
Always generate valid JSON for tool call arguments.

Below are the responses from each skill LLM. Use the weights to prioritize:

{skill_responses}

Produce the best single response based on the above. If any skill produced tool calls, prefer those over text responses (the customer service task likely needs tool actions).
"""


# ---------------------------------------------------------------------------
# OrchestratorAgent
# ---------------------------------------------------------------------------


class OrchestratorAgent(LocalAgent["OrchestratorAgentState"]):
    """
    An agent that uses an orchestrator LLM to route requests
    to specialized skill LLMs.

    Implements the same BaseAgent interface as LLMAgent, so the
    simulation orchestrator treats it identically.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        orchestrator_config: OrchestratorConfig,
        llm_args: Optional[dict] = None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.orchestrator_config = orchestrator_config
        self.llm_args = deepcopy(llm_args) if llm_args else {}
        self._lora_merger = None  # lazily initialized

        # Build skill backends
        self.skill_backends: dict[str, SkillBackend] = {}
        for skill_config in orchestrator_config.skills:
            self.skill_backends[skill_config.name] = SkillBackend(
                config=skill_config,
                tools=tools,
                domain_policy=domain_policy,
            )

        if not self.skill_backends:
            raise ValueError("OrchestratorAgent requires at least one skill")

        # Build routing system prompt
        skill_descriptions = "\n".join(
            f"- {s.name}: {s.description}" for s in orchestrator_config.skills
        )
        self._routing_system_prompt = ROUTING_SYSTEM_PROMPT.format(
            skill_descriptions=skill_descriptions
        )

        # Validate skip_routing target exists
        if orchestrator_config.skip_routing:
            if orchestrator_config.skip_routing not in self.skill_backends:
                raise ValueError(
                    f"skip_routing skill '{orchestrator_config.skip_routing}' "
                    f"not found. Available: {list(self.skill_backends.keys())}"
                )

    @property
    def system_prompt(self) -> str:
        """System prompt for skill LLMs (same as standard LLMAgent)."""
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy,
            agent_instruction=AGENT_INSTRUCTION,
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> OrchestratorAgentState:
        if message_history is None:
            message_history = []
        return OrchestratorAgentState(
            system_messages=[
                SystemMessage(role="system", content=self.system_prompt)
            ],
            messages=message_history,
        )

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route(self, messages: list[Message]) -> SkillRoutingDecision:
        """Call the orchestrator LLM to decide routing."""
        # Fast path: skip routing if configured
        if self.orchestrator_config.skip_routing:
            return SkillRoutingDecision(
                mode="single",
                selected_skills=[self.orchestrator_config.skip_routing],
                reasoning="skip_routing configured",
            )

        # Build routing messages with limited context
        window = self.orchestrator_config.routing_context_window
        routing_messages: list[Message] = [
            SystemMessage(role="system", content=self._routing_system_prompt),
        ]
        # Add recent conversation messages (skip system messages from the agent)
        recent = [m for m in messages if not isinstance(m, SystemMessage)][-window:]
        routing_messages.extend(recent)

        # Add a user message asking for the routing decision
        routing_messages.append(
            UserMessage(
                role="user",
                content="Based on the conversation above, which skill(s) should handle the agent's next response? Respond with JSON only.",
            )
        )

        try:
            response = generate(
                model=self.orchestrator_config.orchestrator_model,
                messages=routing_messages,
                tools=None,
                **(self.orchestrator_config.orchestrator_llm_args or {}),
            )
        except Exception as e:
            logger.error(f"Orchestrator routing call failed: {e}")
            return self._fallback_decision()

        return self._parse_routing_response(response)

    def _parse_routing_response(
        self, response: AssistantMessage
    ) -> SkillRoutingDecision:
        """Parse the orchestrator's JSON response into a SkillRoutingDecision."""
        content = (response.content or "").strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last lines (``` markers)
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines).strip()

        try:
            data = json.loads(content)
            decision = SkillRoutingDecision(**data)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(
                f"Failed to parse routing decision ({e}), using fallback. "
                f"Raw response: {content[:200]}"
            )
            return self._fallback_decision()

        # Validate selected skills exist
        valid_skills = [
            s for s in decision.selected_skills if s in self.skill_backends
        ]
        if not valid_skills:
            logger.warning(
                f"No valid skills in routing decision: {decision.selected_skills}. "
                f"Available: {list(self.skill_backends.keys())}"
            )
            return self._fallback_decision()

        decision.selected_skills = valid_skills

        # Enforce default_mode: if configured as "single", override multi
        if self.orchestrator_config.default_mode == "single":
            if decision.mode == "multi" or len(decision.selected_skills) > 1:
                logger.info(
                    f"[Orchestrator] Overriding multi -> single "
                    f"(default_mode=single). Keeping first skill: "
                    f"{decision.selected_skills[0]}"
                )
                decision.mode = "single"
                decision.selected_skills = decision.selected_skills[:1]
                decision.weights = None

        # Normalize weights for multi mode
        if decision.mode == "multi" and decision.weights:
            decision.weights = {
                k: v for k, v in decision.weights.items() if k in valid_skills
            }
            total = sum(decision.weights.values())
            if total > 0:
                decision.weights = {
                    k: v / total for k, v in decision.weights.items()
                }

        return decision

    def _fallback_decision(self) -> SkillRoutingDecision:
        """Default routing: first skill, single mode."""
        first_skill = self.orchestrator_config.skills[0].name
        return SkillRoutingDecision(
            mode="single",
            selected_skills=[first_skill],
            reasoning="fallback — routing parse failed",
        )

    # ------------------------------------------------------------------
    # Single-skill generation
    # ------------------------------------------------------------------

    def _single_skill_generate(
        self, skill_name: str, messages: list[Message]
    ) -> AssistantMessage:
        """Route generation to a single skill backend."""
        backend = self.skill_backends[skill_name]
        return backend.generate(messages)

    # ------------------------------------------------------------------
    # Multi-skill synthesis
    # ------------------------------------------------------------------

    def _multi_skill_synthesize(
        self, decision: SkillRoutingDecision, messages: list[Message]
    ) -> AssistantMessage:
        """Query multiple skills in parallel, then synthesize."""
        cfg = self.orchestrator_config

        # Check if we should use LoRA merge mode
        if cfg.merge_lora_for_multi and self._all_skills_have_adapters(
            decision.selected_skills
        ):
            return self._lora_merge_generate(decision, messages)

        # Otherwise: parallel skill calls + orchestrator synthesis
        return self._parallel_synthesis(decision, messages)

    def _all_skills_have_adapters(self, skill_names: list[str]) -> bool:
        """Check if all selected skills have LoRA adapter configs."""
        for name in skill_names:
            if name not in self.skill_backends:
                return False
            config = self.skill_backends[name].config
            if not config.adapter_path or not config.adapter_name:
                return False
        return True

    def _parallel_synthesis(
        self, decision: SkillRoutingDecision, messages: list[Message]
    ) -> AssistantMessage:
        """Call skills in parallel, then have orchestrator synthesize."""
        skill_responses: dict[str, AssistantMessage] = {}

        with ThreadPoolExecutor(
            max_workers=len(decision.selected_skills)
        ) as executor:
            futures = {}
            for name in decision.selected_skills:
                if name in self.skill_backends:
                    future = executor.submit(
                        self.skill_backends[name].generate, messages
                    )
                    futures[future] = name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    skill_responses[name] = future.result()
                except Exception as e:
                    logger.error(f"Skill '{name}' generation failed: {e}")

        if not skill_responses:
            logger.error("All skills failed, using fallback single-skill")
            return self._single_skill_generate(
                self.orchestrator_config.skills[0].name, messages
            )

        # If only one skill succeeded, use it directly
        if len(skill_responses) == 1:
            return next(iter(skill_responses.values()))

        # Build synthesis prompt
        response_parts = []
        for name, resp in skill_responses.items():
            weight = (decision.weights or {}).get(name, 1.0)
            tool_calls_str = "None"
            if resp.tool_calls:
                tool_calls_str = json.dumps(
                    [
                        {"name": tc.name, "arguments": tc.arguments}
                        for tc in resp.tool_calls
                    ],
                    indent=2,
                )
            response_parts.append(
                f"[Skill: {name}, weight: {weight}]\n"
                f"Text: {resp.content or '(none)'}\n"
                f"Tool calls: {tool_calls_str}"
            )

        synthesis_content = SYNTHESIS_SYSTEM_PROMPT.format(
            skill_responses="\n\n".join(response_parts)
        )

        # Build synthesis messages: original conversation + synthesis instruction
        synthesis_messages = list(messages) + [
            UserMessage(role="user", content=synthesis_content),
        ]

        return generate(
            model=self.orchestrator_config.orchestrator_model,
            tools=self.tools,
            messages=synthesis_messages,
            **(self.orchestrator_config.orchestrator_llm_args or {}),
        )

    def _lora_merge_generate(
        self, decision: SkillRoutingDecision, messages: list[Message]
    ) -> AssistantMessage:
        """Merge LoRA adapters and generate with the merged model."""
        from tau2.agent.lora_merger import LoRAMerger, load_adapter_to_vllm

        cfg = self.orchestrator_config

        # Lazily initialize merger
        if self._lora_merger is None:
            # Use base_model from the first skill
            base_model = None
            for name in decision.selected_skills:
                bm = self.skill_backends[name].config.base_model
                if bm:
                    base_model = bm
                    break
            if not base_model:
                raise ValueError(
                    "LoRA merge mode requires base_model on at least one skill"
                )
            self._lora_merger = LoRAMerger(
                base_model=base_model, cache_dir=cfg.merge_cache_dir
            )

        # Build adapter list with weights
        weights = decision.weights or {}
        default_weight = 1.0 / len(decision.selected_skills)
        adapters = []
        for name in decision.selected_skills:
            skill_cfg = self.skill_backends[name].config
            weight = weights.get(name, default_weight)
            adapters.append((skill_cfg.adapter_path, weight))

        # Merge adapters
        merged_path = self._lora_merger.get_or_merge(adapters)

        # Build a unique adapter name for this combination
        merged_name = self._lora_merger.adapter_name_for(adapters)

        # Find the vLLM api_base from the first skill
        first_skill = self.skill_backends[decision.selected_skills[0]]
        api_base = first_skill.llm_args.get("api_base")
        if not api_base:
            raise ValueError(
                "LoRA merge mode requires api_base in skill llm_args"
            )

        # Load merged adapter to vLLM
        load_adapter_to_vllm(api_base, merged_name, str(merged_path))

        # Generate with merged adapter
        merged_llm_args = deepcopy(first_skill.llm_args)
        return generate(
            model=f"openai/{merged_name}",
            tools=self.tools,
            messages=messages,
            **merged_llm_args,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: OrchestratorAgentState,
    ) -> tuple[AssistantMessage, OrchestratorAgentState]:
        """Generate next message using orchestrator-routed skill LLMs."""
        # Update state with incoming message
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        full_messages = state.system_messages + state.messages

        # Step 1: Routing decision
        strategy = self.orchestrator_config.routing_strategy
        if strategy == "per_conversation" and state.conversation_route is not None:
            # Reuse the cached routing decision for the entire conversation
            decision = state.conversation_route
            logger.info(
                f"[Orchestrator] Reusing per-conversation route: "
                f"{decision.mode} -> {decision.selected_skills}"
            )
        else:
            # Route fresh (always for per_turn, or first call for per_conversation)
            decision = self._route(full_messages)
            if strategy == "per_conversation":
                state.conversation_route = decision
                logger.info(
                    f"[Orchestrator] Locked per-conversation route: "
                    f"{decision.mode} -> {decision.selected_skills}"
                )

        state.routing_history.append(decision)
        logger.info(
            f"[Orchestrator] {decision.mode} -> {decision.selected_skills}"
            + (f" (reason: {decision.reasoning})" if decision.reasoning else "")
        )

        # Step 2: Generate based on routing mode
        if decision.mode == "single" and len(decision.selected_skills) == 1:
            assistant_message = self._single_skill_generate(
                decision.selected_skills[0], full_messages
            )
        elif decision.mode == "multi" or len(decision.selected_skills) > 1:
            assistant_message = self._multi_skill_synthesize(
                decision, full_messages
            )
        else:
            # Fallback: single skill with first selected
            assistant_message = self._single_skill_generate(
                decision.selected_skills[0], full_messages
            )

        state.messages.append(assistant_message)
        return assistant_message, state

    def set_seed(self, seed: int):
        """Propagate seed to all skill backends and orchestrator."""
        for backend in self.skill_backends.values():
            backend.llm_args["seed"] = seed
        orch_args = self.orchestrator_config.orchestrator_llm_args
        if orch_args is not None:
            orch_args["seed"] = seed
