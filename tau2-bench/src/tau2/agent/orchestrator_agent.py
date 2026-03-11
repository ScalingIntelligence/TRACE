"""
Orchestrator + Skill LLMs Agent.

An agent that uses an orchestrator LLM to route requests to specialized skill LLMs.
The orchestrator picks ONE skill per turn to handle the response.

Supports two serving modes:
  - Separate servers: each skill on its own vLLM endpoint
  - LoRA adapters: skills are LoRA adapters on a shared vLLM server
"""

import json
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
    skills: List[SkillConfig]
    skip_routing: Optional[str] = None  # always use this skill (ablation)
    routing_context_window: Optional[int] = None  # None = use entire conversation
    routing_strategy: str = "per_turn"  # "per_turn" or "per_conversation"
    routing_mode: str = "llm"  # "llm" (JSON output), "classifier" (1 forward pass with structured_outputs), or "embedding" (sentence similarity)
    embedding_model: str = "Qwen/Qwen3-Embedding-8B"  # model name for sentence_transformers when routing_mode == "embedding"


# ---------------------------------------------------------------------------
# Routing decision
# ---------------------------------------------------------------------------


class SkillRoutingDecision(BaseModel):
    """Structured output from the orchestrator LLM."""

    selected_skill: str
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

Given the current conversation, decide which skill should handle the agent's next response.

Available skills:
{skill_descriptions}

You MUST respond with ONLY a JSON object (no markdown, no extra text):
{{
  "selected_skill": "skill_name",
  "reasoning": "brief explanation"
}}

Rules:
- Select the single best skill for the current task.
- Only select a skill from the available list above.
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

        # Build label-to-skill mapping for classifier mode
        if orchestrator_config.routing_mode == "classifier":
            self._label_to_skill: dict[str, str] = {}
            self._skill_to_label: dict[str, str] = {}
            label_lines = []
            for i, skill_config in enumerate(orchestrator_config.skills):
                label = chr(ord("A") + i)
                self._label_to_skill[label] = skill_config.name
                self._skill_to_label[skill_config.name] = label
                label_lines.append(
                    f"{label}: {skill_config.name} - {skill_config.description}"
                )
            self._classifier_system_prompt = (
                "You are a routing classifier for a customer service agent system.\n"
                "Given the current conversation, select the skill that should handle "
                "the agent's next response.\n\n"
                + "\n".join(label_lines)
                + "\n\nOnly output the label (e.g. A, B, C). Do not output anything else."
            )
            self._classifier_labels = list(self._label_to_skill.keys())

        # Build embedding model and pre-encode skill descriptions for embedding mode
        if orchestrator_config.routing_mode == "embedding":
            from sentence_transformers import SentenceTransformer

            logger.info(
                f"[Embedding Router] Loading model: {orchestrator_config.embedding_model}"
            )
            self._embedding_model = SentenceTransformer(
                orchestrator_config.embedding_model,
                model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
                tokenizer_kwargs={"padding_side": "left"},
            )
            # Map each skill name to its description for encoding
            self._embedding_skill_names: list[str] = []
            skill_docs: list[str] = []
            for skill_config in orchestrator_config.skills:
                self._embedding_skill_names.append(skill_config.name)
                skill_docs.append(
                    f"{skill_config.name}: {skill_config.description}"
                )
            # Pre-encode skill descriptions as documents (no query prompt)
            self._embedding_skill_vectors = self._embedding_model.encode(skill_docs)
            logger.info(
                f"[Embedding Router] Pre-encoded {len(skill_docs)} skill descriptions"
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
                selected_skill=self.orchestrator_config.skip_routing,
                reasoning="skip_routing configured",
            )

        if self.orchestrator_config.routing_mode == "classifier":
            return self._route_classifier(messages)

        if self.orchestrator_config.routing_mode == "embedding":
            return self._route_embedding(messages)

        return self._route_llm(messages)

    def _route_llm(self, messages: list[Message]) -> SkillRoutingDecision:
        """Route using the LLM JSON output format (original behavior)."""
        # Build routing messages with limited context
        window = self.orchestrator_config.routing_context_window
        routing_messages: list[Message] = [
            SystemMessage(role="system", content=self._routing_system_prompt),
        ]
        # Add conversation messages (skip system messages from the agent)
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]
        recent = non_system[-window:] if window is not None else non_system
        routing_messages.extend(recent)

        # Add a user message asking for the routing decision
        routing_messages.append(
            UserMessage(
                role="user",
                content="Based on the conversation above, which skill should handle the agent's next response? Respond with JSON only.",
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

    def _route_classifier(self, messages: list[Message]) -> SkillRoutingDecision:
        """Route using single-forward-pass classifier with vLLM structured_outputs.

        Uses vLLM's structured_outputs choice API to constrain the output to
        one of the label IDs (A, B, C, ...) in a single forward pass.
        """
        from openai import OpenAI

        orch_args = self.orchestrator_config.orchestrator_llm_args or {}

        # Extract connection params from orchestrator_llm_args
        api_base = orch_args.get("api_base", "http://localhost:8080/v1")
        api_key = orch_args.get("api_key", "EMPTY")

        # Strip litellm "openai/" prefix for direct OpenAI client
        model_name = self.orchestrator_config.orchestrator_model
        if model_name.startswith("openai/"):
            model_name = model_name[len("openai/"):]

        # Build classifier messages (simple dicts for OpenAI client)
        window = self.orchestrator_config.routing_context_window
        classifier_messages: list[dict] = [
            {"role": "system", "content": self._classifier_system_prompt},
        ]
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]
        recent = non_system[-window:] if window is not None else non_system

        for m in recent:
            content = getattr(m, "content", None) or ""
            if not content:
                continue
            # Map tool messages to user role for simple context
            role = "assistant" if m.role == "assistant" else "user"
            classifier_messages.append({"role": role, "content": content})

        classifier_messages.append(
            {"role": "user", "content": "Which skill should handle the next response?"}
        )

        try:
            client = OpenAI(base_url=api_base, api_key=api_key)
            completion = client.chat.completions.create(
                model=model_name,
                messages=classifier_messages,
                extra_body={
                    "structured_outputs": {
                        "choice": self._classifier_labels,
                    }
                },
                temperature=orch_args.get("temperature", 0.0),
                max_tokens=1,
                seed=orch_args.get("seed"),
            )
            chosen_label = completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Classifier routing call failed: {e}")
            return self._fallback_decision()

        if chosen_label in self._label_to_skill:
            skill_name = self._label_to_skill[chosen_label]
            logger.info(
                f"[Classifier] label={chosen_label} -> skill={skill_name}"
            )
            return SkillRoutingDecision(
                selected_skill=skill_name,
                reasoning=f"classifier label {chosen_label}",
            )
        else:
            logger.warning(
                f"Classifier returned unexpected label: '{chosen_label}'. "
                f"Expected one of {self._classifier_labels}"
            )
            return self._fallback_decision()

    def _route_embedding(self, messages: list[Message]) -> SkillRoutingDecision:
        """Route by encoding the conversation and finding the most similar skill description.

        Uses SentenceTransformer cosine similarity between a query built from
        recent conversation messages and the pre-encoded skill descriptions.
        No API call is needed — inference runs locally on the loaded model.
        """
        # Build query from recent conversation messages
        window = self.orchestrator_config.routing_context_window
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]
        recent = non_system[-window:] if window is not None else non_system

        # Concatenate recent message contents into a single query string
        query_parts: list[str] = []
        for m in recent:
            content = getattr(m, "content", None) or ""
            if content:
                query_parts.append(content)

        if not query_parts:
            logger.warning("[Embedding Router] No message content for routing query")
            return self._fallback_decision()

        query_text = "\n".join(query_parts)

        try:
            # Encode the query using the "query" prompt (as recommended by Qwen3-Embedding)
            query_embedding = self._embedding_model.encode(
                [query_text], prompt_name="query"
            )
            # Compute cosine similarity against pre-encoded skill descriptions
            similarity = self._embedding_model.similarity(
                query_embedding, self._embedding_skill_vectors
            )
            # similarity shape: (1, num_skills) — get the best match
            scores = similarity[0]  # first (only) query row
            best_idx = int(scores.argmax())
            best_score = float(scores[best_idx])
            skill_name = self._embedding_skill_names[best_idx]

            logger.info(
                f"[Embedding Router] Best match: {skill_name} "
                f"(score={best_score:.4f}, scores={[f'{self._embedding_skill_names[i]}:{float(scores[i]):.4f}' for i in range(len(self._embedding_skill_names))]})"
            )
            return SkillRoutingDecision(
                selected_skill=skill_name,
                reasoning=f"embedding similarity {best_score:.4f}",
            )
        except Exception as e:
            logger.error(f"Embedding routing failed: {e}")
            return self._fallback_decision()

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

        # Validate selected skill exists
        if decision.selected_skill not in self.skill_backends:
            logger.warning(
                f"Invalid skill in routing decision: {decision.selected_skill}. "
                f"Available: {list(self.skill_backends.keys())}"
            )
            return self._fallback_decision()

        return decision

    def _fallback_decision(self) -> SkillRoutingDecision:
        """Default routing: first skill."""
        first_skill = self.orchestrator_config.skills[0].name
        return SkillRoutingDecision(
            selected_skill=first_skill,
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
                f"{decision.selected_skill}"
            )
        else:
            # Route fresh (always for per_turn, or first call for per_conversation)
            decision = self._route(full_messages)
            if strategy == "per_conversation":
                state.conversation_route = decision
                logger.info(
                    f"[Orchestrator] Locked per-conversation route: "
                    f"{decision.selected_skill}"
                )

        state.routing_history.append(decision)
        logger.info(
            f"[Orchestrator] -> {decision.selected_skill}"
            + (f" (reason: {decision.reasoning})" if decision.reasoning else "")
        )
        print(f"[ROUTING] Task routing -> {decision.selected_skill} (reason: {decision.reasoning})")

        # Step 2: Generate using the selected skill
        assistant_message = self._single_skill_generate(
            decision.selected_skill, full_messages
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
