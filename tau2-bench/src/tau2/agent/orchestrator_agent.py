"""
Orchestrator + Skill LLMs Agent.

An agent that uses an orchestrator LLM to route requests to specialized skill LLMs.
The orchestrator picks ONE skill per turn to handle the response.

Supports two serving modes:
  - Separate servers: each skill on its own vLLM endpoint
  - LoRA adapters: skills are LoRA adapters on a shared vLLM server

Supports three routing modes:
  - llm: JSON output routing (original)
  - classifier: 1 forward pass with structured_outputs + logprobs -> weighted LoRA
  - embedding: cosine similarity between conversation and skill descriptions (SentenceTransformer)
"""

import json
import math
from copy import deepcopy
from typing import Dict, List, Optional

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
# Embedding model singleton — loaded once, reused across agent instances
# ---------------------------------------------------------------------------

_embedding_model_cache: dict[tuple[str, str], object] = {}


def _get_embedding_model(model_name: str, device: str):
    """Return a cached SentenceTransformer, loading it only on first call."""
    key = (model_name, device)
    if key not in _embedding_model_cache:
        from sentence_transformers import SentenceTransformer

        logger.info(f"[Embedding] Loading {model_name} on {device}")
        _embedding_model_cache[key] = SentenceTransformer(
            model_name,
            device=device,
            tokenizer_kwargs={"padding_side": "left"},
        )
    else:
        logger.info(f"[Embedding] Reusing cached {model_name} on {device}")
    return _embedding_model_cache[key]


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
    routing_mode: str = "llm"  # "llm", "llm_multi", "classifier", or "embedding"
    routing_topk: Optional[int] = None  # limit routing to top-k adapters; applies to classifier and embedding modes (None=all)
    routing_topp: Optional[float] = None  # top-p nucleus filtering: keep fewest adapters whose cumulative weight >= topp; applies to classifier and embedding
    classifier_temperature: float = 1.0  # temperature scaling applied to logprobs before softmax (>1 = softer, <1 = sharper)
    min_blend_confidence: Optional[float] = None  # only blend when primary's raw (pre-temp) probability < this; else hard-route
    is_weighted: bool = False  # True: use classifier logprob weights; False: weight=1.0 for all selected adapters
    # Embedding-based routing
    embedding_model: Optional[str] = None  # e.g. "Qwen/Qwen3-Embedding-8B"; required when routing_mode="embedding"
    embedding_gpu: Optional[str] = None  # device for the embedding model, e.g. "cuda:0" (default: "cpu")
    embedding_threshold: float = 0.5  # similarity threshold; skills above this are activated


# ---------------------------------------------------------------------------
# Routing decision
# ---------------------------------------------------------------------------


class SkillRoutingDecision(BaseModel):
    """Structured output from the orchestrator LLM."""

    selected_skill: str
    reasoning: Optional[str] = None
    skill_weights: Optional[Dict[str, float]] = None  # soft weights for weighted LoRA merge


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

ROUTING_SYSTEM_PROMPT = """You are a routing classifier for a customer service agent system.

Given the current conversation, decide which skill should handle the agent's next response. Emotional tone does not affect routing. The number of actions does not determine routing — focus on the nature of the request.

Available skills:
{skill_descriptions}

Output your answer as: SELECTED_SKILL: skill_name
"""

ROUTING_MULTI_SYSTEM_PROMPT = """You are a routing classifier for a customer service agent system.
Each skill is a complete, independently trained agent.

Available skills:
{skill_descriptions}

Route by what the customer needs done, not by emotional tone.

Output your answer as:
SELECTED_SKILL: skill_name
CONFIDENCE: high or low

Use high when one skill clearly matches. Use low when the request
touches two different skill areas and you are unsure which is best.
If low, add: SECONDARY_SKILL: skill_name

Examples of CONFIDENCE calibration:
- "Change the name on my reservation" → high (single clear action)
- "Cancel my flight and get a refund" → high (single clear action)
- "I want to cancel two bookings and upgrade a third to business" → low (execution + comparison)
- "Book the exact same flight I had last week" → low (execution + checking past data)
- "My flight was delayed, I need compensation" → low (data lookup + policy check)
- "I want to downgrade my cabin class due to money issues" → low (comparison + possible restriction)
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
        self._routing_multi_system_prompt = ROUTING_MULTI_SYSTEM_PROMPT.format(
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

        # Pre-load embedding model (singleton) and encode skill descriptions
        if orchestrator_config.routing_mode == "embedding":
            emb_model_name = orchestrator_config.embedding_model or "Qwen/Qwen3-Embedding-8B"
            emb_device = orchestrator_config.embedding_gpu or "cpu"

            self._embedding_model = _get_embedding_model(emb_model_name, emb_device)

            # Encode skill descriptions as retrieval documents (no query prompt)
            self._embedding_skill_names: list[str] = [
                s.name for s in orchestrator_config.skills
            ]
            skill_desc_texts = [s.description for s in orchestrator_config.skills]
            self._skill_description_embeddings = self._embedding_model.encode(
                skill_desc_texts
            )
            self._embedding_threshold = orchestrator_config.embedding_threshold
            logger.info(
                f"[Embedding] Encoded {len(skill_desc_texts)} skill descriptions"
            )

        # Validate skip_routing target(s) exist (supports comma-separated)
        if orchestrator_config.skip_routing:
            for sk in orchestrator_config.skip_routing.split(","):
                sk = sk.strip()
                if sk not in self.skill_backends:
                    raise ValueError(
                        f"skip_routing skill '{sk}' "
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
        # Clear weighted LoRA config at the start of each conversation
        self._clear_weighted_lora()
        return OrchestratorAgentState(
            system_messages=[
                SystemMessage(role="system", content=self.system_prompt)
            ],
            messages=message_history,
        )

    def _clear_weighted_lora(self):
        """Clear the weighted LoRA config file to disable weighted mode.

        Deletes the per-port config file(s) so that the engine's reader
        sees an OSError and falls back to single-adapter mode.
        """
        import os, tempfile
        from urllib.parse import urlparse

        # Collect unique ports from skill api_base URLs
        ports: set[str] = set()
        for skill_config in self.orchestrator_config.skills:
            api_base = (skill_config.llm_args or {}).get("api_base", "")
            if api_base:
                parsed = urlparse(api_base)
                if parsed.port:
                    ports.add(str(parsed.port))
        orchestrator_api_base = (
            self.orchestrator_config.orchestrator_llm_args or {}
        ).get("api_base", "")
        if orchestrator_api_base:
            parsed = urlparse(orchestrator_api_base)
            if parsed.port:
                ports.add(str(parsed.port))

        # Fall back to "default" if no ports found
        if not ports:
            ports = {"default"}

        for port_tag in ports:
            config_path = os.path.join(
                tempfile.gettempdir(),
                f"vllm_weighted_lora_config_{port_tag}.json",
            )
            try:
                os.remove(config_path)
            except OSError:
                pass

    def _hot_swap_adapter(self, skill_name: str):
        """Unload all LoRA adapters then reload only the selected one.

        This ensures the adapter occupies slot 0 on a clean slate,
        producing identical results to single-adapter serving.  Avoids
        non_blocking copy races and stale-slot padding that arise when
        max_loras < number of registered adapters.
        """
        import requests as http_requests

        selected_config = None
        for sc in self.orchestrator_config.skills:
            if sc.name == skill_name and sc.adapter_name:
                selected_config = sc
                break
        if selected_config is None:
            return  # base model skill, nothing to swap

        # Resolve the vLLM server URL
        api_base = (selected_config.llm_args or {}).get("api_base")
        if not api_base:
            api_base = (self.orchestrator_config.orchestrator_llm_args or {}).get(
                "api_base", "http://localhost:8080/v1"
            )
        server_url = api_base.rstrip("/").replace("/v1", "")

        # 1. Unload every adapter that is NOT the selected one
        try:
            resp = http_requests.get(f"{server_url}/v1/models", timeout=10)
            models = resp.json().get("data", [])
        except Exception as e:
            logger.warning(f"[HotSwap] Failed to list models: {e}")
            return

        for m in models:
            mid = m["id"]
            # Skip base model and the adapter we want
            if m.get("parent") is None:
                continue  # base model
            if mid == selected_config.adapter_name:
                continue  # keep this one
            try:
                http_requests.post(
                    f"{server_url}/v1/unload_lora_adapter",
                    json={"lora_name": mid},
                    timeout=10,
                )
                logger.info(f"[HotSwap] Unloaded adapter '{mid}'")
            except Exception as e:
                logger.warning(f"[HotSwap] Failed to unload '{mid}': {e}")

        # 2. Unload the selected adapter too, then reload it fresh into slot 0
        try:
            http_requests.post(
                f"{server_url}/v1/unload_lora_adapter",
                json={"lora_name": selected_config.adapter_name},
                timeout=10,
            )
        except Exception:
            pass  # may already be unloaded

        try:
            http_requests.post(
                f"{server_url}/v1/load_lora_adapter",
                json={
                    "lora_name": selected_config.adapter_name,
                    "lora_path": selected_config.adapter_path,
                },
                timeout=30,
            )
            logger.info(
                f"[HotSwap] Loaded adapter '{selected_config.adapter_name}' "
                f"as sole adapter (slot 0)"
            )
        except Exception as e:
            logger.error(f"[HotSwap] Failed to reload adapter: {e}")

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route(self, messages: list[Message]) -> SkillRoutingDecision:
        """Call the orchestrator LLM to decide routing."""
        # Fast path: skip routing if configured
        # Supports comma-separated skill names for multi-adapter blending
        if self.orchestrator_config.skip_routing:
            skills = [s.strip() for s in self.orchestrator_config.skip_routing.split(",")]
            weights = {s: 1.0 for s in skills if s in self.skill_backends}
            primary = skills[0] if skills[0] in self.skill_backends else list(self.skill_backends.keys())[0]
            return SkillRoutingDecision(
                selected_skill=primary,
                reasoning="skip_routing configured",
                skill_weights=weights,
            )

        if self.orchestrator_config.routing_mode == "classifier":
            return self._route_classifier(messages)

        if self.orchestrator_config.routing_mode == "llm_multi":
            return self._route_llm_multi(messages)

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
                content="Based on the conversation above, which skill should handle the agent's next response? Answer with SELECTED_SKILL: skill_name",
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
            decision = self._fallback_decision()
            decision.skill_weights = {decision.selected_skill: 1.0}
            return decision

        decision = self._parse_routing_response(response)
        # Use deterministic weighted LoRA path (weight=1.0 for hard selection)
        decision.skill_weights = {decision.selected_skill: 1.0}
        return decision

    def _route_llm_multi(self, messages: list[Message]) -> SkillRoutingDecision:
        """Route using LLM that predicts multiple skills with equal weights."""
        import re

        window = self.orchestrator_config.routing_context_window
        routing_messages: list[Message] = [
            SystemMessage(role="system", content=self._routing_multi_system_prompt),
        ]
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]
        recent = non_system[-window:] if window is not None else non_system
        routing_messages.extend(recent)

        routing_messages.append(
            UserMessage(
                role="user",
                content=(
                    "Based on the conversation above, which skill should handle this? "
                    "Answer with SELECTED_SKILL and CONFIDENCE."
                ),
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
            logger.error(f"LLM multi routing call failed: {e}")
            return self._fallback_decision()

        content = (response.content or "").strip()

        # Parse primary skill: SELECTED_SKILL(S): skill_name(, skill_2)
        match = re.search(
            r'SELECTED_SKILLS?:\s*(.+)', content, re.IGNORECASE
        )
        if not match:
            logger.warning(f"LLM multi routing parse failed: {content[:200]}")
            return self._fallback_decision()

        raw = match.group(1).strip()
        skill_names = [s.strip().rstrip(".") for s in raw.split(",")]
        valid = [s for s in skill_names if s in self.skill_backends]
        if not valid:
            logger.warning(f"No valid skills in: {skill_names}")
            return self._fallback_decision()

        primary = valid[0]
        skills = [primary]

        # Check for SECONDARY_SKILL (confidence-gated format)
        sec_match = re.search(
            r'SECONDARY_SKILL:\s*(\S+)', content, re.IGNORECASE
        )
        if sec_match:
            sec = sec_match.group(1).strip().rstrip(".")
            if sec in self.skill_backends and sec != primary:
                skills.append(sec)

        # Also include comma-separated skills from SELECTED_SKILLS line
        for s in valid[1:]:
            if s not in skills:
                skills.append(s)

        weights = {s: 1.0 for s in skills}
        reasoning = content[:match.start()].strip()
        short_reasoning = reasoning[-200:] if len(reasoning) > 200 else reasoning

        weight_str = ", ".join(f"{s}:1.0" for s in skills)
        print(
            f"[LLM_MULTI] {' + '.join(skills)} | {weight_str}",
            flush=True,
        )
        logger.info(f"[LLM_MULTI] skills={skills}")

        return SkillRoutingDecision(
            selected_skill=primary,
            reasoning=short_reasoning,
            skill_weights=weights,
        )

    def _route_classifier(self, messages: list[Message]) -> SkillRoutingDecision:
        """Route using classifier + logprobs -> softmax weights for LoRA merge.

        Uses vLLM's structured_outputs choice API to constrain the output to
        one of the label IDs (A, B, C, ...) in a single forward pass, and
        requests logprobs to get a soft probability distribution over all
        skills. These weights are used to create a weighted LoRA merge via
        /v1/create_weighted_lora.

        NOTE: temperature controls weight sharpness. At temp=0.0, weights will
        be extremely peaked (~hard routing). Use temp>0 (e.g., 0.5-1.0) for
        softer blending. The logprobs are computed on the pre-softmax logits
        AFTER the structured_outputs choice mask is applied.
        """
        from openai import OpenAI

        orch_args = self.orchestrator_config.orchestrator_llm_args or {}

        api_base = orch_args.get("api_base", "http://localhost:8080/v1")
        api_key = orch_args.get("api_key", "EMPTY")

        model_name = self.orchestrator_config.orchestrator_model
        if model_name.startswith("openai/"):
            model_name = model_name[len("openai/"):]

        # Build classifier messages
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
                logprobs=True,
                top_logprobs=len(self._classifier_labels) - 1,
                temperature=orch_args.get("temperature", 0.0),
                max_tokens=1,
                seed=orch_args.get("seed"),
            )

            chosen_label = completion.choices[0].message.content.strip()

            # Extract logprobs for all labels
            label_logprobs: Dict[str, float] = {}
            logprobs_data = completion.choices[0].logprobs
            if logprobs_data and logprobs_data.content:
                token_logprob = logprobs_data.content[0]
                # The chosen token
                label_logprobs[token_logprob.token.strip()] = token_logprob.logprob
                # Alternative tokens
                if token_logprob.top_logprobs:
                    for alt in token_logprob.top_logprobs:
                        label_logprobs[alt.token.strip()] = alt.logprob

            # Compute raw primary probability (before temperature scaling) for confidence gating
            if label_logprobs:
                raw_probs = {l: math.exp(lp) for l, lp in label_logprobs.items()}
                raw_total = sum(raw_probs.values())
                if raw_total > 0:
                    raw_probs = {l: p / raw_total for l, p in raw_probs.items()}
                self._last_primary_raw_prob = max(raw_probs.values()) if raw_probs else 1.0
            else:
                self._last_primary_raw_prob = 1.0

            # Convert to skill weights via exp + normalize
            # Apply classifier_temperature scaling: logprob / T before exp
            cls_temp = getattr(self.orchestrator_config, 'classifier_temperature', 1.0)
            skill_weights: Dict[str, float] = {}
            if label_logprobs:
                # exp(logprob / T) for each label that maps to a known skill
                exp_probs = {}
                for label, lp in label_logprobs.items():
                    if label in self._label_to_skill:
                        exp_probs[label] = math.exp(lp / cls_temp) if cls_temp > 0 else (1.0 if lp == max(label_logprobs.values()) else 0.0)

                # Normalize
                total = sum(exp_probs.values())
                if total > 0:
                    for label, prob in exp_probs.items():
                        skill_name = self._label_to_skill[label]
                        skill_weights[skill_name] = prob / total

            # Warn if we didn't get logprobs for all labels
            if len(label_logprobs) < len(self._classifier_labels):
                missing = set(self._classifier_labels) - set(label_logprobs.keys())
                logger.warning(
                    f"[Classifier] Missing labels in logprobs: {missing}. "
                    f"Got: {list(label_logprobs.keys())}"
                )

            # Log weights
            weight_str = ", ".join(
                f"{s}:{w:.4f}" for s, w in sorted(skill_weights.items())
            )
            logger.info(
                f"[Classifier] label={chosen_label} weights=[{weight_str}]"
            )
            print(
                f"[CLASSIFIER] label={chosen_label} | {weight_str}",
                flush=True,
            )

        except Exception as e:
            logger.error(f"Classifier routing call failed: {e}")
            return self._fallback_decision()

        if chosen_label in self._label_to_skill:
            skill_name = self._label_to_skill[chosen_label]
            return SkillRoutingDecision(
                selected_skill=skill_name,
                reasoning=f"classifier label {chosen_label}",
                skill_weights=skill_weights,
            )
        else:
            logger.warning(
                f"Classifier returned unexpected label: '{chosen_label}'. "
                f"Expected one of {self._classifier_labels}"
            )
            return self._fallback_decision()

    def _route_embedding(self, messages: list[Message]) -> SkillRoutingDecision:
        """Route using embedding cosine similarity between conversation and skill descriptions.

        Encodes the recent conversation as a query and computes cosine similarity
        against pre-encoded skill descriptions. Skills with similarity > threshold
        are activated with normalized similarity scores as weights. If none exceed
        the threshold, the highest-similarity skill is selected alone.
        """
        window = self.orchestrator_config.routing_context_window
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]
        recent = non_system[-window:] if window is not None else non_system

        # Build query text from conversation (user messages only, skip greetings)
        query_parts = []
        for m in recent:
            content = getattr(m, "content", None) or ""
            if content and m.role == "user":
                query_parts.append(content)
        query_text = "\n".join(query_parts) if query_parts else "general inquiry"

        try:
            # Encode query with custom instruction for skill routing
            query_embedding = self._embedding_model.encode(
                [query_text],
                prompt="Instruct: Given a customer service request, identify which skill category best handles it\nQuery: "
            )

            # Compute cosine similarity: shape (1, num_skills) -> (num_skills,)
            similarities = self._embedding_model.similarity(
                query_embedding, self._skill_description_embeddings
            )[0]
        except Exception as e:
            logger.error(f"Embedding routing failed: {e}")
            return self._fallback_decision()

        # Build (name, score) pairs sorted descending
        skill_scores = sorted(
            [
                (self._embedding_skill_names[i], float(similarities[i]))
                for i in range(len(self._embedding_skill_names))
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        # Filter by threshold
        threshold = self._embedding_threshold
        above_threshold = [
            (name, score) for name, score in skill_scores if score > threshold
        ]

        if not above_threshold:
            # None above threshold -> route to highest similarity only
            selected = [skill_scores[0]]
        else:
            selected = above_threshold

        # Build weights
        if len(selected) == 1:
            # Single skill: weight = 1.0 (same execution path as topk=1)
            skill_weights: Dict[str, float] = {selected[0][0]: 1.0}
        else:
            # Multiple skills: use similarity scores as weights, clamp >= 0
            raw = {name: max(score, 0.0) for name, score in selected}
            total = sum(raw.values())
            if total > 0:
                skill_weights = {name: w / total for name, w in raw.items()}
            else:
                skill_weights = {name: 1.0 / len(selected) for name, _ in selected}

        primary = selected[0][0]

        # Log
        weight_str = ", ".join(f"{s}:{w:.4f}" for s, w in skill_weights.items())
        score_str = ", ".join(f"{s}:{sc:.4f}" for s, sc in skill_scores)
        logger.info(
            f"[Embedding] scores=[{score_str}] threshold={threshold} "
            f"selected=[{weight_str}]"
        )
        print(
            f"[EMBEDDING] {primary} | {weight_str} | scores: {score_str}",
            flush=True,
        )

        return SkillRoutingDecision(
            selected_skill=primary,
            reasoning=f"embedding similarity (top={skill_scores[0][1]:.4f})",
            skill_weights=skill_weights,
        )

    def _set_weighted_lora(
        self, skill_weights: Dict[str, float]
    ) -> str:
        """Call vLLM /v1/create_weighted_lora to set weighted config.

        Only includes skills that have LoRA adapters (base model skills are
        the zero expert with Δ=0, excluded from the weighted set).

        Returns the model name to use for inference (first adapter with
        nonzero weight — triggers the LoRA code path in vLLM).
        """
        import requests as http_requests

        # Collect adapters with LoRA (exclude base model skills)
        lora_adapters = []
        first_adapter_model = None
        for skill_config in self.orchestrator_config.skills:
            if skill_config.adapter_name and skill_config.name in skill_weights:
                weight = skill_weights[skill_config.name]
                if weight > 0:
                    lora_adapters.append({
                        "name": skill_config.adapter_name,
                        "weight": weight,
                    })
                    if first_adapter_model is None:
                        first_adapter_model = f"openai/{skill_config.adapter_name}"

        # Confidence gating: if primary adapter's raw probability exceeds threshold, hard-route
        min_blend = getattr(self.orchestrator_config, 'min_blend_confidence', None)
        if min_blend and len(lora_adapters) > 1:
            sorted_adapters = sorted(lora_adapters, key=lambda a: a["weight"], reverse=True)
            # Compute raw (pre-temperature) probability of primary
            raw_total = sum(math.exp(lp) for lp in label_logprobs_raw.values()) if hasattr(self, '_last_label_logprobs_raw') else 0
            primary_raw = sorted_adapters[0]["weight"]  # Use the weight as proxy
            # Actually, we need the raw logprobs. Store them from _route_classifier.
            if hasattr(self, '_last_primary_raw_prob') and self._last_primary_raw_prob >= min_blend:
                lora_adapters = [sorted_adapters[0]]
                first_adapter_model = f"openai/{lora_adapters[0]['name']}"
                logger.info(
                    f"[ConfGate] Primary raw prob {self._last_primary_raw_prob:.4f} >= {min_blend}, hard-routing"
                )

        # Apply top-k filtering if configured
        routing_topk = getattr(self.orchestrator_config, 'routing_topk', None)
        if routing_topk and len(lora_adapters) > routing_topk:
            lora_adapters.sort(key=lambda a: a["weight"], reverse=True)
            lora_adapters = lora_adapters[:routing_topk]
            first_adapter_model = f"openai/{lora_adapters[0]['name']}"

        # Apply top-p (nucleus) filtering if configured (not used for embedding mode)
        routing_topp = getattr(self.orchestrator_config, 'routing_topp', None)
        if routing_topp and len(lora_adapters) > 1 and self.orchestrator_config.routing_mode != "embedding":
            lora_adapters.sort(key=lambda a: a["weight"], reverse=True)
            cumsum = 0.0
            keep = []
            for a in lora_adapters:
                keep.append(a)
                cumsum += a["weight"]
                if cumsum >= routing_topp:
                    break
            lora_adapters = keep
            first_adapter_model = f"openai/{lora_adapters[0]['name']}"
            adapter_strs = [f"{a['name']}:{a['weight']:.3f}" for a in lora_adapters]
            logger.info(
                f"[TopP] Kept {len(lora_adapters)} adapters "
                f"(cumsum={cumsum:.3f}, topp={routing_topp}): {adapter_strs}"
            )

        # When is_weighted=False, set all selected adapters to equal 1/N
        # (classifier ranking was only used for topk selection above).
        # Equal weighting preserves the same total LoRA magnitude as a
        # single adapter: y += sum_i (1/N) * B_i @ A_i @ x
        if not self.orchestrator_config.is_weighted and lora_adapters:
            equal_w = 1.0 / len(lora_adapters)
            for a in lora_adapters:
                a["weight"] = equal_w
        elif lora_adapters:
            # Renormalize weights (is_weighted=True)
            total_w = sum(a["weight"] for a in lora_adapters)
            if total_w > 0:
                for a in lora_adapters:
                    a["weight"] /= total_w

        if not lora_adapters:
            # All weight on base model skill -> use base model directly.
            # Clear any stale config file so the engine doesn't pick up
            # weighted LoRA from a prior routing decision.
            self._clear_weighted_lora()
            first_skill = self.orchestrator_config.skills[0]
            if first_skill.model:
                return first_skill.model
            return self.orchestrator_config.orchestrator_model

        # Get api_base
        api_base = None
        for skill_config in self.orchestrator_config.skills:
            if skill_config.llm_args and skill_config.llm_args.get("api_base"):
                api_base = skill_config.llm_args["api_base"]
                break
        if not api_base:
            api_base = (self.orchestrator_config.orchestrator_llm_args or {}).get(
                "api_base", "http://localhost:8080/v1"
            )

        # Call the weighted lora endpoint to set config
        server_url = api_base.rstrip("/").replace("/v1", "")
        merge_url = f"{server_url}/v1/create_weighted_lora"

        payload = {"adapters": lora_adapters}

        try:
            response = http_requests.post(merge_url, json=payload, timeout=30)
            if response.status_code == 200:
                logger.info(f"[WeightedLoRA] Config set: {lora_adapters}")
            else:
                logger.error(
                    f"[WeightedLoRA] Failed: {response.status_code} - {response.text}"
                )
                return None
        except http_requests.exceptions.RequestException as e:
            logger.error(f"[WeightedLoRA] Request failed: {e}")
            return None

        # Return first adapter as model name — triggers LoRA code path
        return first_adapter_model

    def _parse_routing_response(
        self, response: AssistantMessage
    ) -> SkillRoutingDecision:
        """Parse the orchestrator's response into a SkillRoutingDecision.

        Supports two formats:
        1. Reasoning-first: free text followed by "SELECTED_SKILL: skill_name"
        2. Legacy JSON: {"selected_skill": "...", "reasoning": "..."}
        """
        content = (response.content or "").strip()

        # Try reasoning-first format: look for SELECTED_SKILL: line
        import re
        match = re.search(r'SELECTED_SKILL:\s*(\S+)', content, re.IGNORECASE)
        if match:
            skill_name = match.group(1).strip().rstrip(".")
            # Extract reasoning as everything before the SELECTED_SKILL line
            reasoning = content[:match.start()].strip()
            # Truncate reasoning for logging
            short_reasoning = reasoning[-200:] if len(reasoning) > 200 else reasoning
            print(f"[LLM ROUTER] {skill_name} | full reasoning:\n{reasoning}\n---", flush=True)
            decision = SkillRoutingDecision(
                selected_skill=skill_name,
                reasoning=short_reasoning,
            )
            if decision.selected_skill in self.skill_backends:
                return decision
            logger.warning(
                f"Invalid skill in routing decision: {decision.selected_skill}. "
                f"Available: {list(self.skill_backends.keys())}"
            )
            return self._fallback_decision()

        # Fallback: try legacy JSON format
        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines).strip()

        try:
            data = json.loads(content)
            decision = SkillRoutingDecision(**data)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(
                f"Failed to parse routing decision ({e}), using fallback. "
                f"Raw response: {content[:300]}"
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
            skill_weights={first_skill: 1.0},
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

    def _weighted_generate(
        self, skill_weights: Dict[str, float], messages: list[Message],
        selected_skill: Optional[str] = None,
    ) -> AssistantMessage:
        """Generate using a weighted LoRA merge of multiple skills.

        Sets weighted LoRA config, then generates through the appropriate
        skill backend. Falls back to hard selection if config fails.

        If `selected_skill` is a base-model skill (no LoRA adapter),
        bypasses the weighted LoRA path entirely and generates directly
        with the base model to avoid residual adapter weights from the
        soft classifier distribution.
        """
        # If the selected skill is a base-model skill (no adapter),
        # generate directly without weighted LoRA.
        if selected_skill:
            is_base_model_skill = not any(
                s.adapter_name for s in self.orchestrator_config.skills
                if s.name == selected_skill
            )
            if is_base_model_skill:
                self._clear_weighted_lora()
                return self._single_skill_generate(selected_skill, messages)

        merged_model = self._set_weighted_lora(skill_weights)

        if merged_model is None:
            # All weight on base model skill or merge failed
            top_skill = max(skill_weights, key=skill_weights.get)
            return self._single_skill_generate(top_skill, messages)

        # Find the skill backend for the top LoRA adapter
        top_skill = max(
            ((name, w) for name, w in skill_weights.items()
             if any(s.adapter_name for s in self.orchestrator_config.skills
                    if s.name == name)),
            key=lambda x: x[1],
            default=(None, 0),
        )[0]

        if top_skill and top_skill in self.skill_backends:
            return self.skill_backends[top_skill].generate(messages)

        # Fallback: use first skill with llm_args
        top_skill = max(skill_weights, key=skill_weights.get)
        return self._single_skill_generate(top_skill, messages)

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

        # Step 2: For per_conversation, hot-swap so only the selected
        # adapter is loaded (once, on first routing decision) when the
        # effective number of adapters is 1.  This ensures the adapter
        # always occupies slot 0 on a clean slate, matching single-adapter
        # serving behavior exactly.
        topk = self.orchestrator_config.routing_topk
        effective_single = topk == 1
        if not effective_single and len(state.routing_history) == 1:
            # Check if topp filtering would leave only 1 adapter (not used for embedding)
            routing_topp = getattr(self.orchestrator_config, 'routing_topp', None)
            skill_weights = decision.skill_weights or {}
            if routing_topp and skill_weights and self.orchestrator_config.routing_mode != "embedding":
                adapter_weights = sorted(
                    (w for name, w in skill_weights.items()
                     if any(s.adapter_name for s in self.orchestrator_config.skills
                            if s.name == name) and w > 0),
                    reverse=True,
                )
                if adapter_weights:
                    cumsum = 0.0
                    n_kept = 0
                    for w in adapter_weights:
                        cumsum += w
                        n_kept += 1
                        if cumsum >= routing_topp:
                            break
                    effective_single = n_kept == 1

        # For embedding mode, single-skill selection → same path as topk=1
        if not effective_single and self.orchestrator_config.routing_mode == "embedding":
            if decision.skill_weights and len(decision.skill_weights) == 1:
                effective_single = True

        if (
            strategy == "per_conversation"
            and effective_single
            and len(state.routing_history) == 1  # first turn only
        ):
            self._hot_swap_adapter(decision.selected_skill)

        # Step 3: Generate using weighted LoRA (deterministic torch.matmul path).
        # Always prefer _weighted_generate for determinism — the standard Triton
        # lora_shrink uses SPLIT_K>1 atomic_add which is nondeterministic.
        # Construct skill_weights if missing (e.g. from fallback/skip_routing).
        # When effective_single, use {skill: 1.0} to match solo config path exactly.
        weights = decision.skill_weights
        if not weights or effective_single:
            weights = {decision.selected_skill: 1.0}

        # When is_weighted=False, keep classifier ranking for topk filtering
        # but set final weights to equal (1.0) for all selected adapters.
        # _set_weighted_lora does topk filtering using the weights, so we
        # pass them through for filtering, then equalize in _set_weighted_lora.
        # We signal this by storing the flag on the config (already there).

        assistant_message = self._weighted_generate(
            weights, full_messages, selected_skill=decision.selected_skill
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
