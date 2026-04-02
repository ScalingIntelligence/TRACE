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
    routing_mode: str = "llm"  # "llm", "llm_multi", "classifier", "embedding", or "rac"
    routing_topk: Optional[int] = None  # limit routing to top-k adapters; applies to classifier and embedding modes (None=all)
    routing_topp: Optional[float] = None  # top-p nucleus filtering: keep fewest adapters whose cumulative weight >= topp; applies to classifier and embedding
    classifier_temperature: float = 1.0  # temperature scaling applied to logprobs before softmax (>1 = softer, <1 = sharper)
    # Embedding-based routing
    embedding_model: Optional[str] = None  # e.g. "Qwen/Qwen3-Embedding-8B"; required when routing_mode="embedding"
    embedding_gpu: Optional[str] = None  # device for the embedding model, e.g. "cuda:0" (default: "cpu")
    embedding_threshold: float = 0.5  # similarity threshold; skills above this are activated
    # RAC (Retrieval-Augmented Classification) routing
    rac_corpus_path: Optional[str] = None  # path to gold_trajectories JSON corpus
    rac_topk: int = 5  # number of trajectories to retrieve per query
    rac_threshold: float = 0.5  # minimum similarity to include a skill as candidate


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
        base_model_name: Optional[str] = None,
    ):
        self.config = config
        self.tools = tools
        self.domain_policy = domain_policy

        # Resolve model name: adapter-backed skills use the base model name
        # because adapters are merged into base weights via /v1/merge_lora_into_base.
        if config.adapter_name and base_model_name:
            self.model = base_model_name
        elif config.model:
            self.model = config.model
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

_ROUTING_EXAMPLES_PATH = __import__("os").path.join(
    __import__("os").path.dirname(__file__),
    "..", "..", "..", "evals", "benchmarks", "tau2_bench_eval", "routing_examples.txt",
)
try:
    with open(_ROUTING_EXAMPLES_PATH) as _f:
        _ROUTING_EXAMPLES = _f.read()
except FileNotFoundError:
    _ROUTING_EXAMPLES = ""

ROUTING_SYSTEM_PROMPT = """You are a routing expert. Given the conversation, select which skill the agent needs. Match the conversation to the most similar training example.

""" + _ROUTING_EXAMPLES + """

Available skills:
{skill_descriptions}

Based on the conversation, which training example pattern does this most closely match?
Output: SELECTED_SKILL: skill_name
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

    # Class-level: persists across instances (new agent created per task).
    # Tracks which adapter is currently merged into base weights on the server.
    _merge_active_adapter: Optional[str] = None

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
                base_model_name=orchestrator_config.orchestrator_model,
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

        # RAC (Retrieval-Augmented Classification) mode setup
        if orchestrator_config.routing_mode == "rac":
            import json as _json

            # Load embedding model (reuse singleton)
            emb_model_name = orchestrator_config.embedding_model or "Qwen/Qwen3-Embedding-8B"
            emb_device = orchestrator_config.embedding_gpu or "cpu"
            self._rac_embedding_model = _get_embedding_model(emb_model_name, emb_device)

            # Load and embed corpus (one-time)
            corpus_path = orchestrator_config.rac_corpus_path
            if not corpus_path:
                raise ValueError("rac_corpus_path is required when routing_mode='rac'")
            with open(corpus_path) as f:
                self._rac_corpus = _json.load(f)
            corpus_texts = [t["text"] for t in self._rac_corpus]
            self._rac_corpus_embeddings = self._rac_embedding_model.encode(corpus_texts)
            logger.info(
                f"[RAC] Loaded corpus: {len(self._rac_corpus)} trajectories from {corpus_path}"
            )

            # Store skill descriptions for the classifier prompt
            self._rac_skill_descs = {
                s.name: s.description.strip() for s in orchestrator_config.skills
            }

            # Build label-to-skill mapping (same as classifier mode)
            self._label_to_skill = {}
            self._skill_to_label = {}
            for i, skill_config in enumerate(orchestrator_config.skills):
                label = chr(ord("A") + i)
                self._label_to_skill[label] = skill_config.name
                self._skill_to_label[skill_config.name] = label
            self._classifier_labels = list(self._label_to_skill.keys())

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
        return OrchestratorAgentState(
            system_messages=[
                SystemMessage(role="system", content=self.system_prompt)
            ],
            messages=message_history,
        )


    def _weight_merge_adapter(self, skill_name: str, skill_weights: Optional[Dict[str, float]] = None):
        """Merge LoRA adapter(s) into base model weights.

        Single adapter: merges the named skill's adapter with weight 1.0.
        Multi adapter (skill_weights provided): merges all adapter-backed
        skills with their classifier weights, sorted by path for determinism.

        Uses /v1/merge_lora_into_base endpoint. For base-model skills
        (no adapter), restores original weights.
        """
        import requests as http_requests

        # Build skill→config lookup
        skill_map = {s.name: s for s in self.orchestrator_config.skills}

        # Determine api_base
        api_base = None
        for s in self.orchestrator_config.skills:
            if s.llm_args and s.llm_args.get("api_base"):
                api_base = s.llm_args["api_base"]
                break
        if not api_base:
            api_base = (self.orchestrator_config.orchestrator_llm_args or {}).get(
                "api_base", "http://localhost:8080/v1"
            )
        server_url = api_base.rstrip("/").replace("/v1", "")

        # Build adapter list
        if skill_weights and len(skill_weights) > 1:
            # Multi-adapter merge: include all adapter-backed skills with weight > 0
            adapters = []
            for sname, weight in skill_weights.items():
                sc = skill_map.get(sname)
                if sc and sc.adapter_path and sc.adapter_name and weight > 0:
                    adapters.append({"adapter_path": sc.adapter_path, "weight": weight})
            if not adapters:
                # All weight on base-model skill → restore
                self._do_restore(server_url)
                return
            merge_key = str(sorted((a["adapter_path"], a["weight"]) for a in adapters))
        else:
            # Single adapter merge
            sc = skill_map.get(skill_name)
            if sc is None:
                logger.warning(f"[WeightMerge] Skill '{skill_name}' not found")
                return
            if not sc.adapter_path or not sc.adapter_name:
                self._do_restore(server_url)
                return
            adapters = None  # use single-path format
            merge_key = sc.adapter_path

        # Skip if same merge is already active
        if merge_key == self._merge_active_adapter:
            logger.info(f"[WeightMerge] Already merged, skipping")
            return

        # Call merge endpoint
        if adapters:
            payload = {"adapters": adapters}
        else:
            payload = {"adapter_path": skill_map[skill_name].adapter_path}

        logger.info(f"[WeightMerge] Merging: {payload}")
        try:
            resp = http_requests.post(
                f"{server_url}/v1/merge_lora_into_base",
                json=payload,
                timeout=600,
            )
            if resp.status_code == 200:
                logger.info(f"[WeightMerge] {resp.text[:120]}")
                print(f"[WEIGHT_MERGE] {resp.text[:120]}")
            else:
                logger.error(f"[WeightMerge] Failed: {resp.status_code} {resp.text[:200]}")
                return
        except http_requests.exceptions.RequestException as e:
            logger.error(f"[WeightMerge] Request failed: {e}")
            return

        OrchestratorAgent._merge_active_adapter = merge_key

    def _do_restore(self, server_url: str):
        """Restore base weights (helper for _weight_merge_adapter)."""
        import requests as http_requests
        logger.info("[WeightMerge] Restoring base weights")
        try:
            resp = http_requests.post(f"{server_url}/v1/restore_base_weights", timeout=300)
            if resp.status_code == 200:
                print(f"[WEIGHT_MERGE] Restored base weights")
            else:
                logger.error(f"[WeightMerge] Restore failed: {resp.status_code}")
        except http_requests.exceptions.RequestException as e:
            logger.error(f"[WeightMerge] Restore failed: {e}")
        OrchestratorAgent._merge_active_adapter = None

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

        if self.orchestrator_config.routing_mode == "rac":
            return self._route_rac(messages)

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

    def _route_rac(self, messages: list[Message]) -> SkillRoutingDecision:
        """Route using Retrieval-Augmented Classification.

        Stage 1: Embed user message → retrieve top-K similar trajectories from corpus
        Stage 2: Deduplicate to unique candidate skills (best trajectory per skill)
        Stage 3: Classifier picks from candidates using descriptions + retrieved examples

        Returns SkillRoutingDecision with skill_weights from classifier logprobs.
        """
        from openai import OpenAI

        orch_args = self.orchestrator_config.orchestrator_llm_args or {}
        api_base = orch_args.get("api_base", "http://localhost:8080/v1")
        api_key = orch_args.get("api_key", "EMPTY")
        model_name = self.orchestrator_config.orchestrator_model
        if model_name.startswith("openai/"):
            model_name = model_name[len("openai/"):]

        # --- Stage 1: Retrieve from corpus ---
        # Extract the user message text for query embedding
        window = self.orchestrator_config.routing_context_window
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]
        recent = non_system[-window:] if window is not None else non_system
        # Use the last user message as the retrieval query
        query_text = ""
        for m in reversed(recent):
            if m.role == "user" and getattr(m, "content", None):
                query_text = m.content
                break
        if not query_text:
            logger.warning("[RAC] No user message found for retrieval, using fallback")
            return self._fallback_decision()

        query_emb = self._rac_embedding_model.encode(
            [query_text],
            prompt="Instruct: Given a customer service request, identify which skill category best handles it\nQuery: ",
        )
        similarities = self._rac_embedding_model.similarity(
            query_emb, self._rac_corpus_embeddings
        )[0]
        top_indices = similarities.argsort(descending=True)[: self.orchestrator_config.rac_topk]

        # --- Stage 2: Filter by threshold, deduplicate to best trajectory per skill ---
        threshold = self.orchestrator_config.rac_threshold
        candidates: dict[str, tuple[float, str]] = {}  # skill -> (score, text)
        for idx in top_indices:
            score = float(similarities[idx])
            if score < threshold:
                continue
            skill = self._rac_corpus[idx]["skill"]
            if skill not in candidates or score > candidates[skill][0]:
                candidates[skill] = (score, self._rac_corpus[idx]["text"])

        # Fallback: if nothing passes threshold, take the single best match
        if not candidates:
            best_idx = int(top_indices[0])
            skill = self._rac_corpus[best_idx]["skill"]
            candidates[skill] = (
                float(similarities[best_idx]),
                self._rac_corpus[best_idx]["text"],
            )

        cand_str = " ".join(f"{s}:{sc:.2f}" for s, (sc, _) in candidates.items())
        logger.info(f"[RAC] Candidates: [{cand_str}]  n={len(candidates)}")

        # --- Stage 3: Classify among candidates ---
        if len(candidates) == 1:
            skill_name = list(candidates.keys())[0]
            print(
                f"[RAC] single candidate: {skill_name} | {cand_str}",
                flush=True,
            )
            return SkillRoutingDecision(
                selected_skill=skill_name,
                reasoning=f"rac single candidate ({cand_str})",
                skill_weights={skill_name: 1.0},
            )

        # Build classifier prompt with score-sorted candidates (best = label A)
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)
        label_to_skill: dict[str, str] = {}
        prompt_lines = []
        for i, (skill, (score, example)) in enumerate(sorted_candidates):
            label = chr(ord("A") + i)
            label_to_skill[label] = skill
            desc = self._rac_skill_descs.get(skill, "")
            prompt_lines.append(f"{label}: {skill} (relevance: {score:.2f}) — {desc}")
            prompt_lines.append(f"   Similar scenario: {example}")

        classifier_prompt = (
            "You are a routing classifier. Select which skill best matches "
            "the customer's request.\n\n"
            + "\n".join(prompt_lines)
            + "\n\nOnly output the label."
        )

        labels = list(label_to_skill.keys())

        try:
            client = OpenAI(base_url=api_base, api_key=api_key)
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": classifier_prompt},
                    {"role": "user", "content": query_text},
                    {"role": "user", "content": "Which skill?"},
                ],
                extra_body={"structured_outputs": {"choice": labels}},
                logprobs=True,
                top_logprobs=max(len(labels) - 1, 1),
                temperature=orch_args.get("temperature", 0.0),
                max_tokens=1,
                seed=orch_args.get("seed"),
            )

            chosen_label = completion.choices[0].message.content.strip()

            # Extract logprobs for all candidate labels
            label_logprobs: Dict[str, float] = {}
            logprobs_data = completion.choices[0].logprobs
            if logprobs_data and logprobs_data.content:
                token_logprob = logprobs_data.content[0]
                label_logprobs[token_logprob.token.strip()] = token_logprob.logprob
                if token_logprob.top_logprobs:
                    for alt in token_logprob.top_logprobs:
                        label_logprobs[alt.token.strip()] = alt.logprob

            # Compute raw primary probability for confidence gating
            if label_logprobs:
                raw_probs = {l: math.exp(lp) for l, lp in label_logprobs.items()}
                raw_total = sum(raw_probs.values())
                if raw_total > 0:
                    raw_probs = {l: p / raw_total for l, p in raw_probs.items()}
                self._last_primary_raw_prob = max(raw_probs.values()) if raw_probs else 1.0
            else:
                self._last_primary_raw_prob = 1.0

            # Convert to skill weights via temperature-scaled softmax
            cls_temp = getattr(self.orchestrator_config, "classifier_temperature", 1.0)
            skill_weights: Dict[str, float] = {}
            if label_logprobs:
                exp_probs = {}
                for label, lp in label_logprobs.items():
                    if label in label_to_skill:
                        exp_probs[label] = (
                            math.exp(lp / cls_temp)
                            if cls_temp > 0
                            else (1.0 if lp == max(label_logprobs.values()) else 0.0)
                        )
                total = sum(exp_probs.values())
                if total > 0:
                    for label, prob in exp_probs.items():
                        skill_weights[label_to_skill[label]] = prob / total

            # Also include candidates that weren't in logprobs with zero weight
            for skill in candidates:
                if skill not in skill_weights:
                    skill_weights[skill] = 0.0

            weight_str = ", ".join(f"{s}:{w:.4f}" for s, w in sorted(skill_weights.items()))
            logger.info(f"[RAC] label={chosen_label} weights=[{weight_str}]")
            print(
                f"[RAC] label={chosen_label} | {weight_str} | candidates=[{cand_str}]",
                flush=True,
            )

        except Exception as e:
            logger.error(f"RAC classifier call failed: {e}")
            return self._fallback_decision()

        if chosen_label in label_to_skill:
            skill_name = label_to_skill[chosen_label]
            # Apply routing_topk: when topk=1, use hard routing (single adapter)
            topk = self.orchestrator_config.routing_topk
            if topk is not None and topk == 1:
                skill_weights = {skill_name: 1.0}
            elif topk is not None and topk < len(skill_weights):
                sorted_sw = sorted(skill_weights.items(), key=lambda x: x[1], reverse=True)
                kept = dict(sorted_sw[:topk])
                total = sum(kept.values())
                if total > 0:
                    skill_weights = {s: w / total for s, w in kept.items()}
            return SkillRoutingDecision(
                selected_skill=skill_name,
                reasoning=f"rac label {chosen_label} ({cand_str})",
                skill_weights=skill_weights,
            )
        else:
            logger.warning(
                f"[RAC] Unexpected label: '{chosen_label}'. Expected {labels}"
            )
            return self._fallback_decision()

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
            # Route fresh (always for per_turn, or first call for per_conversation/triggered)
            decision = self._route(full_messages)
            if strategy in ("per_conversation", "per_turn_triggered"):
                state.conversation_route = decision
                logger.info(
                    f"[Orchestrator] Locked initial route: "
                    f"{decision.selected_skill}"
                )

        state.routing_history.append(decision)
        logger.info(
            f"[Orchestrator] -> {decision.selected_skill}"
            + (f" (reason: {decision.reasoning})" if decision.reasoning else "")
        )
        print(f"[ROUTING] Task routing -> {decision.selected_skill} (reason: {decision.reasoning})")

        # Step 2 + 3: Merge adapter(s) into base model weights, then generate.
        # Merge is per-conversation (first turn only) or per-turn.
        if len(state.routing_history) == 1 or strategy == "per_turn":
            self._weight_merge_adapter(
                decision.selected_skill,
                skill_weights=decision.skill_weights,
            )
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
