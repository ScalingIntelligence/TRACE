"""Orchestrator agent that routes to specialized LoRA skill adapters.

Mirrors the tau2-bench OrchestratorAgent: reads a YAML config, routes
each conversation to one skill via a classifier call, merges the
selected LoRA adapter into base weights, then generates using the
merged model through the standard OpenAI-compatible API.

Configuration is loaded from the YAML file pointed to by the
``ORCHESTRATOR_CONFIG`` environment variable.  The YAML schema is
identical to the ``orchestrator:`` section used by tau2-bench
single_lora configs.

Usage:
    ORCHESTRATOR_CONFIG=path/to/config.yml \\
    OPENAI_API_KEY=EMPTY \\
    tool_sandbox --agent Orchestrator --user VLLM ...
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Iterable, List, Literal, Optional, Union, cast

import requests as http_requests
import yaml
from loguru import logger
from openai import NOT_GIVEN, NotGiven, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from requests.exceptions import HTTPError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from tool_sandbox.common.execution_context import RoleType, get_current_context
from tool_sandbox.common.message_conversion import (
    Message,
    openai_tool_call_to_python_code,
    to_openai_messages,
)
from tool_sandbox.common.tool_conversion import convert_to_openai_tools
from tool_sandbox.common.utils import all_logging_disabled
from tool_sandbox.roles.base_role import BaseRole


# ---------------------------------------------------------------------------
# Configuration data classes (plain dicts — no pydantic dependency needed)
# ---------------------------------------------------------------------------


def _parse_skill(raw: dict) -> dict:
    """Normalise a skill entry from the YAML config."""
    return {
        "name": raw["name"],
        "description": raw.get("description", ""),
        "adapter_path": raw.get("adapter_path"),
        "adapter_name": raw.get("adapter_name"),
        "model": raw.get("model"),
        "llm_args": raw.get("llm_args", {}),
    }


# ---------------------------------------------------------------------------
# OrchestratorVLLMAgent
# ---------------------------------------------------------------------------


class OrchestratorVLLMAgent(BaseRole):
    """Agent that routes conversations to specialised LoRA skill adapters.

    On the first ``respond()`` call that contains a user message the
    orchestrator picks a skill (via a classifier forward-pass on a
    *separate* base-model server) and merges the corresponding LoRA
    adapter into the agent server's base weights.  All subsequent turns
    in the same scenario reuse that adapter (per-conversation routing).
    """

    role_type: RoleType = RoleType.AGENT

    # Class-level: which adapter is currently merged on the agent server.
    _merge_active_adapter: Optional[str] = None

    def __init__(self) -> None:
        config_path = os.environ.get("ORCHESTRATOR_CONFIG")
        if not config_path:
            raise RuntimeError(
                "ORCHESTRATOR_CONFIG env var must point to a YAML config file"
            )
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        orch = raw["orchestrator"]

        # ---- skills ----
        self.skills: list[dict] = [_parse_skill(s) for s in orch["skills"]]
        self.skill_by_name: dict[str, dict] = {s["name"]: s for s in self.skills}

        # ---- base model name (used for chat completions) ----
        self.base_model_name: str = orch["orchestrator_model"]
        if self.base_model_name.startswith("openai/"):
            self.base_model_name = self.base_model_name[7:]

        # ---- routing config ----
        self.routing_strategy: str = orch.get("routing_strategy", "per_conversation")
        self.routing_mode: str = orch.get("routing_mode", "classifier")
        self.skip_routing: Optional[str] = orch.get("skip_routing")
        routing_ctx = orch.get("routing_context_window")
        self.routing_context_window: Optional[int] = (
            int(routing_ctx) if routing_ctx is not None else None
        )
        self.classifier_temperature: float = float(
            orch.get("classifier_temperature", 1.0)
        )

        # ---- orchestrator (classifier) client — runs on the *user* server ----
        orch_llm = orch.get("orchestrator_llm_args", {})
        orch_base = orch_llm.get("api_base", "http://localhost:9001/v1")
        orch_key = orch_llm.get("api_key", os.environ.get("OPENAI_API_KEY", "EMPTY"))
        self.orch_client = OpenAI(base_url=orch_base, api_key=orch_key)
        self.orch_temperature = float(orch_llm.get("temperature", 0.0))
        self.orch_seed: Optional[int] = (
            int(orch_llm["seed"]) if "seed" in orch_llm else None
        )

        # ---- agent (generation) client — runs on the adapter-merge server ----
        # All skills share the same server; pick api_base from the first skill.
        first_llm = self.skills[0].get("llm_args", {})
        agent_base = first_llm.get(
            "api_base",
            os.environ.get("VLLM_AGENT_URL", "http://localhost:9000/v1"),
        )
        agent_key = first_llm.get(
            "api_key", os.environ.get("OPENAI_API_KEY", "EMPTY")
        )
        self.agent_client = OpenAI(base_url=agent_base, api_key=agent_key)
        self.agent_base_url = agent_base  # for merge HTTP calls

        # Generation defaults (from first skill, overridable per-skill)
        self.agent_temperature = float(first_llm.get("temperature", 0.0))
        self.agent_seed: Optional[int] = (
            int(first_llm["seed"]) if "seed" in first_llm else None
        )

        # ---- classifier label mapping ----
        self._label_to_skill: dict[str, str] = {}
        self._skill_to_label: dict[str, str] = {}
        label_lines: list[str] = []
        for i, skill in enumerate(self.skills):
            label = chr(ord("A") + i)
            self._label_to_skill[label] = skill["name"]
            self._skill_to_label[skill["name"]] = label
            label_lines.append(f'{label}: {skill["name"]} - {skill["description"]}')
        self._classifier_labels = list(self._label_to_skill.keys())
        self._classifier_system_prompt = (
            "You are a routing classifier for a customer service agent system.\n"
            "Given the current conversation, select the skill that should handle "
            "the agent's next response.\n\n"
            + "\n".join(label_lines)
            + "\n\nOnly output the label (e.g. A, B, C). Do not output anything else."
        )

        # ---- per-scenario routing state (reset between scenarios) ----
        self._conversation_route: Optional[dict] = None
        self._route_count: int = 0

        # ---- optional extra system prompt ----
        extra_prompt_path = os.environ.get("TSB_EXTRA_SYSTEM_PROMPT", "")
        self._extra_system_prompt = ""
        if extra_prompt_path and os.path.isfile(extra_prompt_path):
            with open(extra_prompt_path) as f:
                self._extra_system_prompt = f.read().strip()

    # ------------------------------------------------------------------
    # Reset between scenarios
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._conversation_route = None
        self._route_count = 0

    # ------------------------------------------------------------------
    # Weight merge
    # ------------------------------------------------------------------

    def _weight_merge_adapter(
        self,
        skill_name: str,
        skill_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Merge LoRA adapter(s) into base model weights via the vLLM endpoint."""
        skill = self.skill_by_name.get(skill_name)
        if skill is None:
            logger.warning(f"[WeightMerge] Skill '{skill_name}' not found")
            return
        if not skill.get("adapter_path"):
            # Base-model skill — restore original weights.
            self._do_restore()
            return

        merge_key = skill["adapter_path"]

        if merge_key == OrchestratorVLLMAgent._merge_active_adapter:
            logger.info("[WeightMerge] Already merged, skipping")
            return

        server_url = self.agent_base_url.rstrip("/").replace("/v1", "")
        payload = {"adapter_path": skill["adapter_path"]}

        logger.info(f"[WeightMerge] Merging: {payload}")
        try:
            resp = http_requests.post(
                f"{server_url}/v1/merge_lora_into_base",
                json=payload,
                timeout=600,
            )
            if resp.status_code == 200:
                logger.info(f"[WeightMerge] {resp.text[:120]}")
            else:
                logger.error(
                    f"[WeightMerge] Failed: {resp.status_code} {resp.text[:200]}"
                )
                return
        except http_requests.exceptions.RequestException as e:
            logger.error(f"[WeightMerge] Request failed: {e}")
            return

        OrchestratorVLLMAgent._merge_active_adapter = merge_key

    def _do_restore(self) -> None:
        server_url = self.agent_base_url.rstrip("/").replace("/v1", "")
        logger.info("[WeightMerge] Restoring base weights")
        try:
            http_requests.post(f"{server_url}/v1/restore_base_weights", timeout=300)
        except http_requests.exceptions.RequestException as e:
            logger.error(f"[WeightMerge] Restore failed: {e}")
        OrchestratorVLLMAgent._merge_active_adapter = None

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route(
        self, openai_messages: list[dict],
    ) -> dict:
        """Pick a skill for this conversation."""
        if self.skip_routing:
            name = self.skip_routing.strip()
            return {"selected_skill": name, "skill_weights": {name: 1.0}}

        if len(self.skills) == 1:
            name = self.skills[0]["name"]
            return {"selected_skill": name, "skill_weights": {name: 1.0}}

        return self._route_classifier(openai_messages)

    def _route_classifier(
        self, openai_messages: list[dict],
    ) -> dict:
        """One-token classifier routing using structured_outputs + logprobs."""
        # Build context window
        non_system = [m for m in openai_messages if m.get("role") != "system"]
        window = self.routing_context_window
        recent = non_system[-window:] if window is not None else non_system

        # Extract last user text for the classifier query
        query_text = ""
        for m in reversed(recent):
            if m.get("role") == "user" and m.get("content"):
                query_text = m["content"]
                break

        labels = self._classifier_labels
        completion = None
        try:
            completion = self.orch_client.chat.completions.create(
                model=self.base_model_name,
                messages=[
                    {"role": "system", "content": self._classifier_system_prompt},
                    *recent,
                    {"role": "user", "content": "Which skill?"},
                ],
                extra_body={"structured_outputs": {"choice": labels}},
                logprobs=True,
                top_logprobs=max(len(labels) - 1, 1),
                temperature=self.orch_temperature,
                max_tokens=1,
                seed=self.orch_seed,
            )
            chosen_label = completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Classifier call failed: {e}")
            name = self.skills[0]["name"]
            return {"selected_skill": name, "skill_weights": {name: 1.0}}

        # Extract logprobs → skill_weights
        skill_weights: Dict[str, float] = {}
        logprobs_data = getattr(completion.choices[0], "logprobs", None)
        label_logprobs: Dict[str, float] = {}
        if logprobs_data and logprobs_data.content:
            token_lp = logprobs_data.content[0]
            label_logprobs[token_lp.token.strip()] = token_lp.logprob
            if token_lp.top_logprobs:
                for alt in token_lp.top_logprobs:
                    label_logprobs[alt.token.strip()] = alt.logprob

        if label_logprobs:
            cls_temp = self.classifier_temperature
            exp_probs: Dict[str, float] = {}
            for lab, lp in label_logprobs.items():
                if lab in self._label_to_skill:
                    exp_probs[lab] = (
                        math.exp(lp / cls_temp) if cls_temp > 0
                        else (1.0 if lp == max(label_logprobs.values()) else 0.0)
                    )
            total = sum(exp_probs.values())
            if total > 0:
                for lab, prob in exp_probs.items():
                    skill_weights[self._label_to_skill[lab]] = prob / total

        selected = (
            self._label_to_skill[chosen_label]
            if chosen_label in self._label_to_skill
            else self.skills[0]["name"]
        )
        logger.info(f"[Classifier] label={chosen_label} → {selected}")
        return {"selected_skill": selected, "skill_weights": skill_weights}

    # ------------------------------------------------------------------
    # respond — main entry point
    # ------------------------------------------------------------------

    def respond(self, ending_index: Optional[int] = None) -> None:
        messages: List[Message] = self.get_messages(ending_index=ending_index)
        response_messages: List[Message] = []
        self.messages_validation(messages=messages)
        messages = self.filter_messages(messages=messages)

        # Don't respond to system
        if messages[-1].sender == RoleType.SYSTEM:
            return

        # ---- tools ----
        available_tools = self.get_available_tools()
        available_tool_names = set(available_tools.keys())
        openai_tools: Any = (
            convert_to_openai_tools(available_tools)
            if messages[-1].sender in (RoleType.USER, RoleType.EXECUTION_ENVIRONMENT)
            else NOT_GIVEN
        )
        openai_tools = cast(
            Union[Iterable[ChatCompletionToolParam], NotGiven], openai_tools
        )

        # ---- convert messages ----
        current_context = get_current_context()
        openai_messages, _ = to_openai_messages(messages)
        if self._extra_system_prompt:
            for msg in openai_messages:
                if msg.get("role") == "system":
                    msg["content"] = msg["content"] + "\n\n" + self._extra_system_prompt
                    break

        # ---- routing (first user message only for per_conversation) ----
        if self.routing_strategy == "per_conversation" and self._conversation_route is not None:
            decision = self._conversation_route
        else:
            decision = self._route(openai_messages)
            if self.routing_strategy == "per_conversation":
                self._conversation_route = decision

        self._route_count += 1
        skill_name = decision["selected_skill"]

        # ---- merge adapter (first turn only for per_conversation) ----
        if self._route_count == 1 or self.routing_strategy == "per_turn":
            self._weight_merge_adapter(skill_name, decision.get("skill_weights"))

        # ---- generate ----
        with all_logging_disabled():
            response = self.agent_client.chat.completions.create(
                model=self.base_model_name,
                messages=cast(list[ChatCompletionMessageParam], openai_messages),
                tools=openai_tools,
                temperature=self.agent_temperature,
                seed=self.agent_seed,
            )

        # ---- parse response (same as OpenAIAPIAgent) ----
        openai_response_message = response.choices[0].message
        if not openai_response_message.tool_calls:
            assert openai_response_message.content is not None
            response_messages = [
                Message(
                    sender=self.role_type,
                    recipient=RoleType.USER,
                    content=openai_response_message.content,
                )
            ]
        else:
            assert openai_tools is not NOT_GIVEN
            for tool_call in openai_response_message.tool_calls:
                execution_facing_tool_name = (
                    current_context.get_execution_facing_tool_name(
                        tool_call.function.name
                    )
                )
                response_messages.append(
                    Message(
                        sender=self.role_type,
                        recipient=RoleType.EXECUTION_ENVIRONMENT,
                        content=openai_tool_call_to_python_code(
                            tool_call,
                            available_tool_names,
                            execution_facing_tool_name=execution_facing_tool_name,
                        ),
                        openai_tool_call_id=tool_call.id,
                        openai_function_name=tool_call.function.name,
                    )
                )
        self.add_messages(response_messages)
