"""
Multi-Step Sequence Game - Tests pure multi-step planning capability.

Trains the skill of executing dependent tool call sequences where:
- Each step's output provides IDs needed for the next step
- Final action requires correct parameters gathered from the chain
- No conditionals, no verification, no termination ambiguity

Distribution matches exact tau-bench multi-step failure distribution.
"""

import re
import random
import json
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field


# =============================================================================
# Exact distribution from tau-bench multi-step failures (85 failures, 2+ steps)
# =============================================================================

STEP_COUNTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20]
STEP_WEIGHTS = [12, 6, 9, 17, 11, 8, 3, 1, 5, 3, 5, 4, 1]  # Raw counts from tau-bench

# Final action type distribution (abstracted from tau-bench)
ACTION_TYPES = [
    "file_report",       # Maps to modify/update (38%)
    "transfer_item",     # Maps to exchange/return (22%)
    "close_case",        # Maps to cancel (10%)
    "open_case",         # Maps to create/book (8%)
    "issue_warrant",     # Maps to issue_credit (4%)
    "summarize",         # Maps to info_only (9%)
    "escalate",          # Maps to transfer (3%)
    "archive",           # Maps to other (6%)
]
ACTION_WEIGHTS = [38, 22, 10, 8, 4, 9, 3, 6]


# =============================================================================
# Data Generation
# =============================================================================

def generate_id(rng: random.Random, prefix: str) -> str:
    """Generate a random ID with given prefix."""
    return f"{prefix}-{rng.randint(1000, 9999)}"


def generate_name(rng: random.Random) -> str:
    """Generate a random name."""
    first_names = ["Chen", "Park", "Silva", "Jordan", "Morgan", "Taylor", "Kim", "Patel",
                   "Garcia", "Wilson", "Brown", "Davis", "Martinez", "Anderson", "Thomas"]
    return rng.choice(first_names)


@dataclass
class Record:
    """A database record with fields that may reference other records."""
    record_id: str
    record_type: str
    fields: Dict[str, Any]


@dataclass
class ChainStep:
    """One step in the required tool call chain."""
    tool_name: str
    params: Dict[str, Any]
    result: Dict[str, Any]
    extracts: Dict[str, str]  # field_name -> extracted_value (for next step)


@dataclass
class GroundTruth:
    """The correct sequence of tool calls."""
    chain: List[ChainStep]
    final_action: str
    final_params: Dict[str, Any]
    case_id: str


# =============================================================================
# Request Templates (Natural language, no skill hints)
# =============================================================================

# Request templates - Explicit about first step (query person by name)
# num_steps = number of IDs to collect = query + chain_records
REQUEST_TEMPLATES = {
    2: [
        # 2-step: query person only, then final action (no chain records)
        "Case {case_id}: Query person \"{name}\" and {action} with their record ID.",
        "Case {case_id}: Look up person \"{name}\" in the database. {action_cap} with the person's ID.",
        "Case {case_id}: Search for person \"{name}\" and {action} including the person ID.",
    ],
    3: [
        # 3-step: query person + 1 chain record (submission)
        "Case {case_id}: Query person \"{name}\", then get their submission record. {action_cap} with both IDs.",
        "Case {case_id}: Look up person \"{name}\" and retrieve their submission. {action_cap} with both record IDs.",
        "Case {case_id}: Find person \"{name}\" in records, get their linked submission, and {action} with both IDs.",
    ],
    4: [
        # 4-step: query person + 2 chain records
        "Case {case_id}: Query person \"{name}\", get their submission, then examine the linked item. {action_cap} with all 3 IDs.",
        "Case {case_id}: Search for person \"{name}\", retrieve their submission, examine the item it links to. {action_cap} with all IDs.",
        "Case {case_id}: Look up person \"{name}\", trace: submission -> item. {action_cap} with all 3 collected IDs.",
    ],
    5: [
        # 5-step: query person + 3 chain records
        "Case {case_id}: Query person \"{name}\" and follow a 3-hop reference chain (submission -> item -> evidence). {action_cap} with all 4 IDs.",
        "Case {case_id}: Search for person \"{name}\", trace through submission and 2 linked items. {action_cap} with all IDs.",
        "Case {case_id}: Find person \"{name}\", get submission, follow 2 item links. {action_cap} with all 4 IDs.",
    ],
    6: [
        # 6-step: query person + 4 chain records
        "Case {case_id}: Query person \"{name}\" and follow a 4-hop reference chain. {action_cap} with all 5 IDs.",
        "Case {case_id}: Search for \"{name}\", trace: submission -> 3 linked items. {action_cap} with all IDs.",
    ],
    7: [
        # 7-step: query person + 5 chain records
        "Case {case_id}: Query person \"{name}\" and follow a 5-hop reference chain. {action_cap} with all 6 IDs.",
        "Case {case_id}: Look up person \"{name}\", trace through 5 chain records. {action_cap} with all IDs.",
    ],
}

# For longer chains - explicit about chain length
LONG_CHAIN_TEMPLATE = "Case {case_id}: Query person \"{name}\" and trace a {chain_len}-hop reference chain. {action_cap} with all collected IDs."


# =============================================================================
# Tool Definitions
# =============================================================================

TOOL_SPECS = [
    {
        "name": "query_records",
        "description": "Search for a record by type and name/identifier",
        "parameters": ["query_type", "search_term"]
    },
    {
        "name": "get_record",
        "description": "Retrieve full details of a record by its ID",
        "parameters": ["record_id"]
    },
    {
        "name": "examine_item",
        "description": "Get detailed information about an item including its references",
        "parameters": ["item_id"]
    },
    {
        "name": "file_report",
        "description": "File a report for a case with collected evidence IDs",
        "parameters": ["case_id", "evidence_ids"]
    },
    {
        "name": "close_case",
        "description": "Close a case with resolution",
        "parameters": ["case_id", "resolution_ids"]
    },
    {
        "name": "archive",
        "description": "Archive a case with reference chain",
        "parameters": ["case_id", "chain_ids"]
    },
    {
        "name": "issue_warrant",
        "description": "Issue a warrant based on collected evidence",
        "parameters": ["case_id", "evidence_ids"]
    },
    {
        "name": "transfer_item",
        "description": "Transfer an item between records",
        "parameters": ["case_id", "item_ids"]
    },
    {
        "name": "open_case",
        "description": "Open a new case based on findings",
        "parameters": ["reference_ids"]
    },
    {
        "name": "summarize",
        "description": "Summarize findings for a case",
        "parameters": ["case_id", "finding_ids"]
    },
    {
        "name": "escalate",
        "description": "Escalate case to supervisor with evidence",
        "parameters": ["case_id", "evidence_ids"]
    },
]


# =============================================================================
# Game State
# =============================================================================

@dataclass
class GameState:
    """Complete state of a game episode."""
    case_id: str
    target_name: str
    records: Dict[str, Record]
    ground_truth: GroundTruth

    # Tracking
    tool_calls: List[Dict[str, Any]]
    collected_ids: List[str]
    current_step: int

    # Completion
    final_action_taken: bool
    final_action_name: Optional[str]
    final_action_params: Optional[Dict[str, Any]]
    done: bool


# =============================================================================
# Chain Generator
# =============================================================================

class ChainGenerator:
    """Generates the dependency chain for a given step count."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def generate(self, num_steps: int, case_id: str, target_name: str) -> Tuple[Dict[str, Record], GroundTruth]:
        """Generate database and ground truth for num_steps chain."""

        records = {}
        chain = []
        collected_ids = []

        # Step 1: Always query for the person
        person_id = generate_id(self.rng, "PER")
        submission_id = generate_id(self.rng, "SUB")

        person_record = Record(
            record_id=person_id,
            record_type="person",
            fields={
                "name": target_name,
                "status": self.rng.choice(["active", "inactive", "pending"]),
                "submission_ref": submission_id,
                "department": self.rng.choice(["alpha", "beta", "gamma", "delta"]),
            }
        )
        records[person_id] = person_record

        chain.append(ChainStep(
            tool_name="query_records",
            params={"query_type": "person", "search_term": target_name},
            result={"record_id": person_id, "name": target_name, "submission_ref": submission_id},
            extracts={"record_id": person_id}
        ))
        collected_ids.append(person_id)

        # Remaining steps: build reference chain
        # Number of chain records needed (excluding person, which was step 1)
        num_chain_records = num_steps - 2  # -1 for query_records, -1 for final action
        ref_types = ["submission", "item", "evidence", "artifact", "document", "trace"]

        if num_chain_records <= 0:
            # 2-step chain: query person -> final action (no intermediate records)
            # Don't reference a submission that doesn't exist
            person_record.fields["submission_ref"] = None
        else:
            # Build the chain of records
            current_ref_id = submission_id

            for chain_idx in range(num_chain_records):
                ref_type = ref_types[chain_idx % len(ref_types)]
                is_last = (chain_idx == num_chain_records - 1)
                next_ref_id = None if is_last else generate_id(self.rng, ref_types[(chain_idx + 1) % len(ref_types)][:3].upper())

                # Create record
                record = Record(
                    record_id=current_ref_id,
                    record_type=ref_type,
                    fields={
                        "status": self.rng.choice(["verified", "pending", "flagged"]),
                        "category": self.rng.choice(["A", "B", "C", "D"]),
                        "ref_link": next_ref_id,
                        "value": self.rng.randint(100, 999),
                    }
                )
                records[current_ref_id] = record

                # Choose tool based on record type
                if ref_type == "submission":
                    tool_name = "get_record"
                    params = {"record_id": current_ref_id}
                else:
                    tool_name = "examine_item"
                    params = {"item_id": current_ref_id}

                result = {"record_id": current_ref_id, **record.fields}
                extracts = {"record_id": current_ref_id}
                if next_ref_id:
                    extracts["ref_link"] = next_ref_id

                chain.append(ChainStep(
                    tool_name=tool_name,
                    params=params,
                    result=result,
                    extracts=extracts
                ))
                collected_ids.append(current_ref_id)

                if next_ref_id:
                    current_ref_id = next_ref_id

        # Add distractors
        num_distractors = self.rng.randint(num_steps, num_steps * 2)
        for _ in range(num_distractors):
            dist_id = generate_id(self.rng, self.rng.choice(["DIS", "IRR", "OTH"]))
            dist_type = self.rng.choice(ref_types)
            records[dist_id] = Record(
                record_id=dist_id,
                record_type=dist_type,
                fields={
                    "status": self.rng.choice(["verified", "pending", "flagged"]),
                    "category": self.rng.choice(["A", "B", "C", "D"]),
                    "ref_link": generate_id(self.rng, "FAKE"),  # Points nowhere useful
                    "value": self.rng.randint(100, 999),
                }
            )

        # Determine final action
        final_action = self.rng.choices(ACTION_TYPES, weights=ACTION_WEIGHTS, k=1)[0]

        if final_action in ["file_report", "issue_warrant", "transfer_item", "escalate"]:
            final_params = {"case_id": case_id, "evidence_ids": collected_ids}
        elif final_action == "close_case":
            final_params = {"case_id": case_id, "resolution_ids": collected_ids}
        elif final_action == "archive":
            final_params = {"case_id": case_id, "chain_ids": collected_ids}
        elif final_action == "open_case":
            final_params = {"reference_ids": collected_ids}
        elif final_action == "summarize":
            final_params = {"case_id": case_id, "finding_ids": collected_ids}
        else:
            final_params = {"case_id": case_id, "evidence_ids": collected_ids}

        ground_truth = GroundTruth(
            chain=chain,
            final_action=final_action,
            final_params=final_params,
            case_id=case_id,
        )

        return records, ground_truth


# =============================================================================
# Tool Execution
# =============================================================================

def execute_tool(state: GameState, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool call and return result."""

    if tool_name == "query_records":
        query_type = params.get("query_type", "").lower()
        search_term = params.get("search_term", "").lower()

        # Flexible matching: accept various query types for person lookup
        person_query_types = {"person", "user", "name", "analyst", "submitter", "filer"}

        for record in state.records.values():
            # Match person records with flexible query types
            if record.record_type == "person":
                if query_type in person_query_types or query_type == record.record_type:
                    if record.fields.get("name", "").lower() == search_term:
                        return {
                            "found": True,
                            "record_id": record.record_id,
                            "type": "person",
                            "name": record.fields.get("name"),
                            "submission_ref": record.fields.get("submission_ref"),
                            "department": record.fields.get("department"),
                        }
            # Also allow direct record type matching
            elif record.record_type == query_type:
                name_field = record.fields.get("name", "").lower()
                if name_field == search_term:
                    return {
                        "found": True,
                        "record_id": record.record_id,
                        "type": record.record_type,
                        **{k: v for k, v in record.fields.items() if k != "name"},
                    }

        return {"found": False, "error": f"No record found matching query_type='{query_type}' search_term='{search_term}'. Try query_type='person' to find people by name."}

    elif tool_name == "get_record":
        record_id = params.get("record_id", "")
        if record_id in state.records:
            record = state.records[record_id]
            return {"record_id": record.record_id, "type": record.record_type, **record.fields}
        return {"error": f"Record '{record_id}' not found"}

    elif tool_name == "examine_item":
        item_id = params.get("item_id", "")
        if item_id in state.records:
            record = state.records[item_id]
            return {"item_id": record.record_id, "type": record.record_type, **record.fields}
        return {"error": f"Item '{item_id}' not found"}

    elif tool_name in ["file_report", "close_case", "archive", "issue_warrant",
                       "transfer_item", "open_case", "summarize", "escalate"]:
        # Final action - mark completion
        state.final_action_taken = True
        state.final_action_name = tool_name
        state.final_action_params = params
        state.done = True
        return {"success": True, "action": tool_name}

    return {"error": f"Unknown tool: {tool_name}"}


# =============================================================================
# Action Parsing
# =============================================================================

def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse a tool call from model output."""

    # Try to find JSON object with name and arguments/parameters
    patterns = [
        r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"(?:arguments|parameters|params)"\s*:\s*(\{[^}]+\})\s*\}',
        r'\{\s*"(?:arguments|parameters|params)"\s*:\s*(\{[^}]+\})\s*,\s*"name"\s*:\s*"([^"]+)"\s*\}',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                if "name" in pattern[:30]:  # name comes first
                    name = match.group(1)
                    args = json.loads(match.group(2))
                else:
                    args = json.loads(match.group(1))
                    name = match.group(2)
                return {"name": name, "arguments": args}
            except json.JSONDecodeError:
                continue

    # Try simpler JSON extraction
    json_match = re.search(r'\{[^{}]*"name"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group())
            if "name" in obj:
                return {"name": obj["name"], "arguments": obj.get("arguments", obj.get("parameters", obj.get("params", {})))}
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object
    try:
        # Find the last complete JSON object
        depth = 0
        start = -1
        for i, c in enumerate(text):
            if c == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        obj = json.loads(text[start:i+1])
                        if "name" in obj:
                            return {"name": obj["name"], "arguments": obj.get("arguments", obj.get("parameters", {}))}
                    except:
                        pass
                    start = -1
    except:
        pass

    return None


def extract_action(text: str, legal_actions: List[str]) -> Optional[str]:
    """Extract action from model output for game registry compatibility."""
    tool_call = parse_tool_call(text)
    if tool_call:
        return json.dumps(tool_call)
    return None


# =============================================================================
# Main Game Class
# =============================================================================

class MultiStepSequenceGame:
    """
    Multi-Step Sequence Game.

    Tests pure multi-step planning: execute correct sequence of dependent tool calls.
    Distribution matches exact tau-bench multi-step failure distribution.
    """

    def __init__(self, max_steps: int = 30):
        self.max_steps = max_steps
        self._rng = random.Random()
        self._state: Optional[GameState] = None

        # GameEnv protocol
        self.done = False
        self.current_player = 0
        self.rewards: Dict[int, float] = {0: 0.0}
        self.invalid_player: Optional[int] = None

    def reset(self, seed: int):
        """Reset game with new seed."""
        self._rng = random.Random(seed)

        # Sample step count from exact tau-bench distribution
        num_steps = self._rng.choices(STEP_COUNTS, weights=STEP_WEIGHTS, k=1)[0]

        # Generate case
        case_id = f"CASE-{self._rng.randint(1000, 9999)}"
        target_name = generate_name(self._rng)

        # Generate chain
        generator = ChainGenerator(self._rng)
        records, ground_truth = generator.generate(num_steps, case_id, target_name)

        self._state = GameState(
            case_id=case_id,
            target_name=target_name,
            records=records,
            ground_truth=ground_truth,
            tool_calls=[],
            collected_ids=[],
            current_step=0,
            final_action_taken=False,
            final_action_name=None,
            final_action_params=None,
            done=False,
        )

        self.done = False
        self.current_player = 0
        self.rewards = {0: 0.0}
        self.invalid_player = None

    def _generate_request(self) -> str:
        """Generate the natural language request."""
        state = self._state
        gt = state.ground_truth
        num_steps = len(gt.chain) + 1  # chain + final action

        # Action verb mappings
        action_verbs = {
            "file_report": "file a report",
            "close_case": "close the case",
            "archive": "archive the case",
            "issue_warrant": "issue a warrant",
            "transfer_item": "transfer the items",
            "open_case": "open a new case",
            "summarize": "summarize the findings",
            "escalate": "escalate the case",
        }
        action_verbs_cap = {
            "file_report": "File a report",
            "close_case": "Close the case",
            "archive": "Archive the case",
            "issue_warrant": "Issue a warrant",
            "transfer_item": "Transfer the items",
            "open_case": "Open a new case",
            "summarize": "Summarize the findings",
            "escalate": "Escalate the case",
        }
        action_verb = action_verbs.get(gt.final_action, "file a report")
        action_verb_cap = action_verbs_cap.get(gt.final_action, "File a report")

        # Get template for this step count
        if num_steps in REQUEST_TEMPLATES:
            template = self._rng.choice(REQUEST_TEMPLATES[num_steps])
        elif num_steps <= 7:
            template = self._rng.choice(REQUEST_TEMPLATES[max(REQUEST_TEMPLATES.keys())])
        else:
            template = LONG_CHAIN_TEMPLATE

        return template.format(
            case_id=state.case_id,
            name=state.target_name,
            chain_len=num_steps - 2,  # Number of chain hops (excluding query)
            action=action_verb,
            action_cap=action_verb_cap,
        )

    def observe(self, player_id: int) -> str:
        """Get observation for the agent."""
        state = self._state

        lines = []
        lines.append("You are a case analyst with access to a records database.")
        lines.append("")
        lines.append("Available tools (use JSON format with \"name\" and \"arguments\"):")
        lines.append("- query_records: Search for a person by name. Args: {\"query_type\": \"person\", \"search_term\": \"<name>\"}")
        lines.append("- get_record: Get record details by ID. Args: {\"record_id\": \"<id>\"}")
        lines.append("- examine_item: Examine an item by ID. Args: {\"item_id\": \"<id>\"}")
        lines.append("- file_report: File report. Args: {\"case_id\": \"...\", \"evidence_ids\": [...]}")
        lines.append("- close_case: Close case. Args: {\"case_id\": \"...\", \"resolution_ids\": [...]}")
        lines.append("- archive: Archive case. Args: {\"case_id\": \"...\", \"chain_ids\": [...]}")
        lines.append("- issue_warrant: Issue warrant. Args: {\"case_id\": \"...\", \"evidence_ids\": [...]}")
        lines.append("- summarize: Summarize findings. Args: {\"case_id\": \"...\", \"finding_ids\": [...]}")
        lines.append("- escalate: Escalate case. Args: {\"case_id\": \"...\", \"evidence_ids\": [...]}")
        lines.append("")
        lines.append("REQUEST:")
        lines.append(self._generate_request())
        lines.append("")

        # Show conversation history - full results so model can extract IDs
        if state.tool_calls:
            lines.append("PREVIOUS ACTIONS:")
            for tc in state.tool_calls[-5:]:  # Last 5 actions
                result_json = json.dumps(tc['result'])
                lines.append(f"  Tool: {tc['tool']} -> {result_json}")
            lines.append("")

        lines.append("Respond with a single tool call in JSON format.")

        return "\n".join(lines)

    def legal_actions(self) -> List[str]:
        """Return valid action formats."""
        if self.done:
            return []
        return ['{"name": "...", "arguments": {...}}']

    def step(self, action: Optional[str]):
        """Process agent action."""
        if self.done:
            return

        state = self._state

        # Parse action
        if action is None:
            self._finalize_invalid("No action provided")
            return

        tool_call = None
        try:
            tool_call = json.loads(action)
        except json.JSONDecodeError:
            tool_call = parse_tool_call(action)

        if tool_call is None:
            self._finalize_invalid("Could not parse tool call")
            return

        tool_name = tool_call.get("name", "")
        params = tool_call.get("arguments", tool_call.get("parameters", {}))

        # Execute tool
        result = execute_tool(state, tool_name, params)

        # Track
        state.tool_calls.append({
            "tool": tool_name,
            "params": params,
            "result": result,
        })
        state.current_step += 1

        # Check if done (final action taken or max steps)
        if state.done:
            self._finalize()
        elif state.current_step >= self.max_steps:
            self._finalize_timeout()

    def _finalize_invalid(self, reason: str):
        """Handle invalid action."""
        self.done = True
        self._state.done = True
        self.rewards = {0: -1.0}
        self.invalid_player = 0

    def _finalize_timeout(self):
        """Handle max steps reached."""
        self.done = True
        self._state.done = True
        self.rewards = {0: -1.0}  # Failed to complete

    def _finalize(self):
        """Compute final reward based on correctness."""
        state = self._state
        gt = state.ground_truth

        self.done = True

        # Check if final action is correct
        if not state.final_action_taken:
            self.rewards = {0: -1.0}
            return

        # Check action type
        if state.final_action_name != gt.final_action:
            self.rewards = {0: -1.0}
            return

        # Check parameters - specifically the collected IDs
        expected_ids = set(gt.final_params.get("evidence_ids",
                          gt.final_params.get("resolution_ids",
                          gt.final_params.get("chain_ids",
                          gt.final_params.get("reference_ids",
                          gt.final_params.get("finding_ids", []))))))

        actual_params = state.final_action_params or {}
        actual_ids = set(actual_params.get("evidence_ids",
                        actual_params.get("resolution_ids",
                        actual_params.get("chain_ids",
                        actual_params.get("reference_ids",
                        actual_params.get("finding_ids", []))))))

        # Check case_id if required
        if "case_id" in gt.final_params:
            if actual_params.get("case_id") != gt.final_params["case_id"]:
                self.rewards = {0: -0.5}  # Wrong case ID
                return

        # Check evidence IDs
        if expected_ids != actual_ids:
            # Partial credit if some IDs correct
            if expected_ids & actual_ids:  # Some overlap
                self.rewards = {0: -0.5}
            else:
                self.rewards = {0: -1.0}
            return

        # Correct! Check efficiency
        optimal_steps = len(gt.chain) + 1  # chain + final action
        actual_steps = len(state.tool_calls)

        if actual_steps <= optimal_steps:
            self.rewards = {0: 1.0}
        elif actual_steps <= optimal_steps + 2:
            self.rewards = {0: 0.8}
        elif actual_steps <= optimal_steps * 1.5:
            self.rewards = {0: 0.5}
        else:
            self.rewards = {0: 0.3}

    def get_summary(self) -> Dict[str, Any]:
        """Get game summary for logging."""
        state = self._state
        gt = state.ground_truth

        expected_ids = list(gt.final_params.get("evidence_ids",
                           gt.final_params.get("resolution_ids",
                           gt.final_params.get("chain_ids",
                           gt.final_params.get("reference_ids",
                           gt.final_params.get("finding_ids", []))))))

        actual_params = state.final_action_params or {}
        actual_ids = list(actual_params.get("evidence_ids",
                         actual_params.get("resolution_ids",
                         actual_params.get("chain_ids",
                         actual_params.get("reference_ids",
                         actual_params.get("finding_ids", []))))))

        return {
            "case_id": state.case_id,
            "target_name": state.target_name,
            "required_steps": len(gt.chain) + 1,
            "actual_steps": len(state.tool_calls),
            "expected_action": gt.final_action,
            "actual_action": state.final_action_name,
            "expected_ids": expected_ids,
            "actual_ids": actual_ids,
            "ids_correct": set(expected_ids) == set(actual_ids),
            "reward": self.rewards.get(0, 0.0),
            "tool_sequence": [tc["tool"] for tc in state.tool_calls],
        }


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT_MULTISTEP = """You are a case analyst with access to a records database.

Use tools to look up records, examine items, and complete case actions. Each tool returns information that may be needed for subsequent calls.

Respond with a single tool call in JSON format: {"name": "tool_name", "arguments": {...}}
"""


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Multi-Step Sequence Game")
    print("=" * 60)

    game = MultiStepSequenceGame()

    # Test distribution
    print("\nTesting step count distribution (1000 samples):")
    step_counts = []
    for seed in range(1000):
        game.reset(seed)
        gt = game._state.ground_truth
        step_counts.append(len(gt.chain) + 1)

    from collections import Counter
    counts = Counter(step_counts)
    for steps in sorted(counts.keys()):
        pct = counts[steps] / 10
        print(f"  {steps} steps: {counts[steps]} ({pct:.1f}%)")

    # Test a few games
    print("\n" + "=" * 60)
    print("Sample games:")

    for seed in [0, 1, 2, 42, 100]:
        print(f"\n--- Seed {seed} ---")
        game.reset(seed)
        state = game._state
        gt = state.ground_truth

        print(f"Case: {state.case_id}")
        print(f"Target: {state.target_name}")
        print(f"Required steps: {len(gt.chain) + 1}")
        print(f"Final action: {gt.final_action}")
        print(f"Chain: {[step.tool_name for step in gt.chain]} -> {gt.final_action}")
        print(f"\nObservation:\n{game.observe(0)[:500]}...")
