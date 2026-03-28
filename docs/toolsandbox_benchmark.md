# ToolSandbox Benchmark Integration

## Overview

**ToolSandbox** (Lu et al., 2024) is a stateful, conversational tool-use benchmark from Apple Research that evaluates LLM agents on multi-turn interactions with a simulated phone assistant environment. Unlike tau2-bench which focuses on customer service, ToolSandbox tests general-purpose tool calling across contacts, messaging, reminders, settings, and search tools.

- **Paper**: "ToolSandbox: A Stateful, Conversational, Interactive Evaluation Framework for LLM Tool Use Capabilities"
- **GitHub**: https://github.com/apple/ToolSandbox
- **License**: Apple sample code license

## Benchmark Statistics

| Metric | Value |
|--------|-------|
| Base scenarios | 129 |
| With augmentations (distraction tools, scrambled names, etc.) | 1,032 |
| Tool categories | 5 (contacts, messaging, settings, reminders, search) |
| Total unique tools | ~38 |
| Scenario categories | Single tool call, Multiple tool calls, Multiple user turns, Insufficient information |
| Evaluation | Milestone-based DAG with geometric mean similarity |

### Scenario Breakdown (base, no augmentations)

| Category | Count |
|----------|-------|
| Single tool call | 19 |
| Multiple tool calls | 54 |
| Multiple user turns | 28 |
| Insufficient information | 28 |
| **Total** | **129** |

### Tool Augmentation Variants

Each of the 129 base scenarios is expanded into 8 variants:
- No distraction tools (original)
- 3 distraction tools added
- 10 distraction tools added
- All tools available
- Tool descriptions scrambled
- Argument types scrambled
- Argument descriptions scrambled
- Tool names scrambled

This produces 1,032 total scenarios (129 × 8).

### Available Tools (~38 total)

**Contact management**: add_contact, modify_contact, remove_contact, search_contacts
**Messaging**: send_message_with_phone_number, search_messages
**Settings**: set/get_wifi_status, set/get_cellular_service_status, set/get_location_service_status, set/get_low_battery_mode_status, get_current_location
**Reminders**: add_reminder, modify_reminder, remove_reminder, search_reminder
**Search/Utilities**: search_weather, search_stock, convert_currency, search_location, search_holiday, unit_conversion, timestamp utilities, calculate_lat_lon_distance
**System**: end_conversation, get_current_timestamp

## Overlap with tau2-bench

### Similarities

| Aspect | tau2-bench | ToolSandbox |
|--------|-----------|-------------|
| Multi-turn conversations | Yes | Yes |
| Stateful tool execution | Yes (database state) | Yes (execution context state) |
| User simulator | LLM-based | LLM-based |
| Tool calling format | OpenAI function calling | OpenAI function calling |
| Evaluation | DB state + communication | Milestone DAG + similarity |
| State dependencies | Implicit (policy rules) | Explicit (settings affecting tool availability) |

### Differences

| Aspect | tau2-bench | ToolSandbox |
|--------|-----------|-------------|
| Domain | Customer service (airline/retail) | Phone assistant (contacts/messaging/settings) |
| Tools | 14-15 per domain | ~38 across all categories |
| Tasks | 164 (50 airline + 114 retail) | 129 base / 1,032 with augmentations |
| Policy reasoning | Heavy (complex eligibility rules) | Light (settings toggle logic) |
| Adversarial users | Yes (emotional pressure, lies) | No (cooperative user simulator) |
| Multi-step complexity | High (2-5 dependent operations) | Medium (1-3 tool calls typically) |
| Tool augmentation testing | No | Yes (distraction, scrambling) |

### Skill Overlap

| Our Training Skill | Relevant in tau2-bench? | Relevant in ToolSandbox? |
|--------------------|------------------------|-------------------------|
| Structured data reasoning | Yes (flight/product selection) | Yes (contact/reminder search + filtering) |
| Tool calling precision | Yes (wrong IDs/amounts) | Yes (wrong contact IDs, wrong settings) |
| Multi-step task completion | Yes (compound requests) | Yes (multiple tool calls category) |
| Precondition verification | Yes (policy eligibility) | Partially (state dependencies like WiFi off blocks location) |
| Adversarial policy compliance | Yes (user pressure) | No (cooperative users only) |

ToolSandbox provides complementary evaluation: it tests generalization of tool-calling skills to a different domain (phone assistant vs. customer service) with different tool sets, without the adversarial user component. Strong performance on both benchmarks would demonstrate that our skill-based training transfers across domains rather than overfitting to tau2-bench's specific tools and policies.

## Running with Qwen3-30B-A3B

### Challenge: User Simulator

ToolSandbox's user simulator only supports OpenAI GPT models (GPT-3.5, GPT-4, GPT-4o) out of the box. To use Qwen3-30B for both agent and user, we need to either:

1. **Option A (Recommended)**: Add a vLLM-compatible user simulator that points to the same OpenAI-compatible server
2. **Option B**: Use GPT-4o as the user (costs money, but is the official evaluation setup)
3. **Option C**: Modify the OpenAI user class to accept a custom base_url

### Option A: Adding vLLM Support

The agent side already works with vLLM via the `Hermes` agent type, which uses `OPENAI_BASE_URL` for connecting to an OpenAI-compatible server. For the user side, we need to modify `tool_sandbox/cli/utils.py` to add a vLLM user type.

**Step 1: Serve the model via vLLM**

```bash
# Terminal 1: vLLM server
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --port 8080 --max-model-len 32000 --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

Note: `--enable-auto-tool-choice --tool-call-parser hermes` enables vLLM's native tool calling support for Hermes-style models. Qwen3 models support this format.

**Step 2: Add Qwen/vLLM model types**

Add to `tool_sandbox/cli/utils.py`:

```python
# In RoleImplType enum:
Qwen3_vLLM = auto()

# In AGENT_TYPE_TO_FACTORY:
RoleImplType.Qwen3_vLLM: lambda: OpenAICompatibleAgent(
    model_name="Qwen/Qwen3-30B-A3B-Instruct-2507"
),

# In USER_TYPE_TO_FACTORY:
RoleImplType.Qwen3_vLLM: lambda: OpenAICompatibleUser(
    model_name="Qwen/Qwen3-30B-A3B-Instruct-2507"
),
```

This requires creating `OpenAICompatibleAgent` and `OpenAICompatibleUser` classes that use `OPENAI_BASE_URL` instead of hardcoded OpenAI endpoints.

**Step 3: Run evaluation**

```bash
# Run all 129 base scenarios (no augmentations)
OPENAI_BASE_URL=http://localhost:8080/v1 \
OPENAI_API_KEY=EMPTY \
tool_sandbox \
    --user Qwen3_vLLM \
    --agent Qwen3_vLLM \
    --scenario all
```

For augmented scenarios (1,032 total):
```bash
# This will take much longer
OPENAI_BASE_URL=http://localhost:8080/v1 \
OPENAI_API_KEY=EMPTY \
tool_sandbox \
    --user Qwen3_vLLM \
    --agent Qwen3_vLLM
```

### Option C: Quick Hack (Modify OpenAI User)

The fastest path: patch `openai_api_user.py` line 38 to not hardcode the OpenAI URL:

```python
# Change:
self.openai_client: OpenAI = OpenAI(base_url="https://api.openai.com/v1")
# To:
self.openai_client: OpenAI = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
)
```

Then similarly in `openai_api_agent.py`. This lets you use any existing GPT model type but pointed at vLLM.

## Estimated Runtime

| Configuration | Scenarios | Est. Time per Scenario | Total Est. Time |
|--------------|-----------|----------------------|----------------|
| Base scenarios only | 129 | 30-60 sec | ~1-2 hours |
| With 3-distraction augmentation | 258 | 30-60 sec | ~2-4 hours |
| All augmentations | 1,032 | 30-60 sec | ~8-17 hours |

Estimates assume single vLLM server with Qwen3-30B-A3B on 1 GPU. Multi-turn scenarios (multiple user turns, state dependency) take longer due to more conversation rounds.

**Recommendation**: Start with the 129 base scenarios for initial evaluation, then run augmented variants if needed for the paper.

## Evaluation Metrics

ToolSandbox uses milestone-based evaluation:

1. **Milestones**: Each scenario defines a directed acyclic graph (DAG) of required state changes (e.g., "contact added" → "message sent to new contact").
2. **Similarity**: Each milestone is scored using a combination of:
   - Exact match on critical fields
   - ROUGE-L similarity on text fields
   - Snapshot comparison of database state
3. **Aggregation**: Geometric mean across milestones within a scenario, arithmetic mean across scenarios within a category.

This is more granular than tau2-bench's binary pass/fail — ToolSandbox gives partial credit for partially correct solutions.

## Installation

```bash
cd /home/ubuntu/hangook/games/evals/benchmarks/ToolSandbox

# Install with pinned dependencies
pip install polars==0.20.31 strenum rapidfuzz ccy geopy holidays \
    phonenumbers pint rouge-score langchain==0.1.3 langchain-core \
    tree-sitter==0.22.3 tree-sitter-languages==1.10.2

# Install the package
pip install .
```

## Files

- **Repo**: `/home/ubuntu/hangook/games/evals/benchmarks/ToolSandbox/`
- **Scenarios**: `tool_sandbox/scenarios/` (4 category files)
- **Tools**: `tool_sandbox/tools/` (5 category files)
- **Agents**: `tool_sandbox/roles/` (Hermes, OpenAI, Anthropic, etc.)
- **CLI**: `tool_sandbox/cli/` (main entry point)




  OPENAI_BASE_URL=http://localhost:5051/v1 \
  OPENAI_API_KEY=EMPTY \
  VLLM_BASE_URL=http://localhost:5051/v1 \
  python -m tool_sandbox.cli \
      --agent VLLM \
      --user VLLM \
      --scenarios $(python3 -c "
import sys; sys.path.insert(0, '.')
from tool_sandbox.scenarios import named_scenarios
from tool_sandbox.common.tool_discovery import ToolBackend
s = named_scenarios(ToolBackend.DEFAULT)
base = [n for n in s if not any(x in n for x in ['distraction','scrambled','all_tools'])]
print(' '.join(base))
  ") \
      --parallel 1 \
      --output_dir data/qwen3-30b-base


  OPENAI_BASE_URL=http://localhost:5051/v1 \
  OPENAI_API_KEY=EMPTY \
  VLLM_BASE_URL=http://localhost:5051/v1 \
  python -m tool_sandbox.cli \
      --agent VLLM \
      --user VLLM \
      --scenarios $(python /tmp/get_scenarios.py) \
      --parallel 1 \
      --output_dir data/qwen3-30b-base


  OPENAI_BASE_URL=http://localhost:5051/v1 \
  OPENAI_API_KEY=EMPTY \
  VLLM_BASE_URL=http://localhost:5051/v1 \
  tool_sandbox \
      --agent VLLM \
      --user VLLM \
      --scenarios $(python /tmp/get_scenarios.py) \
      --parallel 1 \
      --output_dir data/qwen3-30b-base


  OPENAI_BASE_URL=http://localhost:5051/v1 \
  OPENAI_API_KEY=EMPTY \
  VLLM_BASE_URL=http://localhost:5051/v1 \
  tool_sandbox \
      --agent VLLM \
      --user VLLM \
      --scenarios $(python /tmp/get_scenarios.py) \
      --parallel 1 \
      --output_dir data/qwen3-30b-base

  python -c "
p = 'tool_sandbox/roles/gorilla_api_agent.py'
t = open(p).read()
t = t.replace(
    'from tree_sitter_languages import get_language, get_parser',
    'try:\n    from tree_sitter_languages import get_language, get_parser\nexcept ImportError:\n    get_language = get_parser = None'
)
open(p, 'w').write(t)
print('Patched')
"



  pip install polars==0.20.31 numpy==1.26.4 strenum rapidfuzz==3.9.3 ccy==1.3.1 \
      geopy==2.4.1 holidays==0.51 phonenumbers==8.13.39 pint==0.23 \
      rouge-score==0.1.2 langchain==0.1.3 langchain-core==0.1.23 \
      langchain-community==0.0.20 langsmith==0.0.87 \
      anthropic==0.26.1 openai==1.17.0 pydantic==2.7.4 \
      tree-sitter==0.22.3 tree-sitter-languages==1.10.2 \
      tenacity==8.4.1 scipy==1.13.1 networkx==3.2.1 \
      jsonschema==4.19.2 dill==0.3.8 decorator==5.1.1 \
      pyarrow==16.1.0 sentencepiece==0.2.0 tqdm


cat > /tmp/get_scenarios.py << 'EOF'                                                                                                                                                                                                                                                                                                     
import sys                                                                                                                                                                                                                                                                                                                               
sys.path.insert(0, '.')                                                                                                                                                                                                                                                                                                                  
from tool_sandbox.scenarios import named_scenarios                                                                                                                                                                                                                                                                                       
from tool_sandbox.common.tool_discovery import ToolBackend
s = named_scenarios(ToolBackend.DEFAULT)                                                                                                                                                                                                                                                                                                 
base = [n for n in s if not any(x in n for x in ['distraction','scrambled','all_tools'])]
print(' '.join(base))                   
EOF



PYTHONNOUSERSITE=1 \
  VLLM_AGENT_URL=http://localhost:9090/v1 \
  VLLM_AGENT_MODEL=tarsur909/precondition-v1-40 \
  RAPID_API_KEY=0cee0dd4e6msh4ecca885437f64ap1943fajsn69a861b57791 \
  VLLM_USER_URL=http://localhost:5050/v1 \
  OPENAI_API_KEY=EMPTY \
  tool_sandbox \
      --agent VLLM \
      --user VLLM \
      --scenarios $(python /tmp/get_scenarios.py) \
      --parallel 1 \
      --output_dir data/precondition-v1-40




  OPENAI_BASE_URL=http://localhost:5051/v1 \
  OPENAI_API_KEY=EMPTY \
  VLLM_BASE_URL=http://localhost:5051/v1 \
  tool_sandbox \
      --agent VLLM \
      --user VLLM \
      --scenarios $(python /tmp/get_scenarios.py) \
      --parallel 1 \
      --output_dir data/qwen3-30b-base



    OPENAI_BASE_URL=http://localhost:5051/v1 \
  OPENAI_API_KEY=EMPTY \
  VLLM_BASE_URL=http://localhost:5051/v1 \
  tool_sandbox \
      --agent VLLM \
      --user VLLM \
      --scenarios \
          search_phone_number_with_name \
          wifi_off \
          cellular_off \
          add_contact_with_name_and_phone_number \
          add_reminder_content_and_date_and_time \
          convert_currency \
          turn_on_wifi_low_battery_mode \
          search_reminder_with_creation_recency_yesterday \
          update_contact_with_id_and_phone_number \
      --parallel 1 \
      --output_dir data/qwen3-30b-fix-test




  VLLM_AGENT_URL=http://localhost:5050/v1 \
  VLLM_USER_URL=http://localhost:5051/v1 \
  OPENAI_API_KEY=EMPTY \
  RAPID_API_KEY=0cee0dd4e6msh4ecca885437f64ap1943fajsn69a861b57791 \
  tool_sandbox \
      --agent VLLM \
      --user VLLM \
      --scenarios convert_currency find_temperature find_days_till_holiday \
      --parallel 1 \
      --output_dir data/rapid-api-test 


  VLLM_AGENT_URL=http://localhost:9090/v1 \
  VLLM_AGENT_MODEL=tarsur909/precondition-v1-40 \
  RAPID_API_KEY=0cee0dd4e6msh4ecca885437f64ap1943fajsn69a861b57791 \
  VLLM_USER_URL=http://localhost:5050/v1 \
  OPENAI_API_KEY=EMPTY \
  /home/ubuntu/miniconda3/envs/toolsandbox/bin/python -s -c "
import sys, os
sys.path = [p for p in sys.path if '.local' not in p]
import transformers.utils.versions
transformers.utils.versions.require_version = lambda *a, **k: None
transformers.utils.versions.require_version_core = lambda *a, **k: None
from tool_sandbox.cli import main
main()
  " \
      --agent VLLM \
      --user VLLM \
      --scenarios $(/home/ubuntu/miniconda3/envs/toolsandbox/bin/python -s /tmp/get_scenarios.py 2>/dev/null) \
      --parallel 1 \
      --output_dir data/precondition-v1-40








  cat > /tmp/run_ts.py << 'PYEOF'
import sys, os


sys.path = [p for p in sys.path if '.local' not in p]

import importlib.metadata as _meta
_orig_version = _meta.version
def _patched_version(name):
    if name in ('huggingface-hub', 'huggingface_hub'):
        return '0.23.4'
    return _orig_version(name)
_meta.version = _patched_version

from tool_sandbox.cli import main
main()
PYEOF





VLLM_AGENT_URL=http://localhost:9090/v1 \
  VLLM_AGENT_MODEL=tarsur909/precondition-v1-40 \
  RAPID_API_KEY=0cee0dd4e6msh4ecca885437f64ap1943fajsn69a861b57791 \
  VLLM_USER_URL=http://localhost:5050/v1 \
  OPENAI_API_KEY=EMPTY \
  /home/ubuntu/miniconda3/envs/toolsandbox/bin/python -s /tmp/run_ts.py \
      --agent VLLM \
      --user VLLM \
      --scenarios $(/home/ubuntu/miniconda3/envs/toolsandbox/bin/python -s -c "
  import sys; sys.path = [p for p in sys.path if '.local' not in p]
  import importlib.metadata as m; o=m.version; m.version=lambda n: '0.23.4' if 'huggingface' in n else o(n)
  from tool_sandbox.scenarios import named_scenarios
  from tool_sandbox.common.tool_discovery import ToolBackend
  s = named_scenarios(ToolBackend.DEFAULT)
  base = [n for n in s if not any(x in n for x in ['distraction','scrambled','all_tools'])]
  print(' '.join(base))
  " 2>/dev/null) \
      --parallel 1 \
      --output_dir data/precondition-v1-40






/home/ubuntu/miniconda3/envs/toolsandbox/bin/pip install \
      polars==0.20.31 numpy==1.26.4 pint==0.23 dill safetensors \
      strenum rapidfuzz==3.9.3 ccy geopy holidays phonenumbers \
      rouge-score langchain-core==0.1.23 langchain==0.1.3 \
      langchain-community==0.0.20 langsmith==0.0.87 \
      anthropic==0.26.1 openai==1.30.0 httpx==0.27.0 \
      pydantic==2.7.4 transformers==4.41.2 huggingface-hub==0.23.4 \
      tokenizers==0.19.1 tenacity==8.4.1 scipy==1.13.1 \
      networkx==3.2.1 jsonschema==4.19.2 decorator pyarrow==16.1.0 \
      sentencepiece==0.2.0 tqdm typing-extensions==4.12.2



  /home/ubuntu/miniconda3/envs/toolsandbox/bin/python -c "
import importlib.metadata; print(importlib.metadata.version('huggingface-hub'))
import transformers; print('transformers:', transformers.__version__)
  "


  VLLM_AGENT_URL=http://localhost:9090/v1 \
  VLLM_AGENT_MODEL=tarsur385/tsb-TEC-v2-30it \
  RAPID_API_KEY=0cee0dd4e6msh4ecca885437f64ap1943fajsn69a861b57791 \
  VLLM_USER_URL=http://localhost:5051/v1 \
  OPENAI_API_KEY=EMPTY \
  tool_sandbox \
      --agent VLLM \
      --user VLLM \
      --scenarios $(python /tmp/get_scenarios.py) \
      --parallel 1 \
      --output_dir data/TEC-v2-30it3




VLLM_AGENT_URL=http://localhost:5051/v1 \
  VLLM_USER_URL=http://localhost:9001/v1 \
  RAPID_API_KEY=0cee0dd4e6msh4ecca885437f64ap1943fajsn69a861b57791 \
  OPENAI_API_KEY=EMPTY \
  tool_sandbox \
      --agent VLLM \
      --user VLLM \
      --scenarios $(python /tmp/get_scenarios.py) \
      --parallel 1 \
      --output_dir data/qwen3-30b-base-v3










mkdir -p /home/ubuntu/miniconda3/envs/toolsandbox/etc/conda/activate.d
cat > /home/ubuntu/miniconda3/envs/toolsandbox/etc/conda/activate.d/no_user_site.sh << 'EOF'
export PYTHONNOUSERSITE=1
EOF

mkdir -p /home/ubuntu/miniconda3/envs/toolsandbox/etc/conda/deactivate.d
cat > /home/ubuntu/miniconda3/envs/toolsandbox/etc/conda/deactivate.d/no_user_site.sh << 'EOF'
unset PYTHONNOUSERSITE
EOF