#!/usr/bin/env python3
"""Render TRACE pipeline templates from a YAML configuration file.

Usage:
    python render_pipeline.py configs/capability_selection.yaml --stage capability
    python render_pipeline.py configs/environment_generation.yaml --stage environment

Reads the YAML config and produces a rendered markdown file in prompts/.
"""

import argparse
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
PROMPTS_DIR = REPO_ROOT / "prompts"
CAPABILITY_SELECTION_TEMPLATE = PIPELINE_DIR / "trace_capability_selection.md"
ENVIRONMENT_GENERATION_TEMPLATE = PIPELINE_DIR / "trace_environment_generation.md"


def _fmt_float(val) -> str:
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def _norm_dir(path: str) -> str:
    return path.rstrip("/") + "/"


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------

def load_capability_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = _norm_dir(cfg.get("output_dir", "pipeline/"))

    eval_results = cfg.get("eval_results", [])
    if isinstance(eval_results, str):
        eval_results = [eval_results]

    eval_labels = cfg.get("eval_labels", [])
    rho = cfg.get("rho", 0.10)
    delta = cfg.get("delta", 0.20)
    run_name = cfg.get("run_name", "run")
    file_prefix = cfg.get("file_prefix", run_name)

    return {
        "run_name": run_name,
        "file_prefix": file_prefix,
        "model": cfg["model"],
        "output_dir": output_dir,
        "eval_results": eval_results,
        "eval_labels": eval_labels,
        "n_runs": cfg.get("n_runs", 10),
        "n_candidates": cfg.get("n_candidates", 10),
        "rho": rho,
        "rho_str": _fmt_float(rho),
        "delta": delta,
        "delta_str": _fmt_float(delta),
        "k_consistency": cfg.get("k_consistency", 8),
    }


def load_environment_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = _norm_dir(cfg.get("output_dir", "pipeline/"))

    caps_file = cfg.get("capabilities_file")
    if not caps_file:
        caps_file = f"{output_dir}selected_capabilities.json"

    run_name = cfg.get("run_name", "run")
    file_prefix = cfg.get("file_prefix", run_name)

    return {
        "run_name": run_name,
        "file_prefix": file_prefix,
        "model": cfg["model"],
        "output_dir": output_dir,
        "capabilities_file": caps_file,
        "gpu_device": cfg.get("gpu_device"),
        "port": cfg.get("port", 5050),
        "group_size": cfg.get("group_size", 16),
        "num_seeds": cfg.get("num_seeds", 100),
        "hint_ratio": str(cfg.get("hint_ratio", "0.25-0.5")),
        "temperature": cfg.get("temperature", 0.7),
        "reward_threshold": cfg.get("reward_threshold", 0.0),
        "vllm": cfg.get("vllm", {}),
    }


# ---------------------------------------------------------------------------
# Capability selection helpers
# ---------------------------------------------------------------------------

def _format_eval_results_inline(
    eval_results: list[str], eval_labels: list[str]
) -> str:
    if len(eval_results) == 1:
        return eval_results[0]
    lines = []
    for i, path in enumerate(eval_results):
        label = eval_labels[i].capitalize() if i < len(eval_labels) else ""
        prefix = f"{label}: " if label else ""
        lines.append(f"- {prefix}{path}")
    return "\n" + "\n".join(lines)


def _format_eval_summary_lines(
    eval_results: list[str], eval_labels: list[str]
) -> str:
    lines = []
    for i, path in enumerate(eval_results):
        label = f" ({eval_labels[i]})" if i < len(eval_labels) else ""
        lines.append(f"- **Eval results{label}:** `{path}`")
    return "\n".join(lines)


def _format_diagram_header(
    eval_results: list[str], eval_labels: list[str], model: str
) -> str:
    if len(eval_results) == 1:
        return f"{eval_results[0]} (pass/fail trajectories for {model})"
    if eval_labels:
        return f"Eval results ({' + '.join(eval_labels)} simulations for {model})"
    return f"Eval results ({len(eval_results)} files, pass/fail trajectories for {model})"


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render_capability_selection(template_text: str, cfg: dict) -> str:
    eval_prompt = _format_eval_results_inline(
        cfg["eval_results"], cfg["eval_labels"]
    )
    run_name = cfg["run_name"]
    n_runs = cfg["n_runs"]
    k = cfg["k_consistency"]
    rho = cfg["rho_str"]
    delta = cfg["delta_str"]

    eval_block = _format_eval_summary_lines(
        cfg["eval_results"], cfg["eval_labels"]
    )

    config_summary = (
        f"All placeholders have been filled in for this test run.\n"
        f"\n"
        f"- **Model:** `{cfg['model']}`\n"
        f"{eval_block}\n"
        f"- **Number of Phase 2 runs:** `{n_runs}`\n"
        f"- **Candidates (Phase 1):** `{cfg['n_candidates']}`\n"
        f"- **Coverage threshold (`\u03c1`):** `{rho}` (paper default)\n"
        f"- **Contrastive gap threshold (`\u03b4`):** `{delta}` (paper default)\n"
        f"- **Cross-run consistency (`K`):** `{k}` of `{n_runs}` runs\n"
        f"- **Output directory:** `{cfg['output_dir']}`"
    )

    text = template_text

    text = text.replace(
        "# TRACE Capability Selection Pipeline",
        f"# TRACE Capability Selection Pipeline ({run_name})",
        1,
    )

    text = text.replace("{CONFIG_SUMMARY}", config_summary)

    text = text.replace(
        "Copy the prompt below, fill in the `{PLACEHOLDERS}`, and send it to an LLM agent\n"
        "(Claude, Codex, etc.) along with the evaluation results file(s).",
        "Copy the prompt below and send it to an LLM agent (Claude, Codex, etc.) along with the\n"
        "evaluation results files.",
    )
    text = text.replace(
        "Copy the prompt below, fill in the `{PLACEHOLDERS}`, and pass it to each subagent.",
        "Copy the prompt below and pass it to each subagent.",
    )

    diagram_header = _format_diagram_header(
        cfg["eval_results"], cfg["eval_labels"], cfg["model"]
    )
    text = text.replace(
        "{EVAL_RESULTS} (pass/fail trajectories for {MODEL_NAME})",
        diagram_header,
    )

    text = text.replace(
        "\u2502  Cross-run consistency: \u2265 {K_CONS}/  \u2502  \u2265 {K_CONSISTENCY} of {N_RUNS} runs\n"
        "\u2502  {N_RUNS}                            \u2502",
        f"\u2502  Cross-run consistency: \u2265 {k}/{n_runs}       \u2502  \u2265 {k} of {n_runs} runs",
    )

    text = text.replace(
        "   selected_capabilities.json\n"
        "   (with Cov",
        f"   {cfg['output_dir']}selected_capabilities.json\n"
        f"   (with Cov",
    )

    replacements = {
        "{MODEL_NAME}": cfg["model"],
        "{EVAL_RESULTS}": eval_prompt,
        "{N_RUNS}": str(n_runs),
        "{N_CANDIDATES}": str(cfg["n_candidates"]),
        "{RHO}": rho,
        "{DELTA}": delta,
        "{K_CONSISTENCY}": str(k),
        "{OUTPUT_DIR}": cfg["output_dir"],
    }
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)

    text = text.replace(
        "passes in at least `K_CONSISTENCY` of the `N_RUNS` runs",
        f"passes in at least `{k}` of the `{n_runs}` runs",
    )
    text = text.replace(
        "`n_runs_passed \u2265 K_CONSISTENCY`",
        f"`n_runs_passed \u2265 {k}`",
    )

    text = text.replace(
        "Cov_r(c) \u2265 \u03c1   AND   \u0394_r(c) \u2265 \u03b4",
        f"Cov_r(c) \u2265 {rho}   AND   \u0394_r(c) \u2265 {delta}",
    )

    text = text.replace(cfg["output_dir"] + "/", cfg["output_dir"])
    text = text.replace("located at: \n", "located at:\n")

    file_prefix = cfg["file_prefix"]
    text = text.replace(
        "trace_environment_generation.md",
        f"{file_prefix}_environment_generation.md",
    )

    return text


def _build_vllm_launch_block(cfg: dict) -> str:
    """Build the full vLLM launch bash block from config."""
    vllm = cfg["vllm"]
    lines = []

    # GPU check
    if cfg["gpu_device"] is not None:
        lines.append(f"# Using GPU {cfg['gpu_device']} from config")
    else:
        lines.append("# Check which GPUs are available and pick a free one")
        lines.append("nvidia-smi")
        lines.append("")

    # Conda activation
    conda_env = vllm.get("conda_env")
    if conda_env:
        lines.append(f"conda activate {conda_env}")

    # Environment variables from config
    for key, value in vllm.get("env", {}).items():
        lines.append(f"export {key}={value}")

    # CUDA_VISIBLE_DEVICES (from gpu_device, not the vllm.env block)
    if cfg["gpu_device"] is not None:
        lines.append(f"export CUDA_VISIBLE_DEVICES={cfg['gpu_device']}")
    else:
        lines.append(
            "export CUDA_VISIBLE_DEVICES=<free GPU from nvidia-smi>"
        )

    # vllm serve command
    serve_args = str(vllm.get("serve_args", "")).strip()
    port = cfg["port"]
    model = cfg["model"]

    # Split raw flags into individual --flag tokens for line-wrapped output
    tokens = serve_args.split()
    # Build flag groups: each group starts with -- and includes its value(s)
    flags = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            group = token
            # Consume following non-flag tokens as values
            while i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                i += 1
                group += f" {tokens[i]}"
            flags.append(group)
        i += 1

    parts = [f"vllm serve {model} \\"]
    parts.append(f"  --port {port} \\")
    for flag in flags:
        parts.append(f"  {flag} \\")
    # Remove trailing backslash from the last line
    parts[-1] = parts[-1].rstrip(" \\")

    lines.append("\n".join(parts))
    return "\n".join(lines)


def render_environment_generation(template_text: str, cfg: dict) -> str:
    run_name = cfg["run_name"]
    group_size = cfg["group_size"]
    hint_ratio_str = cfg["hint_ratio"]

    if cfg["gpu_device"] is not None:
        gpu_display = f"`{cfg['gpu_device']}`"
    else:
        gpu_display = "Auto-detect (the agent will run `nvidia-smi` and pick a free GPU)"

    hint_annotation = ""
    parts = hint_ratio_str.split("-")
    try:
        lo = float(parts[0])
        hi = float(parts[1]) if len(parts) > 1 else lo
        lo_count = int(lo * group_size)
        hi_count = int(hi * group_size)
        if lo_count == hi_count:
            hint_annotation = f" ({lo_count} out of {group_size} rollouts get hints)"
        else:
            hint_annotation = f" ({lo_count}-{hi_count} out of {group_size} rollouts get hints)"
    except (ValueError, IndexError):
        pass

    config_summary = (
        f"All placeholders have been filled in for this test run.\n"
        f"\n"
        f"- **Model:** `{cfg['model']}`\n"
        f"- **Capabilities file:** `{cfg['capabilities_file']}` (output from the capability selection step)\n"
        f"- **GPU device:** {gpu_display}\n"
        f"- **vLLM port:** `{cfg['port']}`\n"
        f"- **Group size:** `{group_size}` rollouts per seed\n"
        f"- **Number of seeds:** `{cfg['num_seeds']}`\n"
        f"- **Hint ratio:** `{hint_ratio_str}`{hint_annotation}\n"
        f"- **Output directory:** `{cfg['output_dir']}`"
    )

    text = template_text

    text = text.replace(
        "# TRACE Environment Generation Pipeline",
        f"# TRACE Environment Generation Pipeline ({run_name})",
        1,
    )

    text = text.replace("{CONFIG_SUMMARY}", config_summary)

    text = text.replace(
        "Copy the prompt below and fill in the `{PLACEHOLDERS}` from `{CAPABILITIES_FILE}`.",
        f"Copy the prompt below and fill in the capability-specific values from\n"
        f"`{cfg['capabilities_file']}`.",
    )

    vllm_block = _build_vllm_launch_block(cfg)
    text = text.replace("{VLLM_LAUNCH_BLOCK}", vllm_block)

    replacements = {
        "{MODEL}": cfg["model"],
        "{CAPABILITIES_FILE}": cfg["capabilities_file"],
        "{PORT}": str(cfg["port"]),
        "{GROUP_SIZE}": str(group_size),
        "{NUM_SEEDS}": str(cfg["num_seeds"]),
        "{HINT_RATIO}": hint_ratio_str,
        "{TEMPERATURE}": str(cfg["temperature"]),
        "{REWARD_THRESHOLD}": str(cfg["reward_threshold"]),
        "{OUTPUT_DIR}": cfg["output_dir"],
    }
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)

    text = text.replace(cfg["output_dir"] + "/", cfg["output_dir"])

    file_prefix = cfg["file_prefix"]
    text = text.replace(
        "trace_capability_selection.md",
        f"{file_prefix}_capability_selection.md",
    )

    return text


def main():
    parser = argparse.ArgumentParser(
        description="Render TRACE pipeline templates from a YAML config."
    )
    parser.add_argument("config", help="Path to the YAML configuration file")
    parser.add_argument(
        "--stage",
        choices=["capability", "environment"],
        required=True,
        help="Which pipeline stage to render",
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Output directory for rendered files (default: prompts/)",
        default=None,
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else PROMPTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "capability":
        cfg = load_capability_config(args.config)
        template_text = CAPABILITY_SELECTION_TEMPLATE.read_text()
        rendered = render_capability_selection(template_text, cfg)
        out_file = out_dir / f"{cfg['file_prefix']}_capability_selection.md"
    else:
        cfg = load_environment_config(args.config)
        template_text = ENVIRONMENT_GENERATION_TEMPLATE.read_text()
        rendered = render_environment_generation(template_text, cfg)
        out_file = out_dir / f"{cfg['file_prefix']}_environment_generation.md"

    out_file.write_text(rendered)
    print(f"Rendered: {out_file}")


if __name__ == "__main__":
    main()
