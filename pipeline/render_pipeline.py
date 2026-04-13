#!/usr/bin/env python3
"""Render TRACE pipeline templates from a YAML configuration file.

Usage:
    python pipeline/render_pipeline.py pipeline/trace_config.yaml

Reads the YAML config and produces two rendered markdown files:
    pipeline/<file_prefix>_capability_selection.md
    pipeline/<file_prefix>_environment_generation.md
"""

import argparse
import re
from pathlib import Path

import yaml


PIPELINE_DIR = Path(__file__).resolve().parent
CAPABILITY_SELECTION_TEMPLATE = PIPELINE_DIR / "trace_capability_selection.md"
ENVIRONMENT_GENERATION_TEMPLATE = PIPELINE_DIR / "trace_environment_generation.md"


def _fmt_float(val) -> str:
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def _norm_dir(path: str) -> str:
    return path.rstrip("/") + "/"


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = _norm_dir(cfg.get("output_dir", "pipeline/"))
    cs = cfg.get("capability_selection", {})
    eg = cfg.get("environment_generation", {})

    eval_results = cs.get("eval_results", [])
    if isinstance(eval_results, str):
        eval_results = [eval_results]

    eval_labels = cs.get("eval_labels", [])

    caps_file = eg.get("capabilities_file")
    if not caps_file:
        caps_file = f"{output_dir}selected_capabilities.json"

    gpu_device = eg.get("gpu_device")
    rho = cs.get("rho", 0.10)
    delta = cs.get("delta", 0.20)
    run_name = cfg.get("run_name", "run")
    file_prefix = cfg.get("file_prefix", run_name)

    return {
        "run_name": run_name,
        "file_prefix": file_prefix,
        "model": cfg["model"],
        "output_dir": output_dir,
        "eval_results": eval_results,
        "eval_labels": eval_labels,
        "n_runs": cs.get("n_runs", 10),
        "n_candidates": cs.get("n_candidates", 10),
        "rho": rho,
        "rho_str": _fmt_float(rho),
        "delta": delta,
        "delta_str": _fmt_float(delta),
        "k_consistency": cs.get("k_consistency", 8),
        "capabilities_file": caps_file,
        "gpu_device": gpu_device,
        "port": eg.get("port", 5050),
        "group_size": eg.get("group_size", 16),
        "num_seeds": eg.get("num_seeds", 100),
        "hint_ratio": str(eg.get("hint_ratio", "0.25-0.5")),
    }


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

    config_section_pattern = re.compile(
        r"Before starting, fill in these values\..*?"
        r"- Any JSON with a list of episodes, each containing a reward and conversation/action history",
        re.DOTALL,
    )
    text = config_section_pattern.sub(config_summary, text)

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


def render_environment_generation(template_text: str, cfg: dict) -> str:
    run_name = cfg["run_name"]
    group_size = cfg["group_size"]
    hint_ratio_str = cfg["hint_ratio"]

    if cfg["gpu_device"] is not None:
        gpu_display = f"`{cfg['gpu_device']}`"
        gpu_comment = f"# Using GPU {cfg['gpu_device']} from config"
        cuda_line = f"export CUDA_VISIBLE_DEVICES={cfg['gpu_device']}"
    else:
        gpu_display = "Auto-detect (the agent will run `nvidia-smi` and pick a free GPU)"
        gpu_comment = "# Check which GPUs are available and pick one that's free"
        cuda_line = "export CUDA_VISIBLE_DEVICES=<free GPU from nvidia-smi>"

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

    config_section_pattern = re.compile(
        r"Before starting, fill in these values\..*?"
        r"\*Default:\*\s*`pipeline/`\n",
        re.DOTALL,
    )
    text = config_section_pattern.sub(config_summary + "\n", text)

    text = text.replace(
        "Copy the prompt below and fill in the `{PLACEHOLDERS}` from `{CAPABILITIES_FILE}`.",
        f"Copy the prompt below and fill in the capability-specific values from\n"
        f"`{cfg['capabilities_file']}`.",
    )

    text = text.replace(
        "# Check which GPUs are available (if {GPU_DEVICE} was not specified, pick a free one)",
        gpu_comment,
    )
    text = text.replace(
        "export CUDA_VISIBLE_DEVICES={GPU_DEVICE}   # or whichever GPU is free from nvidia-smi",
        cuda_line,
    )
    if cfg["gpu_device"] is None:
        text = text.replace(
            "# Launch vLLM server\n",
            "# Launch vLLM server (replace CUDA_VISIBLE_DEVICES with the free GPU you found)\n",
        )

    replacements = {
        "{MODEL}": cfg["model"],
        "{CAPABILITIES_FILE}": cfg["capabilities_file"],
        "{PORT}": str(cfg["port"]),
        "{GROUP_SIZE}": str(group_size),
        "{NUM_SEEDS}": str(cfg["num_seeds"]),
        "{HINT_RATIO}": hint_ratio_str,
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
        "-o", "--output-dir",
        help="Output directory for rendered files (default: same as pipeline/)",
        default=None,
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.output_dir) if args.output_dir else PIPELINE_DIR

    cs_template = CAPABILITY_SELECTION_TEMPLATE.read_text()
    eg_template = ENVIRONMENT_GENERATION_TEMPLATE.read_text()

    cs_rendered = render_capability_selection(cs_template, cfg)
    eg_rendered = render_environment_generation(eg_template, cfg)

    file_prefix = cfg["file_prefix"]
    cs_out = out_dir / f"{file_prefix}_capability_selection.md"
    eg_out = out_dir / f"{file_prefix}_environment_generation.md"

    cs_out.write_text(cs_rendered)
    eg_out.write_text(eg_rendered)

    print(f"Rendered capability selection: {cs_out}")
    print(f"Rendered environment generation: {eg_out}")
    print(f"\nHand these files to your coding agent to execute each pipeline step.")


if __name__ == "__main__":
    main()
