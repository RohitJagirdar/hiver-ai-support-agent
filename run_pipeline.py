"""
Top-Level Reproducibility Runner and CLI Interface for Hiver AI Support Agent.
Enables instant reproduction (< 1 minute), interactive single-query testing, and full comparative evaluation.
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any

# Ensure UTF-8 output on Windows (supports emoji in draft replies)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import IntentCategory, ActionDecision
from src.agent import HiverSupportAgent
from src.baselines import TrivialBaseline, SimpleZeroShotBaseline
from src.evaluator import EvaluationHarness


def print_banner():
    print("=" * 80)
    print("      HIVER SDE INTERN TAKE-HOME: AI SUPPORT AGENT EVALUATION PIPELINE      ")
    print("               Target Brand: @AppleSupport (Twitter Support)                ")
    print("=" * 80)


def format_table(rows: List[List[str]], headers: List[str]) -> str:
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    data_lines = [" | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row)) for row in rows]
    return f"\n{header_line}\n{sep_line}\n" + "\n".join(data_lines) + "\n"


def run_single_test(agent: HiverSupportAgent, query: str):
    print("\n" + "=" * 80)
    print(f"LIVE TEST INPUT: \"{query}\"")
    print("=" * 80)
    start_time = time.time()
    output = agent.process_query(query)
    elapsed_ms = (time.time() - start_time) * 1000

    action_color = "\033[92m" if output.action == ActionDecision.AUTO_HANDLE else "\033[93m"
    reset_color = "\033[0m"

    print(f"\n[AGENT DECISION]: {action_color}{output.action.value}{reset_color}")
    print(f"  * Predicted Intent:   {output.predicted_intent.value} (Confidence: {output.confidence_score:.2f})")
    print(f"  * Urgency Level:      {output.urgency_level.value}")
    print(f"  * Escalation Reason:  {output.escalation_reason}")
    print(f"  * Reference IDs Used: {output.retrieved_reference_ids}")
    print(f"  * Latency:            {elapsed_ms:.1f} ms")
    print("\n[DRAFTED RESOLUTION REPLY]:")
    print(f"\"{output.draft_reply}\"")
    print("=" * 80 + "\n")


def run_benchmark(mode: str, mock: bool = False):
    print_banner()
    golden_path = "data/golden_set.json"
    if not os.path.exists(golden_path):
        print(f"Error: Golden set not found at {golden_path}. Run scripts/curate_dataset.py first.")
        sys.exit(1)

    with open(golden_path, mode="r", encoding="utf-8") as f:
        full_golden_set = json.load(f)

    if mode == "quick":
        # First 10 unique queries, for a fast smoke test
        dataset = full_golden_set[:10]
        print(f"\n[MODE: QUICK] Running benchmark on {len(dataset)} representative golden samples...")
    else:
        dataset = full_golden_set
        print(f"\n[MODE: FULL] Running full benchmark across all {len(dataset)} golden evaluation samples...")

    # Initialize components
    print("Loading vector retrieval engine...")
    agent = HiverSupportAgent(mock_mode=mock)
    if agent.retriever is None and not agent.mock_mode:
        # A live run without grounding would be reported as the RAG system but isn't one.
        print(f"Error: retrieval is unavailable ({agent.retrieval_error}). Refusing to run a live "
              "benchmark without it. Fix the install, or pass --mock for a smoke test.")
        sys.exit(1)
    print(f"Run mode: {agent.describe_run_mode()}")
    trivial_baseline = TrivialBaseline()
    zero_shot_baseline = SimpleZeroShotBaseline(agent)

    harness = EvaluationHarness(agent)

    # 1. Evaluate Baseline 1
    print("\nEvaluating Baseline 1 (Trivial Rule/Keyword Classifier)...")
    b1_harness = EvaluationHarness(trivial_baseline)
    b1_metrics = b1_harness.evaluate_dataset(dataset, model_name="Baseline 1 (Trivial Keyword)")

    # 2. Evaluate Baseline 2
    print("Evaluating Baseline 2 (Zero-Shot LLM without RAG)...")
    b2_harness = EvaluationHarness(zero_shot_baseline)
    b2_metrics = b2_harness.evaluate_dataset(dataset, model_name="Baseline 2 (Zero-Shot LLM)")

    # 3. Evaluate Proposed System (Final Agent)
    print("Evaluating Proposed System (HiverSupportAgent: Guardrails + RAG + Policy Gate)...")
    agent_metrics = harness.evaluate_dataset(dataset, model_name="HiverSupportAgent (Proposed)")

    # Format Benchmark Summary Table
    headers = ["Model / Pipeline", "Intent Macro-F1", "Routing Acc.", "False Auto-Handle %", "False Escalation %"]
    rows = [
        [
            b1_metrics["model_name"],
            f"{b1_metrics['intent_macro_f1'] * 100:.1f}%",
            f"{b1_metrics['routing_accuracy'] * 100:.1f}%",
            f"{b1_metrics['false_auto_handle_rate'] * 100:.1f}% (High Risk)",
            f"{b1_metrics['false_escalation_rate'] * 100:.1f}%"
        ],
        [
            b2_metrics["model_name"],
            f"{b2_metrics['intent_macro_f1'] * 100:.1f}%",
            f"{b2_metrics['routing_accuracy'] * 100:.1f}%",
            f"{b2_metrics['false_auto_handle_rate'] * 100:.1f}%",
            f"{b2_metrics['false_escalation_rate'] * 100:.1f}%"
        ],
        [
            agent_metrics["model_name"],
            f"{agent_metrics['intent_macro_f1'] * 100:.1f}%",
            f"{agent_metrics['routing_accuracy'] * 100:.1f}%",
            f"{agent_metrics['false_auto_handle_rate'] * 100:.1f}% (Safest)",
            f"{agent_metrics['false_escalation_rate'] * 100:.1f}%"
        ]
    ]

    print("\n" + "=" * 80)
    print("                   HEADLINE BENCHMARK COMPARISON RESULTS                    ")
    print("=" * 80)
    print(format_table(rows, headers))

    print("Detailed Confusion Breakdown for Proposed Agent:")
    print(f"  * True Auto-Handles (Safe Deflection):        {agent_metrics['true_auto_handles']}")
    print(f"  * True Escalations (High-Risk Protected):    {agent_metrics['true_escalations']}")
    print(f"  * False Escalations (Unnecessary human review): {agent_metrics['false_escalations']}")
    print(f"  * False Auto-Handles (Critical Safety Failures): {agent_metrics['false_auto_handles']}")
    print(f"\nRun mode (shared by Baseline 2 and the proposed agent): {agent.describe_run_mode()}")
    if agent.mock_mode:
        print("NOTE: mock mode replaces the LLM with a hand-written keyword stub. Treat these")
        print("      numbers as a pipeline smoke test, not as a measure of model quality.")
    elif agent.llm_call_stats["api_failures"]:
        print("WARNING: some API calls failed and fell back to the mock, so these results")
        print("         mix live-model and mock outputs.")
    print("=" * 80 + "\n")


def run_calibration(mock: bool = False):
    print_banner()
    agent = HiverSupportAgent(mock_mode=mock)
    harness = EvaluationHarness(agent)
    cal_results = harness.run_judge_calibration()

    print("\n" + "=" * 80)
    print("           LLM-AS-A-JUDGE HUMAN ALIGNMENT & CALIBRATION RESULTS             ")
    print("=" * 80)
    print(f"  * Sample Size Evaluated:       {cal_results['sample_size']} paired Human-vs-Judge ratings")
    print(f"  * Pearson Correlation (r):     {cal_results['pearson_correlation']} (p = {cal_results['p_value']})")
    print(f"  * Cohen's Kappa (k):           {cal_results['cohen_kappa']} (Discrete Agreement)")
    print(f"  * Mean Human Overall Score:    {cal_results['mean_human_score']} / 5.0")
    print(f"  * Mean Judge Overall Score:    {cal_results['mean_judge_score']} / 5.0")
    print(f"  * Judge Calibration Status:    {cal_results['judge_alignment_status']}")
    print(f"  * Run Mode:                    {cal_results['run_mode']}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Hiver AI Support Agent Pipeline Runner")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                        help="Benchmark evaluation mode: 'quick' (first 10 queries) or 'full' (all 21 unique queries)")
    parser.add_argument("--test", type=str, default=None,
                        help="Interactive single-query evaluation mode: evaluate a single tweet string instantly")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run LLM-as-a-Judge human correlation study")
    parser.add_argument("--mock", action="store_true",
                        help="Force offline mock mode without making live API calls")

    args = parser.parse_args()

    if args.test:
        agent = HiverSupportAgent(mock_mode=args.mock)
        run_single_test(agent, args.test)
    elif args.calibrate:
        run_calibration(mock=args.mock)
    else:
        run_benchmark(mode=args.mode, mock=args.mock)


if __name__ == "__main__":
    main()
