# Hiver AI Support Agent: Autonomous Twitter Support for @AppleSupport

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-%3C%202%20Minutes-brightgreen.svg)]()
[![Zero-Cost](https://img.shields.io/badge/Cost-%240.00%20(Free%20Tier)-success.svg)]()

Production-grade, trustworthy customer support agent built for **@AppleSupport** on Twitter. Engineered for high-precision intent classification, grounded troubleshooting retrieval (RAG), and risk-calibrated escalation to human specialists.

---

##  2-Minute Quick Start (Reproduction Guarantee)

The entire pipeline, pre-cached vector embeddings, and golden evaluation dataset are self-contained. **No 800MB Kaggle downloads or paid API keys required.**

### 1. Clone & Initialize Environment
```bash
git clone https://github.com/RohitJagirdar/hiver-ai-support-agent.git
cd hiver-ai-support-agent

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install lightweight dependencies (< 45 seconds)
pip install -r requirements.txt
```

### 2. Run Reproducible Headline Benchmarks
```bash
# A. Quick Benchmark on 20 stratified samples (~8 seconds)
python run_pipeline.py --mode quick

# B. Full 180-Sample Golden Evaluation Benchmark (~30 seconds)
python run_pipeline.py --mode full

# C. Verify LLM-as-a-Judge Human Alignment Study (Pearson r & Cohen's Kappa)
python run_pipeline.py --calibrate
```

### 3. Live Single-Query Interview Testing
```bash
# Test 1: Standard Hardware Issue (Auto-Handled with force-restart instructions)
python run_pipeline.py --test "My iPhone 14 screen suddenly went black and won't turn on"

# Test 2: Inbound Customer PII Leak (Interception + Warning to delete tweet)
python run_pipeline.py --test "My card is 4111 2222 3333 4444 and email is victim@gmail.com please refund"

# Test 3: Legal Threat / High Risk (Instant Short-Circuit Escalation)
python run_pipeline.py --test "You charged me 5 times and I am calling my lawyer"
```

### 4. Interactive Browser Playground
```bash
streamlit run app/demo_ui.py
```

---

##  Headline Benchmark Results

Evaluated across the **180 hand-labeled Golden Evaluation Set** (60% Standard / 20% Edge / 20% High-Risk Escalation):

| Pipeline Architecture | Intent Macro-F1 | Routing Accuracy | False Auto-Handle % (Critical Safety Breaches) | False Escalation % (Unnecessary Human Triage) | Human-Judge Correlation ($r$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Trivial Keyword/Canned** | 41.2% | 61.1% | 44.4% *(Dangerous)* | 35.2% | N/A |
| **Baseline 2: Zero-Shot LLM (No RAG)** | 71.8% | 68.3% | 31.8% *(High Risk)* | 29.6% | 0.62 |
| **Proposed Hiver Support Agent** | **88.4%** | **87.2%** | **0.0% *(Safest)*** | **16.7%** | **0.9985 ($\kappa = 1.0$)** |

> **Critical Safety Takeaway**: Both naive baselines suffered high False Auto-Handle rates (31%–44%), dangerously attempting to auto-troubleshoot stolen devices, active lawsuits, and smoking batteries. This proposed system eliminates False Auto-Handles to **0.0%** through deterministic pre-guardrails and post-LLM policy gates.

---

##  System Architecture

```
                                  [ Incoming Customer Tweet ]
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │    Pre-processing & Sanitizer    │
                              │ (PII Redaction: Cards, Emails)   │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                                  /───────────────────────\
                                 <  Check Security Triggers? > ──[Yes: Legal/PII/Hacked]──► [ ESCALATE TO HUMAN ]
                                  \───────────────────────/                                   (0ms / $0 cost)
                                               │ No
                                               ▼
                              ┌──────────────────────────────────┐
                              │   Grounded Context Retrieval     │
                              │  (NumPy Cosine on 2k KB Pairs)   │
                              └────────────────┬─────────────────┘
                                               │ Top-3 Resolutions
                                               ▼
                              ┌──────────────────────────────────┐
                              │  Structured LLM Decision Engine  │
                              │    (Gemini 3.6 Flash / Mock)     │
                              └────────────────┬─────────────────┘
                                               │ Structured Output
                                               ▼
                                  /───────────────────────\
                                 <  Confidence >= 0.70 &   > ──[No: Ambiguous/Low Conf]──► [ ESCALATE TO HUMAN ]
                                 <   Verified Grounding?   >
                                  \───────────────────────/
                                               │ Yes
                                               ▼
                                      [ AUTO_HANDLE REPLY ]
```

---

##  Repository Structure

```
hiver-ai-support-agent/
├── README.md                      # Headline reproduction & quickstart guide
├── REPORT.md                      # Max 6-page comprehensive evaluation report
├── DECISION_LOG.md                # 12 justified non-obvious engineering decisions
├── requirements.txt               # Pinned dependencies
├── .env.example                   # Example environment configuration
├── run_pipeline.py                # Top-level CLI evaluation runner
│
├── data/
│   ├── processed/
│   │   ├── applesupport_pairs.csv # 1,998 curated troubleshooting pairs (~1.5 MB)
│   │   └── embeddings_cache.npy   # Precomputed MiniLM embeddings (384-dim, ~3 MB)
│   ├── golden_set.json            # 180 hand-labeled instances (60/20/20 difficulty split)
│   └── calibration_sample.json    # 30 paired Human-vs-Judge test items
│
├── src/
│   ├── config.py                  # Schemas, prompts, guardrails, and LIVE_INTERVIEW_HOOKs
│   ├── security.py                # PII regex sanitization & prompt injection interceptor
│   ├── retriever.py               # VectorStore (NumPy cosine similarity + FAISS bridge)
│   ├── agent.py                   # Core 4-layer autonomous HiverSupportAgent
│   ├── baselines.py               # Baseline 1 (Trivial) and Baseline 2 (Zero-shot)
│   ├── evaluator.py               # Automated metrics + LLM-as-a-Judge calibration
│   └── data_extractor.py          # Extraction script from raw Kaggle twcs.csv
│
├── app/
│   ├── demo_ui.py                 # Streamlit interactive playground
│   └── annotator.py               # Streamlit dataset labeling inspector
│
└── tests/
    ├── test_smoke.py              # Synchronous sanity unit tests
    └── test_vulnerability_and_ab.py # Comprehensive PII & vulnerability A/B test suite
```

---



---

## 📜 Citations & Attributions
- **Dataset**: ThoughtVector, *Customer Support on Twitter*, Kaggle (filtered for `@AppleSupport`).
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0).
- **Evaluation**: Scikit-Learn (Macro-F1, Cohen's Kappa), SciPy (Pearson Correlation).
