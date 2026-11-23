# Hiver AI Support Agent: Autonomous Twitter Support for @AppleSupport

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-%3C%202%20Minutes-brightgreen.svg)]()
[![Zero-Cost](https://img.shields.io/badge/Cost-%240.00%20(Free%20Tier)-success.svg)]()

First-reply triage agent for **@AppleSupport** on Twitter: classifies the tweet's intent, retrieves similar past Apple Support replies, and either drafts a reply or escalates to a human. Built to escalate whenever there is legal, safety, security or privacy risk.

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
# A. Quick smoke test on the first 10 golden queries
python run_pipeline.py --mode quick --mock

# B. Full benchmark on all 21 golden queries
python run_pipeline.py --mode full --mock

# C. LLM-as-a-Judge vs human scores (Pearson r & Cohen's Kappa)
python run_pipeline.py --calibrate --mock
```

### 3. Interactive Single-Query Testing
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

Golden set: **21 hand-labeled tweets** (11 standard, 5 edge, 5 high-risk; 11 need a human, 10 can be auto-handled). Numbers below are from `--mock` mode, where the LLM is replaced by a keyword stub. See [REPORT.md](REPORT.md) for how to read them.

| System | Intent Macro-F1 | Routing Accuracy | False Auto-Handles (of 11) | False Escalations (of 10) |
| :--- | :---: | :---: | :---: | :---: |
| Baseline 1: keywords + canned replies | 81.4% | 57% (12/21) | 9 | 0 |
| Baseline 2: zero-shot LLM (mock stub) | 77.6% | 48% (10/21) | 11 | 0 |
| Proposed agent (mock LLM) | 86.1% | 100% (21/21) | 0 | 0 |

Baseline 1 uses no LLM, so it is the one independent comparison: it auto-handles 9 of the 11 tweets that needed a human. The agent's 21/21 is a pipeline check, not a quality claim, because the mock rules were written with these tweets in view. On 10 unseen paraphrases it routed 7 correctly and auto-handled 2 of 6 risky ones (details in the report). No live-LLM numbers are currently reproducible.

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
                                 <  Not a Mandatory-Esc.   >
                                 <        Intent?          >
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
│   │   ├── applesupport_pairs.csv # 1,998 rows (772 unique) troubleshooting pairs
│   │   └── embeddings_cache.npy   # Precomputed MiniLM embeddings (384-dim, ~3 MB)
│   ├── golden_set.json            # 21 hand-labeled queries (11 standard / 5 edge / 5 high-risk)
│   └── calibration_sample.json    # 5 human-scored replies for judge calibration
│
├── src/
│   ├── config.py                  # Schemas, prompts, guardrails, and extension points
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
