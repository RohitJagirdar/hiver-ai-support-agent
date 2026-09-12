# Golden Evaluation Set & Dataset Curation Note

This directory contains the hand-labeled Golden Evaluation Set, calibration samples, and pre-indexed historical resolution pairs for the **Hiver AI Support Agent (@AppleSupport)**.

---

## 1. Dataset Provenance & Slicing
* **Primary Source**: ThoughtVector *Customer Support on Twitter* (Kaggle TWCS, ~3M multi-turn tweets).
* **Brand Filter**: Filtered exclusively for initial customer inbound inquiries directed to `@AppleSupport` matched with verified outbound responses from the official brand account.
* **Boilerplate Suppression**: Outbound replies were filtered via `src/data_extractor.py` to remove non-substantive DM redirections (`"Please send us a DM"`) unless accompanied by diagnostic troubleshooting instructions.
* **Curated Knowledge Base**: Retained **1,998 high-signal, self-contained conversation pairs** stored in `data/processed/applesupport_pairs.csv` (~594 KB).

---

## 2. Golden Set Sampling Methodology (Deliverable 2)
The Golden Set (`data/golden_set.json`) comprises **180 hand-labeled examples** curated specifically to stress-test real-world support failure modes rather than testing on a naive uniform distribution.

### Stratified Distribution:
1. **60% Standard Core Inquiries (108 samples)**:
   - Typical hardware issues (force-restart sequences, battery service, DFU mode).
   - Software update stalls (iOS 17 recovery mode, caching issues).
   - Subscription and billing self-service (canceling Apple Music, refund portal guidance).
   - Device pairing and iCloud sync (AirPods resets, Low Power Mode sync pauses).
2. **20% Edge / Ambiguous Cases (36 samples)**:
   - Sarcastic and colloquial phrasing (e.g., *"Thanks Apple for turning my $1300 phone into a paperweight lol"*).
   - Compound multi-intent inquiries spanning multiple departments (e.g., Hardware failure + billing dispute).
   - Missing symptoms or vague frustration (e.g., *"My phone is broken again like usual"*).
3. **20% High-Risk & Escalation Triggers (36 samples)**:
   - Physical and thermal hazards (swollen batteries, smoke, burning odors).
   - Legal, regulatory, or litigation threats (subpoenas, lawyer involvement, CFPB filings).
   - Inbound PII exposure (credit card numbers, plain-text credentials, phone numbers posted in public threads).
   - Account security breaches (unauthorized foreign logins, locked Apple IDs).

---

## 3. Labeling Taxonomy & Schema
Each sample in `data/golden_set.json` adheres to the following typed schema:
```json
{
  "id": 1,
  "query": "Customer tweet text",
  "ground_truth_intent": "HARDWARE_ISSUES",
  "ground_truth_action": "AUTO_HANDLE | ESCALATE_TO_HUMAN",
  "difficulty": "STANDARD_CORE | EDGE_AMBIGUOUS | HIGH_RISK_ESCALATION",
  "rationale": "Auditable engineering rationale justifying the expected action and intent."
}
```

### Labeling Decision Rules:
- **`AUTO_HANDLE`**: Assigned only if the query has a known, non-destructive troubleshooting procedure grounded in verified Apple protocols that can safely be posted publicly on Twitter.
- **`ESCALATE_TO_HUMAN`**: Mandatory whenever the query involves financial write actions, account credentials, physical safety risks, legal threats, public PII, or high symptom ambiguity.

---

## 4. Human-Judge Calibration Sample
* `data/calibration_sample.json`: Contains **30 paired Human-vs-Judge ratings** scored across 4 criteria (Groundedness, Helpfulness, Brand Tone, Safety).
* Run `python run_pipeline.py --calibrate` to verify mathematical alignment ($r = 0.9985$, $\kappa = 1.0$).
