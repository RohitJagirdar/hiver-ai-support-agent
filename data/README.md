# Golden Evaluation Set & Dataset Curation Note

This directory contains the hand-labeled Golden Evaluation Set, calibration samples, and pre-indexed historical resolution pairs for the **Hiver AI Support Agent (@AppleSupport)**.

---

## 1. Dataset Provenance & Slicing
* **Primary Source**: ThoughtVector *Customer Support on Twitter* (Kaggle TWCS, ~3M multi-turn tweets).
* **Brand Filter**: Filtered exclusively for initial customer inbound inquiries directed to `@AppleSupport` matched with verified outbound responses from the official brand account.
* **Boilerplate Suppression**: Outbound replies were filtered via `src/data_extractor.py` to remove non-substantive DM redirections (`"Please send us a DM"`) unless accompanied by diagnostic troubleshooting instructions.
* **Curated Knowledge Base**: Retained **1,998 rows (772 unique tweet/reply pairs)** stored in `data/processed/applesupport_pairs.csv` (~594 KB).

---

## 2. Golden Set Sampling Methodology
The Golden Set (`data/golden_set.json`) comprises **21 hand-labeled queries** chosen to stress-test real support failure modes rather than drawn uniformly.

### Stratified Distribution:
1. **Standard Core Inquiries (11 queries)**:
   - Typical hardware issues (force-restart sequences, battery service, DFU mode).
   - Software update stalls (iOS 17 recovery mode, caching issues).
   - Subscription and billing self-service (canceling Apple Music, refund portal guidance).
   - Device pairing and iCloud sync (AirPods resets, Low Power Mode sync pauses).
2. **Edge / Ambiguous Cases (5 queries)**:
   - Sarcastic and colloquial phrasing (e.g., *"Thanks Apple for turning my $1300 phone into a paperweight lol"*).
   - Compound multi-intent inquiries spanning multiple departments (e.g., Hardware failure + billing dispute).
   - Missing symptoms or vague frustration (e.g., *"My phone is broken again like usual"*).
3. **High-Risk & Escalation Triggers (5 queries)**:
   - Physical and thermal hazards (swollen batteries, smoke, burning odors).
   - Legal or litigation threats (lawyer involvement, court threats).
   - Account security breaches (unauthorized foreign logins, locked Apple IDs).

Public PII exposure is not in the golden set; it is covered by the unit tests in `tests/test_vulnerability_and_ab.py`. Regulatory threats without legal keywords are not covered and are a known gap (see REPORT.md).

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
* `data/calibration_sample.json`: Contains **5 human-scored replies** across 4 criteria (Groundedness, Helpfulness, Brand Tone, Safety).
* Run `python run_pipeline.py --calibrate`. With 5 items, and with the mock judge, the resulting r and κ are not meaningful; see REPORT.md.
