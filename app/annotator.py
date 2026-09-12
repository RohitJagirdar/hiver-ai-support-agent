"""
Streamlit Annotator & Golden Set Inspector.
Allows rapid review, verification, and inspection of the 180-sample hand-labeled Golden Set.
"""

import os
import sys
import json
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import IntentCategory, ActionDecision

st.set_page_config(page_title="Golden Set Inspector", layout="wide")

GOLDEN_PATH = "data/golden_set.json"

st.title("🏷️ Golden Set Evaluation Dataset Inspector")
st.markdown("Review and verify the 180 hand-labeled customer support test cases.")

if not os.path.exists(GOLDEN_PATH):
    st.error(f"Missing {GOLDEN_PATH}")
    st.stop()

with open(GOLDEN_PATH, mode="r", encoding="utf-8") as f:
    golden_data = json.load(f)

# Sidebar filters
difficulty_filter = st.sidebar.multiselect(
    "Filter by Difficulty:",
    options=["STANDARD_CORE", "EDGE_AMBIGUOUS", "HIGH_RISK_ESCALATION"],
    default=["STANDARD_CORE", "EDGE_AMBIGUOUS", "HIGH_RISK_ESCALATION"]
)

intent_filter = st.sidebar.multiselect(
    "Filter by Intent:",
    options=[i.value for i in IntentCategory],
    default=[i.value for i in IntentCategory]
)

filtered = [
    item for item in golden_data
    if item.get("difficulty") in difficulty_filter and item.get("ground_truth_intent") in intent_filter
]

st.metric("Matching Examples", f"{len(filtered)} / {len(golden_data)}")

# Data Table / Card View
for idx, item in enumerate(filtered[:50], 1):
    with st.expander(f"#{item.get('id', idx)} | [{item.get('difficulty')}] Intent: {item.get('ground_truth_intent')} | Action: {item.get('ground_truth_action')}"):
        st.markdown(f"**Customer Query:** {item.get('query')}")
        st.markdown(f"**Annotation Rationale:** {item.get('rationale')}")
