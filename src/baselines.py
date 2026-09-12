"""
Baselines Module for Comparative Evaluation.
Implements the required baselines per the Hiver assignment specification:
1. Baseline 1 (Trivial Baseline): Regex keyword matching + Canned static reply templates + Naive sentiment escalation.
2. Baseline 2 (Simple Baseline): Zero-shot generic LLM without RAG retrieval context and without confidence calibration.
"""

import re
import json
import logging
from typing import Dict, Any, Optional
from src.config import IntentCategory, ActionDecision, AgentResolutionOutput, UrgencyLevel
from src.agent import HiverSupportAgent

logger = logging.getLogger(__name__)


# =============================================================================
# BASELINE 1: TRIVIAL RULE-BASED / CANNED TEMPLATE SYSTEM
# =============================================================================

CANNED_REPLIES = {
    IntentCategory.HARDWARE_ISSUES: "Please visit your nearest Apple Retail Store or check https://support.apple.com/repair for hardware service.",
    IntentCategory.SOFTWARE_UPDATE: "Please ensure your device is connected to Wi-Fi and restart your device to complete the update.",
    IntentCategory.BILLING_SUBSCRIPTIONS: "For billing questions or refund requests, please visit https://reportaproblem.apple.com.",
    IntentCategory.ACCOUNT_SECURITY: "If you cannot access your account, please visit https://iforgot.apple.com to reset your credentials.",
    IntentCategory.DEVICE_SETUP_SYNC: "Please toggle Bluetooth or Wi-Fi off and back on in Settings, then attempt to pair again.",
    IntentCategory.GENERAL_OTHER: "Thank you for reaching out to Apple Support. Please check https://support.apple.com for general information."
}

KEYWORD_INTENT_MAP = {
    IntentCategory.HARDWARE_ISSUES: ["screen", "battery", "charge", "charging", "display", "speaker", "cracked", "black screen", "button", "overheating"],
    IntentCategory.SOFTWARE_UPDATE: ["update", "ios", "version", "install", "stuck", "apple logo", "freeze", "crash", "restore"],
    IntentCategory.BILLING_SUBSCRIPTIONS: ["bill", "charge", "refund", "subscription", "cancel", "payment", "invoice", "applecare", "receipt"],
    IntentCategory.ACCOUNT_SECURITY: ["apple id", "password", "locked", "hacked", "stolen", "2fa", "security", "verification", "phishing"],
    IntentCategory.DEVICE_SETUP_SYNC: ["airpod", "airpods", "pair", "bluetooth", "sync", "icloud photos", "transfer", "backup", "connect"]
}

ESCALATION_KEYWORDS = ["lawyer", "sue", "terrible", "worst", "unacceptable", "furious", "police", "stolen", "scam", "fraud"]


class TrivialBaseline:
    """Baseline 1: Rule-based keyword classifier with static template replies."""

    def process_query(self, query: str) -> AgentResolutionOutput:
        q_lower = query.lower()

        # 1. Intent Classification via Keyword Matching
        matched_intent = IntentCategory.GENERAL_OTHER
        for intent, keywords in KEYWORD_INTENT_MAP.items():
            if any(re.search(r'\b' + re.escape(kw) + r'\b', q_lower) for kw in keywords):
                matched_intent = intent
                break

        # 2. Naive Escalation Rule
        should_escalate = any(kw in q_lower for kw in ESCALATION_KEYWORDS) or len(query) > 180
        action = ActionDecision.ESCALATE_TO_HUMAN if should_escalate else ActionDecision.AUTO_HANDLE
        reason = "Keyword-based escalation trigger." if should_escalate else "Handled by static template."

        # 3. Static Canned Reply
        draft_reply = CANNED_REPLIES.get(matched_intent, CANNED_REPLIES[IntentCategory.GENERAL_OTHER])

        return AgentResolutionOutput(
            predicted_intent=matched_intent,
            confidence_score=0.60 if matched_intent != IntentCategory.GENERAL_OTHER else 0.30,
            action=action,
            escalation_reason=reason,
            draft_reply=draft_reply,
            urgency_level=UrgencyLevel.HIGH if should_escalate else UrgencyLevel.LOW,
            retrieved_reference_ids=[]
        )


# =============================================================================
# BASELINE 2: SIMPLE ZERO-SHOT LLM (NO RAG, NO CALIBRATION)
# =============================================================================

ZERO_SHOT_PROMPT = """Classify this customer tweet and generate a response.
Tweet: "{query}"

Intents: ACCOUNT_SECURITY, HARDWARE_ISSUES, SOFTWARE_UPDATE, BILLING_SUBSCRIPTIONS, DEVICE_SETUP_SYNC, GENERAL_OTHER
Decide: AUTO_HANDLE or ESCALATE_TO_HUMAN.

Respond in JSON format:
{
  "predicted_intent": "<intent>",
  "confidence_score": <float 0-1>,
  "action": "<AUTO_HANDLE or ESCALATE_TO_HUMAN>",
  "escalation_reason": "<reason>",
  "draft_reply": "<reply>",
  "urgency_level": "LOW" | "MEDIUM" | "HIGH"
}
"""

class SimpleZeroShotBaseline:
    """Baseline 2: Zero-shot LLM without RAG retrieval context and without defensive post-gating."""

    def __init__(self, agent: HiverSupportAgent):
        self.agent = agent

    def process_query(self, query: str) -> AgentResolutionOutput:
        prompt = ZERO_SHOT_PROMPT.replace("{query}", query)
        raw_json_str = self.agent._call_llm_api(prompt=prompt, system_prompt="You are a helpful customer support bot.")

        try:
            cleaned_json = re.sub(r"^```json\s*", "", raw_json_str.strip())
            cleaned_json = re.sub(r"\s*```$", "", cleaned_json)
            parsed_dict = json.loads(cleaned_json)
            parsed_dict["retrieved_reference_ids"] = []
            return AgentResolutionOutput(**parsed_dict)
        except Exception:
            # Fallback
            return AgentResolutionOutput(
                predicted_intent=IntentCategory.GENERAL_OTHER,
                confidence_score=0.50,
                action=ActionDecision.AUTO_HANDLE,
                escalation_reason="Zero-shot fallback",
                draft_reply="Please contact Apple Support for assistance with your device.",
                urgency_level=UrgencyLevel.MEDIUM,
                retrieved_reference_ids=[]
            )
