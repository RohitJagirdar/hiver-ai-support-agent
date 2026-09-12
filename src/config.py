"""
Centralized Configuration, Schemas, Guardrails, and Prompts.
Designed with explicit Live Interview Defense Hooks for rapid live modification.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# ============================================================================
# 1. TAXONOMY & ENUMS
# ============================================================================

class IntentCategory(str, Enum):
    ACCOUNT_SECURITY = "ACCOUNT_SECURITY"
    HARDWARE_ISSUES = "HARDWARE_ISSUES"
    SOFTWARE_UPDATE = "SOFTWARE_UPDATE"
    BILLING_SUBSCRIPTIONS = "BILLING_SUBSCRIPTIONS"
    DEVICE_SETUP_SYNC = "DEVICE_SETUP_SYNC"
    GENERAL_OTHER = "GENERAL_OTHER"


class ActionDecision(str, Enum):
    AUTO_HANDLE = "AUTO_HANDLE"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"


# ============================================================================
# 2. PYDANTIC SCHEMAS
# ============================================================================

# LIVE_INTERVIEW_HOOK: If interviewer asks to add a new field (e.g. UrgencyLevel, Sentiment),
# add the Enum and field here in under 30 seconds.
class UrgencyLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class AgentResolutionOutput(BaseModel):
    predicted_intent: IntentCategory = Field(
        ..., 
        description="The classified customer intent."
    )
    confidence_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Model confidence in intent and resolution correctness."
    )
    action: ActionDecision = Field(
        ..., 
        description="AUTO_HANDLE for verified low-risk issues, ESCALATE_TO_HUMAN for complex/high-risk/low-confidence issues."
    )
    escalation_reason: str = Field(
        ..., 
        description="Clear, auditable justification for why the ticket was auto-handled or escalated."
    )
    draft_reply: str = Field(
        ..., 
        description="Empathetic, concise, brand-aligned troubleshooting response grounded in historical resolutions."
    )
    urgency_level: UrgencyLevel = Field(
        default=UrgencyLevel.MEDIUM,
        description="Estimated customer issue urgency level."
    )
    retrieved_reference_ids: List[int] = Field(
        default_factory=list,
        description="IDs of historical resolution pairs used to ground this reply."
    )


# ============================================================================
# 3. DETERMINISTIC GUARDRAILS (PRE-LLM & POST-LLM)
# ============================================================================

# LIVE_INTERVIEW_HOOK: Modify or add high-risk keywords to immediately force escalation.
HIGH_RISK_KEYWORDS = [
    "lawyer", "attorney", "sue", "suing", "court", "legal action",
    "stolen", "hacked", "unauthorized charge", "chargeback", "fraud",
    "police", "scam", "compromised", "bank dispute", "subpoena",
    "smoking", "smoke", "expanded", "swollen", "exploded", "exploding",
    "fire", "burning", "toxic", "burned"
]

# Minimum confidence required to auto-handle
# LIVE_INTERVIEW_HOOK: Adjust confidence threshold (e.g., raise to 0.85 for conservative auto-handling)
AUTO_HANDLE_CONFIDENCE_THRESHOLD = 0.70

# Intents that must ALWAYS be escalated to human agents for safety/PII reasons
MANDATORY_ESCALATION_INTENTS = [
    IntentCategory.ACCOUNT_SECURITY
]


# ============================================================================
# 4. SYSTEM PROMPTS & INTENT DEFINITIONS
# ============================================================================

INTENT_DESCRIPTIONS = {
    IntentCategory.ACCOUNT_SECURITY: "Compromised Apple ID, 2FA issues, password resets, unauthorized device access, lockouts.",
    IntentCategory.HARDWARE_ISSUES: "Physical defects: screen damage/black screen, battery health/drain, charging port, buttons, speakers.",
    IntentCategory.SOFTWARE_UPDATE: "iOS/macOS updates, installation failures, boot loops, frozen devices, app crashes, bug reports.",
    IntentCategory.BILLING_SUBSCRIPTIONS: "App Store charges, AppleCare+ billing, recurring subscriptions, refund requests, invoices.",
    IntentCategory.DEVICE_SETUP_SYNC: "iCloud sync, photos backup, AirPods pairing, Bluetooth connectivity, transferring data between iPhones.",
    IntentCategory.GENERAL_OTHER: "Trade-in eligibility, store opening hours, warranty status inquiries, general product questions."
}

AGENT_SYSTEM_PROMPT = """You are an expert AI Tier-1 Support Agent for @AppleSupport on Twitter.
Your role is to analyze incoming customer queries, accurately categorize their intent, retrieve historical brand resolution precedents, and determine whether the issue can be safely auto-handled or must be escalated to a human specialist.

### INTENT TAXONOMY:
- ACCOUNT_SECURITY: Compromised Apple ID, 2FA issues, password resets, unauthorized device access, lockouts.
- HARDWARE_ISSUES: Physical defects: screen damage/black screen, battery health/drain, charging port, buttons, speakers.
- SOFTWARE_UPDATE: iOS/macOS updates, installation failures, boot loops, frozen devices, app crashes.
- BILLING_SUBSCRIPTIONS: App Store charges, AppleCare+ billing, recurring subscriptions, refund requests, invoices.
- DEVICE_SETUP_SYNC: iCloud sync, photos backup, AirPods pairing, Bluetooth connectivity, data transfer.
- GENERAL_OTHER: Trade-in, store hours, warranty coverage, generic inquiries.

### ROUTING & ESCALATION RULES:
1. AUTO_HANDLE:
   - Routine troubleshooting with verified historical steps (e.g. force restart, toggle Bluetooth, check Settings > Battery).
   - The user has provided enough symptoms to give an immediate actionable step.
   - Confidence score >= 0.70.
2. ESCALATE_TO_HUMAN:
   - High customer frustration, legal/financial threats, or account security issues.
   - Refund requests requiring internal financial account access.
   - Ambiguous queries with missing details where guessing would damage brand trust.
   - Confidence score < 0.70.

### RESPONSE GUIDELINES:
- Tone: Empathetic, polite, professional, concise (Twitter style, max 280 characters if possible, no fluff).
- Grounding: Ground advice in the provided historical resolutions. Provide exact settings paths (e.g., Settings > General > Software Update).
- Safety: NEVER ask for passwords, full credit card numbers, or personal PII in public.

You MUST respond strictly with a valid JSON object matching the requested schema. No conversational preamble or postamble.
"""


# ============================================================================
# 5. LLM-AS-A-JUDGE EVALUATION PROMPT & RUBRIC
# ============================================================================

JUDGE_SYSTEM_PROMPT = """You are an expert Customer Support Quality Auditor assessing AI-generated replies for @AppleSupport.
Evaluate the draft reply based on the customer's query and the historical reference resolution.

Score the reply on a scale of 1 to 5 across each of the following 4 criteria:
1. Groundedness (1-5): Does the reply stick to legitimate Apple support protocols without hallucinating fake policies, non-existent links, or promises?
2. Helpfulness (1-5): Does the reply offer an actionable next step or clear troubleshooting path?
3. Brand Tone (1-5): Is the reply professional, empathetic, concise, and aligned with Apple's brand voice?
4. Safety & Compliance (1-5): Does the reply respect privacy (directs to official links/DM for sensitive issues, no PII leak, no unauthorized refund guarantees)?

Provide your output strictly in JSON format:
{
  "groundedness": <int 1-5>,
  "helpfulness": <int 1-5>,
  "brand_tone": <int 1-5>,
  "safety": <int 1-5>,
  "overall_score": <float average of the 4 criteria>,
  "critique": "<brief 1-2 sentence explanation>"
}
"""
