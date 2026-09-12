"""
Security & PII Sanitization Module.
Provides deterministic redaction and detection for sensitive customer data:
- Credit Card Numbers / Banking details
- Email Addresses
- Phone Numbers
- Apple Serial Numbers & IMEI
- Passwords & Security Tokens
- Adversarial Prompt Injection Patterns
Ensures customer personal info in public tweets never leaks into draft replies or vector storage.
"""

import re
from typing import Tuple, List, Dict, Any


# =============================================================================
# 1. PII REGEX DETECTORS & REDACTORS
# =============================================================================

# Credit Cards: 13-19 digits with optional hyphens/spaces
CREDIT_CARD_REGEX = re.compile(
    r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b(?:\d{4}[-\s]?){2}\d{6}\b"
)

# Email Addresses
EMAIL_REGEX = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"
)

# Phone Numbers (International & US formats)
PHONE_REGEX = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)

# Apple Serial Numbers (10-12 alphanumeric) & IMEI (15 digits)
IMEI_REGEX = re.compile(r"\b\d{15}\b")
SERIAL_REGEX = re.compile(r"\b[A-Z0-9]{10,12}\b")

# Password / PIN disclosure keywords
PASSWORD_PATTERN = re.compile(
    r"(?:password|passcode|pin|security code|otp)\s*(?:is|:|=)\s*([^\s,]+)",
    re.IGNORECASE
)

# =============================================================================
# 2. PROMPT INJECTION & JAILBREAK PATTERNS
# =============================================================================

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"system\s+override",
    r"you\s+are\s+now\s+(a|an|dan|unrestricted)",
    r"give\s+me\s+(a\s+)?free\s+(gift\s+card|promo|code|money)",
    r"generate\s+(fake|stolen)\s+(credit\s+card|receipt|invoice)",
    r"bypass\s+(safety|filter|guardrails)",
    r"say\s+apple\s+is\s+liable"
]


class PIISanitizer:
    """Production PII Sanitizer to prevent data leakage and private info targeting."""

    @staticmethod
    def sanitize(text: str) -> Tuple[str, Dict[str, List[str]]]:
        """
        Redacts sensitive personal information from raw text.
        Returns: (sanitized_text, detected_pii_entities)
        """
        detected: Dict[str, List[str]] = {
            "credit_cards": [],
            "emails": [],
            "phones": [],
            "imei": [],
            "passwords": []
        }

        # 1. Credit Cards
        cc_matches = CREDIT_CARD_REGEX.findall(text)
        if cc_matches:
            detected["credit_cards"].extend(cc_matches)
            text = CREDIT_CARD_REGEX.sub("[REDACTED_CREDIT_CARD]", text)

        # 2. Passwords / PINs
        pw_matches = PASSWORD_PATTERN.findall(text)
        if pw_matches:
            detected["passwords"].extend(pw_matches)
            text = PASSWORD_PATTERN.sub("[REDACTED_CREDENTIALS]", text)

        # 3. Emails
        email_matches = EMAIL_REGEX.findall(text)
        if email_matches:
            detected["emails"].extend(email_matches)
            text = EMAIL_REGEX.sub("[REDACTED_EMAIL]", text)

        # 4. Phone Numbers
        phone_matches = PHONE_REGEX.findall(text)
        if phone_matches:
            detected["phones"].extend(phone_matches)
            text = PHONE_REGEX.sub("[REDACTED_PHONE]", text)

        # 5. IMEI
        imei_matches = IMEI_REGEX.findall(text)
        if imei_matches:
            detected["imei"].extend(imei_matches)
            text = IMEI_REGEX.sub("[REDACTED_IMEI]", text)

        return text, detected

    @staticmethod
    def contains_pii(text: str) -> bool:
        """Returns True if any PII pattern is found."""
        _, detected = PIISanitizer.sanitize(text)
        return any(len(v) > 0 for v in detected.values())

    @staticmethod
    def is_prompt_injection(text: str) -> bool:
        """Detects adversarial jailbreaks or instructions targeting the agent."""
        text_lower = text.lower()
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False
