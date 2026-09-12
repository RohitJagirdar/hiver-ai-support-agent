"""
Data Extractor & Cleaning Pipeline.
Extracts high-signal, troubleshooting-rich conversation pairs from the Kaggle Customer Support dataset.
Filters out generic 'Please send us a DM' boilerplate to ensure RAG retrieval contains substantive advice.
"""

import os
import re
import csv
from typing import List, Dict, Tuple, Optional


DM_BOILERPLATE_REGEX = re.compile(
    r"^(hi|hello|hey)?\s*(there,?)?\s*(please\s+)?(dm|send\s+us\s+a\s+dm|direct\s+message|pm)\s+us",
    re.IGNORECASE
)

ACTIONABLE_KEYWORDS = [
    "restart", "settings", "update", "reset", "battery", "appleid",
    "icloud", "support.apple.com", "backup", "turn off", "force",
    "steps", "article", "guide", "check", "press", "hold"
]


def clean_tweet_text(text: str) -> str:
    """Strips handle mentions, decodes HTML entities, and normalizes whitespace."""
    if not text:
        return ""
    # Strip @usernames
    text = re.sub(r"@\w+", "", text)
    # Decode basic HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    # Normalize excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_high_signal_response(response_text: str) -> bool:
    """
    Evaluates whether an AppleSupport reply contains substantive troubleshooting instructions
    rather than purely passive 'Please DM us' boilerplate.
    """
    cleaned = clean_tweet_text(response_text).lower()
    
    # 1. Reject very short responses (< 45 characters)
    if len(cleaned) < 45:
        return False
        
    # 2. Check for DM boilerplate
    if DM_BOILERPLATE_REGEX.search(cleaned):
        # Only accept if it ALSO contains substantive troubleshooting instructions
        has_actionable_step = any(kw in cleaned for kw in ACTIONABLE_KEYWORDS)
        if not has_actionable_step:
            return False
            
    return True


def extract_pairs_from_twcs(
    twcs_csv_path: str,
    output_csv_path: str,
    target_brand: str = "AppleSupport",
    max_pairs: int = 2000
) -> int:
    """
    Extracts customer inquiry -> AppleSupport resolution pairs from raw twcs.csv.
    Optimized for streaming so it does not load the entire 800MB CSV into RAM.
    """
    if not os.path.exists(twcs_csv_path):
        raise FileNotFoundError(f"Raw Kaggle dataset not found at {twcs_csv_path}")

    print(f"Streaming from {twcs_csv_path} for brand @{target_brand}...")
    
    # Pass 1: Index AppleSupport direct replies and their inbound tweet IDs
    # Schema: tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id
    brand_replies: Dict[int, str] = {}
    inbound_needed = set()

    with open(twcs_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("author_id") == target_brand:
                resp_to = row.get("in_response_to_tweet_id")
                if resp_to and resp_to.isdigit():
                    resp_to_id = int(resp_to)
                    reply_text = row.get("text", "")
                    if is_high_signal_response(reply_text):
                        brand_replies[resp_to_id] = reply_text
                        inbound_needed.add(resp_to_id)
                        if len(brand_replies) >= max_pairs * 2:
                            break

    print(f"Found {len(brand_replies)} high-signal candidate replies from @{target_brand}.")

    # Pass 2: Retrieve original customer questions
    extracted_pairs: List[Dict[str, Any]] = []
    with open(twcs_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_id = row.get("tweet_id")
            if t_id and t_id.isdigit() and int(t_id) in inbound_needed:
                t_id_int = int(t_id)
                inbound_text = clean_tweet_text(row.get("text", ""))
                outbound_text = clean_tweet_text(brand_replies[t_id_int])
                
                # Check for minimum customer question length
                if len(inbound_text) >= 20:
                    extracted_pairs.append({
                        "id": len(extracted_pairs),
                        "tweet_id": t_id_int,
                        "inbound_text": inbound_text,
                        "outbound_text": outbound_text
                    })
                    if len(extracted_pairs) >= max_pairs:
                        break

    # Save to output CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "tweet_id", "inbound_text", "outbound_text"])
        writer.writeheader()
        writer.writerows(extracted_pairs)

    print(f"Successfully wrote {len(extracted_pairs)} high-signal pairs to {output_csv_path}.")
    return len(extracted_pairs)


if __name__ == "__main__":
    raw_path = "data/raw/twcs.csv"
    out_path = "data/processed/applesupport_pairs.csv"
    if os.path.exists(raw_path):
        extract_pairs_from_twcs(raw_path, out_path)
    else:
        print(f"Note: {raw_path} not found. Using pre-curated dataset in data/processed/")
