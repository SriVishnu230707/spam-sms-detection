import re
import numpy as np
import pandas as pd

def normalize(text: str) -> str:
    """Normalize raw text by lowercasing and replacing high-signal patterns with tokens.
    
    Tokens used:
    - urltoken: URLs and web links
    - moneytoken: Currency symbols followed by numeric amounts
    - longnumtoken: 5+ digit numbers (phone numbers, shortcodes, PINs)
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"[£$€]\s?\d+[\d,.]*", " moneytoken ", text)
    text = re.sub(r"\d{5,}", " longnumtoken ", text)   # phone numbers, shortcodes
    return text

def extra_features(texts):
    """Extract dense metadata features from raw message texts:
    1. Message character length
    2. Count of numeric digits
    3. Proportion of uppercase characters (shouting / urgency)
    4. Binary indicator of web URL presence
    5. Binary indicator of currency symbol presence (£, $, €)
    """
    s = pd.Series(texts).astype(str)
    length = s.str.len().clip(lower=1)
    return np.column_stack([
        length,
        s.str.count(r"\d"),
        s.str.count(r"[A-Z]") / length,                # share of capitals
        s.str.contains(r"http|www\.", case=False, regex=True).astype(int),
        s.str.contains(r"[£$€]", regex=True).astype(int),
    ])

