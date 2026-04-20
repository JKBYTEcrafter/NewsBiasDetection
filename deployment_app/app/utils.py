import re

def clean_text(text: str) -> str:
    """
    Standardize incoming text precisely as performed during model training.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text
