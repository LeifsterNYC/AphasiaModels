import re

PREFIX = "normalize in English: "


def clean_input(text):
    """Canonical form of a model input. Training data and the serving API
    both go through this, so the model never sees a casing or punctuation
    pattern at inference that it did not see in training."""
    text = text.lower().replace("’", "'")
    is_question = "?" in text
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^\w\s]|_", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text and is_question:
        text += " ?"
    return text
