# src/bias_checker.py

gendered_words = [
    "nurturing", "supportive", "dominant", "aggressive", "competitive"
]

age_indicators = [
    "born in", "20+ years experience", "senior citizen", "old", "young"
]

def detect_bias(text):
    text_lower = text.lower()
    flags = {
        "gendered_words": [word for word in gendered_words if word in text_lower],
        "age_indicators": [phrase for phrase in age_indicators if phrase in text_lower]
    }
    return flags
