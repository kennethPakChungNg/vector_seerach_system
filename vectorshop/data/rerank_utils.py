def custom_boost(row):
    score = 0
    text_lower = row["combined_text"].lower()
    if "wireless" in text_lower:
        score += 1
    if "excellent noise cancellation" in text_lower:
        score += 2.0
    elif "noise cancellation" in text_lower or "anc" in text_lower:
        score += 0.5
    return score
