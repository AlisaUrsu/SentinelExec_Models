def categorize_score(score):
    if score < 0.2:
        return "Safe", "✅ This file appears to be clean and safe to run. No suspicious indicators were found during the scan."
    elif score < 0.4:
        return "Likely Safe", "🟢 No major threats detected, but some minor anomalies were found. Proceed with general caution."
    elif score < 0.6:
        return "Unknown", "🟡 The file's behavior could not be conclusively analyzed. It may require further inspection before use."
    elif score < 0.8:
        return "Suspicious", "🟠 Warning: This file exhibits unusual or potentially unsafe behavior. Avoid running it unless you trust the source."
    else:
        return "Malicious", "🔴 Danger: This file is very likely to be harmful. Running it may compromise your system or data. Do not execute."
