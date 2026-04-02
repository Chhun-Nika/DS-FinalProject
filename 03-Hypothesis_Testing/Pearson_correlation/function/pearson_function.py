import pandas as pd
from scipy import stats


def pearson_test(data: pd.DataFrame, x: str, y: str, alpha: float = 0.05) -> dict:
    r, p = stats.pearsonr(data[x], data[y])

    strength = (
        "very weak" if abs(r) < 0.2 else
        "weak" if abs(r) < 0.4 else
        "moderate" if abs(r) < 0.6 else
        "strong" if abs(r) < 0.8 else
        "very strong"
    )
    direction = "positive" if r > 0 else "negative"
    decision = "Reject H0" if p < alpha else "Fail to reject H0"
    significant = "significant" if p < alpha else "not significant"

    return {
        "Variables": f"{x} vs {y}",
        "Correlation (r)": r,
        "p-value": p,
        "Relationship": f"{strength} {direction}",
        "Decision": decision,
        "Conclusion": f"The relationship between {x} and {y} is {significant}."
    }
