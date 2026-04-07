import pandas as pd
from scipy import stats


def numerical_vs_numerical(
    data: pd.DataFrame,
    x: str,
    y: str,
    alpha: float = 0.05
) -> dict:
    temp = data[[x, y]].dropna()

    r, p = stats.pearsonr(temp[x], temp[y])

    if abs(r) < 0.2:
        strength = "very weak"
    elif abs(r) < 0.4:
        strength = "weak"
    elif abs(r) < 0.6:
        strength = "moderate"
    elif abs(r) < 0.8:
        strength = "strong"
    else:
        strength = "very strong"

    if r > 0:
        direction = "positive"
    elif r < 0:
        direction = "negative"
    else:
        direction = "no relationship"

    decision = "Reject H0" if p < alpha else "Fail to reject H0"
    significant = "significant" if p < alpha else "not significant"

    return {
        "Test": "Pearson Correlation",
        "Variables": f"{x} vs {y}",
        "Correlation (r)": round(r, 4),
        "p-value": round(p, 6),
        "Relationship": f"{strength} {direction}",
        "Decision": decision,
        "Conclusion": f"The relationship between {x} and {y} is {significant}."
    }