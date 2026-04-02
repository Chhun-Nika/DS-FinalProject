import pandas as pd
from scipy import stats


def independent_t_test(
    data: pd.DataFrame,
    group_col: str,
    target_col: str,
    group_a: str,
    group_b: str,
    alpha: float = 0.05,
) -> dict:
    sample_a = data[data[group_col] == group_a][target_col]
    sample_b = data[data[group_col] == group_b][target_col]

    t_stat, p = stats.ttest_ind(sample_a, sample_b, equal_var=False)
    decision = "Reject H0" if p < alpha else "Fail to reject H0"
    significant = "significant" if p < alpha else "not significant"

    return {
        "Comparison": f"{group_a} vs {group_b} on {target_col}",
        f"Mean {group_a}": sample_a.mean(),
        f"Mean {group_b}": sample_b.mean(),
        "t-statistic": t_stat,
        "p-value": p,
        "Decision": decision,
        "Conclusion": (
            f"The difference in {target_col} between "
            f"{group_a} and {group_b} is {significant}."
        ),
    }
