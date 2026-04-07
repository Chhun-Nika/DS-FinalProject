import pandas as pd
from scipy import stats


def categorical_vs_numerical(
    data: pd.DataFrame,
    category_col: str,
    numerical_col: str,
    alpha: float = 0.05
) -> dict:
    temp = data[[category_col, numerical_col]].dropna()

    groups = [
        group[numerical_col].values
        for _, group in temp.groupby(category_col)
    ]

    group_names = temp[category_col].unique().tolist()
    n_groups = len(group_names)

    if n_groups == 2:
        stat, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
        test_name = "Independent t-test"
    else:
        stat, p = stats.f_oneway(*groups)
        test_name = "One-way ANOVA"

    decision = "Reject H0" if p < alpha else "Fail to reject H0"
    significant = "significant" if p < alpha else "not significant"

    means = temp.groupby(category_col)[numerical_col].mean().round(4).to_dict()

    return {
        "Test": test_name,
        "Category Variable": category_col,
        "Numerical Variable": numerical_col,
        "Number of Groups": n_groups,
        "Group Means": means,
        "Statistic": round(stat, 4),
        "p-value": round(p, 6),
        "Decision": decision,
        "Conclusion": f"The difference in {numerical_col} across {category_col} groups is {significant}."
    }