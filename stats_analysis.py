        "t_statistic import numpy as np
import pandas as pd

from scipy.stats import (
    f_oneway,
    linregress,
    ttest_rel
)


def descriptive_statistics(
    df,
    value_column="final_polarization",
    group_column="personalization"
):

    result = (
        df.groupby(group_column)[value_column]
        .agg(
            mean="mean",
            sd="std",
            min="min",
            max="max",
            count="count"
        )
        .reset_index()
    )

    return result


def one_way_anova(
    df,
    value_column="final_polarization",
    group_column="personalization"
):

    groups = []

    for _, group in df.groupby(group_column):

        groups.append(
            group[value_column].dropna().values
        )

    F, p = f_oneway(*groups)

    return {
        "F_statistic": float(F),
        "p_value": float(p)
    }


def eta_squared(
    df,
    value_column="final_polarization",
    group_column="personalization"
):

    overall_mean = df[value_column].mean()

    ss_between = 0
    ss_total = 0

    for _, group in df.groupby(group_column):

        n = len(group)

        group_mean = group[value_column].mean()

        ss_between += (
            n * (group_mean - overall_mean) ** 2
        )

    ss_total = np.sum(
        (
            df[value_column] - overall_mean
        ) ** 2
    )

    if ss_total == 0:
        return 0.0

    return float(
        ss_between / ss_total
    )


def linear_trend_test(
    df,
    value_column="final_polarization",
    group_column="personalization"
):

    grouped = (
        df.groupby(group_column)[value_column]
        .mean()
        .reset_index()
    )

    x = grouped[group_column].values
    y = grouped[value_column].values

    result = linregress(x, y)

    return {
        "slope": float(result.slope),
        "r_squared": float(result.rvalue ** 2),
        "p_value": float(result.pvalue)
    }


def paired_comparison(
    df,
    value_column="final_polarization",
    low=0.0,
    high=1.0
):

    low_df = (
        df[df["personalization"] == low]
        .sort_values("run_id")
    )

    high_df = (
        df[df["personalization"] == high]
        .sort_values("run_id")
    )

    common_ids = sorted(
        set(low_df["run_id"])
        & set(high_df["run_id"])
    )

    low_values = (
        low_df[
            low_df["run_id"].isin(common_ids)
        ][value_column]
        .values
    )

    high_values = (
        high_df[
            high_df["run_id"].isin(common_ids)
        ][value_column]
        .values
    )

    t_stat, p_value = ttest_rel(
        low_values,
        high_values
    )

    return {": float(t_stat),
        "p_value": float(p_value),
        "n_pairs": len(common_ids)
    }


def significance_text(p):

    if p < 0.001:
        return "Có bằng chứng rất mạnh về sự khác biệt (p < 0,001)."

    elif p < 0.01:
        return "Có bằng chứng mạnh về sự khác biệt (p < 0,01)."

    elif p < 0.05:
        return "Có bằng chứng thống kê về sự khác biệt (p < 0,05)."

    else:
        return (
            "Chưa có đủ bằng chứng thống kê "
            "để kết luận có sự khác biệt (p ≥ 0,05)."
        )
