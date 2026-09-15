import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from simulation import simulate
from metrics import (
    calculate_final_metrics,
    calculate_trajectory_metrics
)

from stats_analysis import (
    descriptive_statistics,
    one_way_anova,
    eta_squared,
    linear_trend_test,
    paired_comparison,
    significance_text
)


# ============================================================
# CẤU HÌNH NGHIÊN CỨU
# ============================================================

NUM_AGENTS = 100
STEPS = 50

THRESHOLD = 0.5
LEARNING_RATE = 0.1

PERSONALIZATION_LEVELS = [
    0.0,
    0.25,
    0.50,
    0.75,
    1.0
]

RUNS_PER_CONDITION = 30

CLUSTER_GAP = 0.15

# Random seed
BASE_SEEDS = list(range(101, 131))


DATA_DIR = "data"
OUTPUT_DIR = "outputs"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HÀM GHI FILE
# ============================================================

def save_config():

    config = {
        "project_name":
            "AI Social Simulation Lab",

        "num_agents":
            NUM_AGENTS,

        "steps":
            STEPS,

        "threshold":
            THRESHOLD,

        "learning_rate":
            LEARNING_RATE,

        "personalization_levels":
            PERSONALIZATION_LEVELS,

        "runs_per_condition":
            RUNS_PER_CONDITION,

        "cluster_gap":
            CLUSTER_GAP,

        "total_experiments":
            len(PERSONALIZATION_LEVELS)
            * RUNS_PER_CONDITION,

        "created_at":
            datetime.now().isoformat()
    }

    with open(
        os.path.join(DATA_DIR, "config.json"),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# CHẠY 150 THÍ NGHIỆM
# ============================================================

def run_main_experiment():

    summary_rows = []
    trajectory_rows = []
    final_opinion_rows = []

    print("\n" + "=" * 70)
    print("AI SOCIAL SIMULATION LAB")
    print("THỰC NGHIỆM CHÍNH")
    print("=" * 70)

    print(
        f"Số Agent: {NUM_AGENTS}"
    )

    print(
        f"Số vòng: {STEPS}"
    )

    print(
        f"Số mức cá nhân hóa: "
        f"{len(PERSONALIZATION_LEVELS)}"
    )

    print(
        f"Số lần/mức: "
        f"{RUNS_PER_CONDITION}"
    )

    print(
        f"Tổng số thí nghiệm: "
        f"{len(PERSONALIZATION_LEVELS) * RUNS_PER_CONDITION}"
    )

    experiment_number = 0

    for personalization in PERSONALIZATION_LEVELS:

        for run_index in range(
            RUNS_PER_CONDITION
        ):

            experiment_number += 1

            seed = BASE_SEEDS[run_index]

            print(
                f"[{experiment_number:03d}/150] "
                f"Personalization="
                f"{personalization:.2f} | "
                f"Run={run_index + 1:02d} | "
                f"Seed={seed}"
            )

            history = simulate(
                num_agents=NUM_AGENTS,
                steps=STEPS,
                personalization=personalization,
                threshold=THRESHOLD,
                learning_rate=LEARNING_RATE,
                seed=seed
            )

            # ------------------------------------------------
            # ID
            # ------------------------------------------------

            run_id = run_index + 1

            experiment_id = (
                f"P{int(personalization * 100):03d}"
                f"_R{run_id:02d}"
            )

            # ------------------------------------------------
            # METRICS CUỐI
            # ------------------------------------------------

            final_metrics = (
                calculate_final_metrics(
                    history,
                    CLUSTER_GAP
                )
            )

            summary_rows.append({

                "experiment_id":
                    experiment_id,

                "run_id":
                    run_id,

                "personalization":
                    personalization,

                "personalization_percent":
                    personalization * 100,

                "seed":
                    seed,

                "num_agents":
                    NUM_AGENTS,

                "steps":
                    STEPS,

                "threshold":
                    THRESHOLD,

                "learning_rate":
                    LEARNING_RATE,

                "cluster_gap":
                    CLUSTER_GAP,

                **final_metrics

            })

            # ------------------------------------------------
            # TRAJECTORY
            # ------------------------------------------------

            trajectory_metrics = (
                calculate_trajectory_metrics(
                    history,
                    CLUSTER_GAP
                )
            )

            for row in trajectory_metrics:

                row["experiment_id"] = (
                    experiment_id
                )

                row["run_id"] = run_id

                row["personalization"] = (
                    personalization
                )

                row["seed"] = seed

                trajectory_rows.append(row)

            # ------------------------------------------------
            # FINAL OPINIONS
            # ------------------------------------------------

            final_opinions = history[-1]

            for agent_id, opinion in enumerate(
                final_opinions
            ):

                final_opinion_rows.append({

                    "experiment_id":
                        experiment_id,

                    "run_id":
                        run_id,

                    "personalization":
                        personalization,

                    "seed":
                        seed,

                    "agent_id":
                        agent_id,

                    "final_opinion":
                        opinion

                })

    summary_df = pd.DataFrame(
        summary_rows
    )

    trajectory_df = pd.DataFrame(
        trajectory_rows
    )

    final_opinion_df = pd.DataFrame(
        final_opinion_rows
    )

    # --------------------------------------------------------
    # LƯU CSV
    # --------------------------------------------------------

    summary_df.to_csv(
        os.path.join(
            DATA_DIR,
            "experiment_summary.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    trajectory_df.to_csv(
        os.path.join(
            DATA_DIR,
            "trajectories.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    final_opinion_df.to_csv(
        os.path.join(
            DATA_DIR,
            "final_opinions.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    print("\nĐã hoàn thành 150 thí nghiệm.")

    return (
        summary_df,
        trajectory_df,
        final_opinion_df
    )


# ============================================================
# PHÂN TÍCH THỐNG KÊ
# ============================================================

def run_statistical_analysis(
    summary_df
):

    # --------------------------------------------------------
    # DESCRIPTIVE
    # --------------------------------------------------------

    desc = descriptive_statistics(
        summary_df,
        "final_polarization",
        "personalization"
    )

    desc[
        "personalization_percent"
    ] = desc["personalization"] * 100

    desc.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "01_mean_sd.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # ANOVA
    # --------------------------------------------------------

    anova = one_way_anova(
        summary_df,
        "final_polarization",
        "personalization"
    )

    eta = eta_squared(
        summary_df,
        "final_polarization",
        "personalization"
    )

    anova_df = pd.DataFrame([{

        "F_statistic":
            anova["F_statistic"],

        "p_value":
            anova["p_value"],

        "eta_squared":
            eta,

        "interpretation":
            significance_text(
                anova["p_value"]
            )

    }])

    anova_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "02_anova.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    trend = linear_trend_test(
        summary_df,
        "final_polarization",
        "personalization"
    )

    trend_df = pd.DataFrame([trend])

    trend_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "03_trend.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # PAIRED TEST
    # --------------------------------------------------------

    paired = paired_comparison(
        summary_df,
        "final_polarization",
        0.0,
        1.0
    )

    paired_df = pd.DataFrame([{

        **paired,

        "interpretation":
            significance_text(
                paired["p_value"]
            )

    }])

    paired_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "04_paired_test.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    return (
        desc,
        anova_df,
        trend_df,
        paired_df
    )


# ============================================================
# PHÂN TÍCH ĐỘ NHẠY
# ============================================================

def run_sensitivity_analysis():

    thresholds = [
        0.3,
        0.4,
        0.5,
        0.6,
        0.7
    ]

    learning_rates = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25
    ]

    personalization = 0.5

    rows = []

    print("\n" + "=" * 70)
    print("PHÂN TÍCH ĐỘ NHẠY")
    print("=" * 70)

    for threshold in thresholds:

        for lr in learning_rates:

            values = []

            for run in range(10):

                seed = 1000 + run

                history = simulate(
                    num_agents=NUM_AGENTS,
                    steps=STEPS,
                    personalization=personalization,
                    threshold=threshold,
                    learning_rate=lr,
                    seed=seed
                )

                metrics = calculate_final_metrics(
                    history,
                    CLUSTER_GAP
                )

                values.append(
                    metrics[
                        "final_polarization"
                    ]
                )

            rows.append({

                "threshold":
                    threshold,

                "learning_rate":
                    lr,

                "mean_polarization":
                    np.mean(values),

                "sd_polarization":
                    np.std(
                        values,
                        ddof=1
                    ),

                "n":
                    len(values)

            })

    sensitivity_df = pd.DataFrame(rows)

    sensitivity_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "05_sensitivity.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    return sensitivity_df


# ============================================================
# BIỂU ĐỒ
# ============================================================

def create_figures(
    summary_df,
    trajectory_df,
    final_opinion_df,
    sensitivity_df
):

    sns.set_theme(
        style="whitegrid"
    )

    # ========================================================
    # 1. MEAN ± SD
    # ========================================================

    grouped = (
        summary_df
        .groupby("personalization")
        ["final_polarization"]
        .agg(
            ["mean", "std"]
        )
        .reset_index()
    )

    grouped[
        "personalization_percent"
    ] = grouped["personalization"] * 100

    plt.figure(figsize=(9, 5))

    plt.errorbar(
        grouped[
            "personalization_percent"
        ],
        grouped["mean"],
        yerr=grouped["std"],
        marker="o",
        linewidth=2,
        capsize=5
    )

    plt.xlabel(
        "Mức độ cá nhân hóa (%)"
    )

    plt.ylabel(
        "Chỉ số phân tán quan điểm"
    )

    plt.title(
        "Mean ± SD của chỉ số phân tán quan điểm"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "06_mean_sd.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # 2. BOXPLOT
    # ========================================================

    plt.figure(figsize=(9, 5))

    sns.boxplot(
        data=summary_df,
        x="personalization_percent",
        y="final_polarization"
    )

    plt.xlabel(
        "Mức độ cá nhân hóa (%)"
    )

    plt.ylabel(
        "Chỉ số phân tán quan điểm"
    )

    plt.title(
        "Phân bố chỉ số phân tán theo mức cá nhân hóa"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "07_boxplot.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # 3. TRAJECTORY
    # ========================================================

    trajectory_mean = (
        trajectory_df
        .groupby(
            [
                "personalization",
                "step"
            ]
        )["polarization"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=trajectory_mean,
        x="step",
        y="polarization",
        hue="personalization",
        marker="o"
    )

    plt.xlabel(
        "Vòng mô phỏng"
    )

    plt.ylabel(
        "Chỉ số phân tán quan điểm"
    )

    plt.title(
        "Thay đổi phân tán quan điểm theo thời gian"
    )

    plt.legend(
        title="Cá nhân hóa"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "08_trajectory.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # 4. FINAL DISTRIBUTION
    # ========================================================

    plt.figure(figsize=(10, 6))

    sns.kdeplot(
        data=final_opinion_df,
        x="final_opinion",
        hue="personalization",
        fill=False
    )

    plt.xlabel(
        "Quan điểm cuối cùng"
    )

    plt.ylabel(
        "Mật độ"
    )

    plt.title(
        "Phân bố quan điểm cuối cùng"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "09_final_distribution.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # 5. SENSITIVITY HEATMAP
    # ========================================================

    pivot = sensitivity_df.pivot(
        index="threshold",
        columns="learning_rate",
        values="mean_polarization"
    )

    plt.figure(figsize=(9, 6))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd"
    )

    plt.xlabel(
        "Tốc độ cập nhật"
    )

    plt.ylabel(
        "Ngưỡng tương tác"
    )

    plt.title(
        "Heatmap phân tích độ nhạy"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "10_sensitivity_heatmap.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # 6. CLUSTERS
    # ========================================================

    cluster_mean = (
        summary_df
        .groupby(
            "personalization"
        )["final_clusters"]
        .agg(
            ["mean", "std"]
        )
        .reset_index()
    )

    cluster_mean[
        "personalization_percent"
    ] = cluster_mean[
        "personalization"
    ] * 100

    plt.figure(figsize=(9, 5))

    plt.errorbar(
        cluster_mean[
            "personalization_percent"
        ],
        cluster_mean["mean"],
        yerr=cluster_mean["std"],
        marker="o",
        capsize=5,
        linewidth=2
    )

    plt.xlabel(
        "Mức độ cá nhân hóa (%)"
    )

    plt.ylabel(
        "Số cụm quan điểm"
    )

    plt.title(
        "Số cụm quan điểm theo mức cá nhân hóa"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "11_clusters.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# TẠO BÁO CÁO TỰ ĐỘNG
# ============================================================

def create_text_report(
    desc,
    anova_df,
    trend_df,
    paired_df,
    sensitivity_df
):

    report_path = os.path.join(
        OUTPUT_DIR,
        "12_summary_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "BÁO CÁO TỰ ĐỘNG – AI SOCIAL SIMULATION LAB\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            "1. THIẾT KẾ THỰC NGHIỆM\n"
        )

        f.write(
            f"- Số Agent: {NUM_AGENTS}\n"
        )

        f.write(
            f"- Số vòng: {STEPS}\n"
        )

        f.write(
            f"- Mức cá nhân hóa: "
            f"{[int(x*100) for x in PERSONALIZATION_LEVELS]}%\n"
        )

        f.write(
            f"- Số lần chạy mỗi mức: "
            f"{RUNS_PER_CONDITION}\n"
        )

        f.write(
            f"- Tổng số thí nghiệm: "
            f"{len(PERSONALIZATION_LEVELS) * RUNS_PER_CONDITION}\n\n"
        )

        # ----------------------------------------------------
        # DESCRIPTIVE
        # ----------------------------------------------------

        f.write(
            "2. THỐNG KÊ MÔ TẢ\n"
        )

        f.write(
            desc.to_string(
                index=False
            )
        )

        f.write("\n\n")

        # ----------------------------------------------------
        # ANOVA
        # ----------------------------------------------------

        f.write(
            "3. ANOVA\n"
        )

        f.write(
            anova_df.to_string(
                index=False
            )
        )

        f.write("\n\n")

        # ----------------------------------------------------
        # TREND
        # ----------------------------------------------------

        f.write(
            "4. PHÂN TÍCH XU HƯỚNG\n"
        )

        f.write(
            trend_df.to_string(
                index=False
            )
        )

        f.write("\n\n")

        # ----------------------------------------------------
        # PAIRED
        # ----------------------------------------------------

        f.write(
            "5. SO SÁNH 0% VÀ 100%\n"
        )

        f.write(
            paired_df.to_string(
                index=False
            )
        )

        f.write("\n\n")

        # ----------------------------------------------------
        # SENSITIVITY
        # ----------------------------------------------------

        f.write(
            "6. PHÂN TÍCH ĐỘ NHẠY\n"
        )

        f.write(
            sensitivity_df.to_string(
                index=False
            )
        )

        f.write("\n\n")

        f.write(
            "7. LƯU Ý KHOA HỌC\n"
        )

        f.write(
            "- Kết quả chỉ áp dụng cho mô hình mô phỏng.\n"
        )

        f.write(
            "- Không được diễn giải kết quả như bằng chứng trực tiếp về xã hội thực tế.\n"
        )

        f.write(
            "- Chỉ số phân cực trong nghiên cứu là chỉ số vận hành của mô hình.\n"
        )

        f.write(
            "- Không sử dụng số liệu này để khẳng định quan hệ nhân quả ngoài mô hình.\n"
        )

    print(
        f"\nĐã tạo báo cáo: {report_path}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\nBẮT ĐẦU NGHIÊN CỨU...\n")

    save_config()

    (
        summary_df,
        trajectory_df,
        final_opinion_df
    ) = run_main_experiment()

    (
        desc,
        anova_df,
        trend_df,
        paired_df
    ) = run_statistical_analysis(
        summary_df
    )

    sensitivity_df = (
        run_sensitivity_analysis()
    )

    create_figures(
        summary_df,
        trajectory_df,
        final_opinion_df,
        sensitivity_df
    )

    create_text_report(
        desc,
        anova_df,
        trend_df,
        paired_df,
        sensitivity_df
    )

    print("\n" + "=" * 70)

    print(
        "HOÀN THÀNH TOÀN BỘ THỰC NGHIỆM"
    )

    print("=" * 70)

    print(
        "\nDữ liệu: ./data/"
    )

    print(
        "Biểu đồ + bảng phân tích: ./outputs/"
    )
