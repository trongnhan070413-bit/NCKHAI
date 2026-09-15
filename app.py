Đây là giao diện để cô và học sinh chạy thử, xem kết quả và trình diễn sản phẩm.
import os

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from simulation import simulate
from metrics import calculate_final_metrics


st.set_page_config(
    page_title="AI Social Simulation Lab",
    layout="wide"
)

st.title(
    "🧪 AI Social Simulation Lab"
)

st.subheader(
    "Nghiên cứu ảnh hưởng của mức độ cá nhân hóa "
    "đến sự phân bố quan điểm trong xã hội giả lập"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Cấu hình mô phỏng"
)

num_agents = st.sidebar.slider(
    "Số Agent",
    10,
    500,
    100,
    step=10
)

steps = st.sidebar.slider(
    "Số vòng",
    10,
    200,
    50,
    step=10
)

personalization = st.sidebar.slider(
    "Mức cá nhân hóa",
    0.0,
    1.0,
    0.5,
    step=0.05
)

threshold = st.sidebar.slider(
    "Ngưỡng tương tác",
    0.1,
    1.0,
    0.5,
    step=0.05
)

learning_rate = st.sidebar.slider(
    "Tốc độ cập nhật",
    0.01,
    0.5,
    0.1,
    step=0.01
)

seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=999999,
    value=42
)


# ============================================================
# CHẠY MÔ PHỎNG
# ============================================================

if st.button(
    "▶️ Chạy mô phỏng"
):

    history = simulate(
        num_agents=num_agents,
        steps=steps,
        personalization=personalization,
        threshold=threshold,
        learning_rate=learning_rate,
        seed=seed
    )

    final_metrics = (
        calculate_final_metrics(
            history
        )
    )

    st.success(
        "Mô phỏng hoàn thành!"
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Quan điểm trung bình",
        f"{final_metrics['final_mean_opinion']:.3f}"
    )

    col2.metric(
        "Phân tán quan điểm",
        f"{final_metrics['final_polarization']:.3f}"
    )

    col3.metric(
        "Số cụm",
        final_metrics["final_clusters"]
    )

    col4.metric(
        "Thay đổi trung bình",
        f"{final_metrics['mean_absolute_change']:.3f}"
    )

    # --------------------------------------------------------
    # TRAJECTORY
    # --------------------------------------------------------

    st.write(
        "### 📈 Diễn biến quan điểm"
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    history_array = history

    for i in range(
        min(num_agents, 50)
    ):

        ax.plot(
            history_array[:, i],
            alpha=0.35
        )

    ax.set_xlabel(
        "Vòng mô phỏng"
    )

    ax.set_ylabel(
        "Quan điểm"
    )

    ax.set_title(
        "Sự thay đổi quan điểm của các Agent"
    )

    st.pyplot(fig)

    # --------------------------------------------------------
    # FINAL DISTRIBUTION
    # --------------------------------------------------------

    st.write(
        "### 📊 Phân bố quan điểm cuối"
    )

    fig2, ax2 = plt.subplots(
        figsize=(10, 4)
    )

    ax2.hist(
        history[-1],
        bins=20,
        edgecolor="black",
        alpha=0.75
    )

    ax2.set_xlabel(
        "Quan điểm"
    )

    ax2.set_ylabel(
        "Số Agent"
    )

    st.pyplot(fig2)


# ============================================================
# KẾT QUẢ THỰC NGHIỆM
# ============================================================

st.divider()

st.header(
    "🧪 Kết quả thí nghiệm chính"
)

summary_path = (
    "data/experiment_summary.csv"
)

if os.path.exists(summary_path):

    df = pd.read_csv(
        summary_path
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.write(
        "### Mean ± SD"
    )

    grouped = (
        df.groupby(
            "personalization"
        )["final_polarization"]
        .agg(["mean", "std"])
        .reset_index()
    )

    grouped[
        "personalization"
    ] *= 100

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.errorbar(
        grouped["personalization"],
        grouped["mean"],
        yerr=grouped["std"],
        marker="o",
        capsize=5,
        linewidth=2
    )

    ax.set_xlabel(
        "Mức cá nhân hóa (%)"
    )

    ax.set_ylabel(
        "Chỉ số phân tán quan điểm"
    )

    ax.set_title(
        "Mean ± SD"
    )

    st.pyplot(fig)

else:

    st.info(
        "Chưa có dữ liệu. "
        "Hãy chạy experiment.py trước."
    )
