import numpy as np


def mean_opinion(opinions):
    """Quan điểm trung bình."""
    return float(np.mean(opinions))


def polarization_index(opinions):
    """
    Chỉ số phân tán quan điểm.

    P = mean(|opinion_i - mean_opinion|)
    """

    mean = np.mean(opinions)

    return float(
        np.mean(np.abs(opinions - mean))
    )


def count_clusters(opinions, gap_threshold=0.15):
    """
    Đếm số cụm quan điểm.

    Các quan điểm được sắp xếp.
    Nếu khoảng cách giữa hai quan điểm liên tiếp
    lớn hơn gap_threshold thì tạo một cụm mới.
    """

    sorted_opinions = np.sort(opinions)

    if len(sorted_opinions) == 0:
        return 0

    gaps = np.diff(sorted_opinions)

    clusters = 1 + np.sum(
        gaps > gap_threshold
    )

    return int(clusters)


def mean_absolute_change(history):
    """
    Mức thay đổi quan điểm trung bình
    qua toàn bộ quá trình.
    """

    changes = np.abs(
        np.diff(history, axis=0)
    )

    return float(np.mean(changes))


def calculate_final_metrics(
    history,
    cluster_gap=0.15
):
    """
    Tính các chỉ số cuối cùng.
    """

    final_opinions = history[-1]

    return {
        "final_mean_opinion":
            mean_opinion(final_opinions),

        "final_polarization":
            polarization_index(final_opinions),

        "final_clusters":
            count_clusters(
                final_opinions,
                cluster_gap
            ),

        "mean_absolute_change":
            mean_absolute_change(history)
    }


def calculate_trajectory_metrics(
    history,
    cluster_gap=0.15
):
    """
    Tính các chỉ số ở từng vòng.
    """

    rows = []

    for step, opinions in enumerate(history):

        rows.append({
            "step": step,

            "mean_opinion":
                mean_opinion(opinions),

            "polarization":
                polarization_index(opinions),

            "clusters":
                count_clusters(
                    opinions,
                    cluster_gap
                )
        })

    return rows
________________________________________
