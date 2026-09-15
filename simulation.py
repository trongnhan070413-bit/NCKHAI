import numpy as np


def simulate(
    num_agents=100,
    steps=50,
    personalization=0.5,
    threshold=0.5,
    learning_rate=0.1,
    seed=42
):
    """
    Chạy một mô phỏng xã hội Agent-based.

    Parameters
    ----------
    num_agents : int
        Số lượng Agent.

    steps : int
        Số vòng mô phỏng.

    personalization : float
        Mức độ cá nhân hóa từ 0 đến 1.

    threshold : float
        Khoảng cách quan điểm tối đa để hai Agent có thể ảnh hưởng nhau.

    learning_rate : float
        Mức độ thay đổi quan điểm sau tương tác.

    seed : int
        Random seed để tái lập kết quả.

    Returns
    -------
    history : np.ndarray
        Ma trận kích thước (steps + 1, num_agents).
    """

    rng = np.random.default_rng(seed)

    # Khởi tạo quan điểm
    opinions = rng.uniform(-1.0, 1.0, num_agents)

    history = [opinions.copy()]

    for _ in range(steps):

        new_opinions = opinions.copy()

        for i in range(num_agents):

            # Không cho Agent tự tương tác với chính mình
            candidates = np.array(
                [j for j in range(num_agents) if j != i]
            )

            if len(candidates) == 0:
                continue

            # Quyết định có sử dụng cá nhân hóa hay không
            use_personalization = rng.random() < personalization

            if use_personalization:

                # Tìm các Agent có quan điểm tương đồng
                similar_mask = (
                    np.abs(opinions[candidates] - opinions[i])
                    < threshold
                )

                similar_candidates = candidates[similar_mask]

                if len(similar_candidates) > 0:
                    j = rng.choice(similar_candidates)
                else:
                    j = rng.choice(candidates)

            else:
                # Không cá nhân hóa
                j = rng.choice(candidates)

            # Khoảng cách quan điểm
            diff = opinions[j] - opinions[i]

            # Chỉ cập nhật nếu nằm trong vùng ảnh hưởng
            if abs(diff) < threshold:

                new_opinions[i] += learning_rate * diff

        # Giới hạn quan điểm
        opinions = np.clip(new_opinions, -1.0, 1.0)

        history.append(opinions.copy())

    return np.array(history)
________________________________________
