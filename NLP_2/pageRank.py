import numpy as np

def pagerank(similarity_matrix, d=0.85, max_iter=100, tol=1e-6):
    """
    Tính toán điểm PageRank cho các câu trong văn bản.

    :param similarity_matrix: Ma trận độ tương đồng (công thức cosine) giữa các câu
    :param d: Hệ số giảm (thường là 0.85)
    :param max_iter: Số vòng lặp tối đa
    :param tol: Mức độ sai số để dừng vòng lặp (khi sự thay đổi giữa các vòng lặp nhỏ hơn tol)
    :return: Vector điểm PageRank cho mỗi câu
    """
    N = similarity_matrix.shape[0]  # Số lượng câu
    pr = np.ones(N) / N  # Khởi tạo điểm PageRank ban đầu, mỗi câu có điểm bằng nhau
    matrix = similarity_matrix / similarity_matrix.sum(axis=1, keepdims=True)  # Ma trận chuyển tiếp (normalized)

    # Thực hiện tính toán PageRank qua nhiều vòng lặp
    for _ in range(max_iter):
        new_pr = (1 - d) / N + d * matrix.T @ pr  # Công thức PageRank
        # Kiểm tra sự thay đổi giữa các vòng lặp
        if np.linalg.norm(new_pr - pr, 1) < tol:
            break
        pr = new_pr

    return pr

# Ví dụ sử dụng
similarity_matrix = np.array([
    [1.0, 0.5, 0.2],  # Câu 1 liên kết với câu 2, 3
    [0.5, 1.0, 0.3],  # Câu 2 liên kết với câu 1, 3
    [0.2, 0.3, 1.0]   # Câu 3 liên kết với câu 1, 2
])

pagerank_scores = pagerank(similarity_matrix)
print("PageRank Scores:", pagerank_scores)
