import os
import re
import html
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import spacy
from collections import Counter
import math

# Existing imports for stopwords
from nltk.corpus import stopwords
from spacy.lang.am.examples import sentences

# Tải danh sách stop words tiếng Anh
stop_words = set(stopwords.words('english'))


# Existing preprocessing function
def preprocess_text(text):
    """
    Tiền xử lý văn bản:
    - Chuyển thành chữ thường
    - Loại bỏ các ký tự đặc biệt
    - Loại bỏ stop words
    """
    text = text.lower()  # Chuyển thành chữ thường
    text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ ký tự đặc biệt
    words = text.split()  # Tách thành các từ
    words = [word for word in words if word not in stop_words]  # Loại bỏ stop words
    return words  # Trả về danh sách từ thay vì chuỗi


# Custom TF-IDF calculation functions
def calculate_term_frequency(document):
    """
    Tính Term Frequency (TF) cho một văn bản
    """
    word_counts = Counter(document)
    total_words = len(document)

    # Tính TF: số lần xuất hiện / tổng số từ
    tf = {word: count / total_words for word, count in word_counts.items()}
    return tf


def calculate_idf(documents):
    """
    Tính Inverse Document Frequency (IDF) cho toàn bộ tập văn bản
    """
    # Đếm số lượng văn bản chứa từng từ
    word_doc_count = {}
    total_docs = len(documents)

    for doc in documents:
        unique_words = set(doc)
        for word in unique_words:
            word_doc_count[word] = word_doc_count.get(word, 0) + 1

    # Tính IDF: log(tổng số văn bản / số văn bản chứa từ)
    idf = {word: math.log(total_docs / count) for word, count in word_doc_count.items()}
    return idf


def calculate_tfidf(documents):
    """
    Tính TF-IDF cho toàn bộ tập văn bản
    """
    # Tính IDF trước
    idf = calculate_idf(documents)

    # Tính TF-IDF cho từng văn bản
    tfidf_documents = []
    for doc in documents:
        # Tính TF của văn bản
        tf = calculate_term_frequency(doc)

        # Tính TF-IDF
        tfidf = {word: tf.get(word, 0) * idf.get(word, 0) for word in set(doc)}
        tfidf_documents.append(tfidf)

    return tfidf_documents


def create_tfidf_vectors(documents):
    """
    Tạo vector TF-IDF cho các văn bản
    """
    # Lấy toàn bộ từ vựng
    all_words = set(word for doc in documents for word in doc)

    # Tính TF-IDF
    tfidf_docs = calculate_tfidf(documents)

    # Chuyển đổi sang vector
    vectors = []
    for tfidf_doc in tfidf_docs:
        vector = [tfidf_doc.get(word, 0) for word in sorted(all_words)]
        vectors.append(vector)

    return np.array(vectors), sorted(all_words)


def cosine_similarity_custom(v1, v2):
    """
    Tính toán độ tương đồng cosine giữa hai vector
    """
    # Tính tích vô hướng
    dot_product = np.dot(v1, v2)

    # Tính độ dài vector
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Tránh chia cho 0
    if norm_v1 * norm_v2 == 0:
        return 0

    return dot_product / (norm_v1 * norm_v2)


# Hàm tách câu từ văn bản
nlp = spacy.load("en_core_web_sm")  # Tải mô hình spaCy


def split_sentences(text):
    """
    Hàm tách câu sử dụng spaCy
    """
    doc = nlp(text)  # Phân tích văn bản # Tách câu theo dấu câu - Thư viện Spacy
    # sentences = re.split(r'(?<=[.!?]) +', text)

    sentences = [sent.text for sent in doc.sents]  # Lấy câu từ đối tượng doc

    return sentences


# 2. Các hàm tính toán và phân tích

def pagerank(graph, max_iter=100, d=0.85, tol=1e-6):
    """
    Tính PageRank thuần túy.
    :param graph: Ma trận độ tương đồng (adjacency matrix)
    :param max_iter: Số vòng lặp tối đa
    :param d: Hệ số giảm (damping factor), thường là 0.85
    :param tol: Sai số cho việc hội tụ
    :return: Điểm PageRank cho mỗi câu trong văn bản
    """
    n = len(graph)  # Số lượng câu
    matrix = np.array(graph)
    row_sums = matrix.sum(axis=1)  # Tính tổng mỗi hàng
    matrix = matrix / row_sums[:, np.newaxis]  # Chuẩn hóa theo cột

    # Khởi tạo giá trị PageRank ban đầu (mỗi câu có giá trị bằng nhau)
    pagerank_values = np.ones(n) / n
    teleport = np.ones(n) / n  # Tạo ma trận nhảy (teleportation) với giá trị đều

    for _ in range(max_iter):
        new_pagerank_values = (1 - d) * teleport + d * matrix.T @ pagerank_values
        # Kiểm tra sự hội tụ
        if np.linalg.norm(new_pagerank_values - pagerank_values, 1) < tol:
            break
        pagerank_values = new_pagerank_values

    return pagerank_values


# Hàm tóm tắt văn bản sử dụng TextRank
def summarize_text_textrank_auto(text, ratio=0.2):
    """
    Hàm tóm tắt văn bản sử dụng TextRank với TF-IDF tự tính
    """
    sentences = split_sentences(text)
    preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]

    # Vector hóa câu bằng TF-IDF tự tính
    vectors, _ = create_tfidf_vectors(preprocessed_sentences)

    # Tính ma trận độ tương đồng cosine
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            similarity_matrix[i][j] = cosine_similarity_custom(vectors[i], vectors[j])

    # Tính trọng số câu (tổng TF-IDF của tất cả các từ trong câu)
    sentence_weights = []
    for i in range(vectors.shape[0]):
        total_weight = np.sum(vectors[i])  # Tổng TF-IDF của tất cả các từ trong câu
        sentence_weights.append((i, total_weight))  # Lưu index câu và trọng số của câu

    # Sắp xếp câu theo trọng số giảm dần
    sentence_weights.sort(key=lambda x: x[1], reverse=True)

    # Sắp xếp các câu theo điểm PageRank
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)

    ranked_sentences = sorted(((scores[i], sentence) for i, sentence in enumerate(sentences)), reverse=True)

    # Số lượng câu tóm tắt theo tỷ lệ
    num_sentences = max(1, int(len(sentences) * ratio))  # Đảm bảo có ít nhất 1 câu tóm tắt

    # Lấy các câu có điểm cao nhất từ PageRank
    summary = [sentence for _, sentence in ranked_sentences[:num_sentences]]

    return summary, sentence_weights, similarity_matrix, graph


def read_and_process_html(file_path):
    """
    Đọc và xử lý văn bản từ file HTML.
    Loại bỏ các thẻ HTML và giải mã các ký tự đặc biệt.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    clean_text = re.sub(r'<[^>]*>', '', html_content)  # Loại bỏ thẻ HTML
    clean_text = html.unescape(clean_text)  # Giải mã các ký tự đặc biệt (như &amp;)
    return clean_text


def write_summary_with_matrix(output_path, summary, num_sentences, similarity_matrix, sentences, scores,
                              tfidf_docs=None, all_words=None):
    """
    Ghi tóm tắt văn bản, ma trận độ tương đồng, và giá trị TF-IDF vào file HTML.

    :param output_path: Đường dẫn file xuất HTML
    :param summary: Danh sách các câu tóm tắt
    :param num_sentences: Số lượng câu trong tóm tắt
    :param similarity_matrix: Ma trận độ tương đồng
    :param sentences: Danh sách các câu gốc
    :param scores: Điểm của các câu
    :param tfidf_docs: Danh sách các từ điển TF-IDF (tùy chọn)
    :param all_words: Danh sách tất cả các từ (tùy chọn)
    """
    with open(output_path, 'w', encoding='utf-8') as file:
        # Thêm thông tin ngày giờ xuất file
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        file.write("<html>\n<head>\n<title>Text Summary</title>\n")
        file.write('<meta charset="UTF-8">\n')
        file.write('<style>\n')
        file.write('body {font-family: Arial, sans-serif;}\n')
        file.write('table {border-collapse: collapse; width: 100%; margin-top: 20px;}\n')
        file.write('th, td {border: 1px solid #ddd; padding: 8px; text-align: center;}\n')
        file.write('th {background-color: #f2f2f2;}\n')
        file.write('tr:nth-child(even) {background-color: #f9f9f9;}\n')
        file.write('tr:hover {background-color: #f1f1f1;}\n')
        file.write('</style>\n</head>\n<body>\n')

        file.write(f"<h1>Text Summary (generated on {now})</h1>\n")
        file.write(
            f"<h3>Number of sentences in the original text: {len(sentences)}, Number of sentences in the summary: {num_sentences}</h3>\n")

        # Ghi phần tóm tắt
        file.write("<h2>Summary:</h2>\n")
        for idx, sentence in enumerate(summary):
            weight = scores[idx]  # Lấy trọng số của câu
            file.write(f"<p>{sentence} </p>")  # Ghi câu vào file HTML

        # Ghi ma trận độ tương đồng
        file.write("<h2>Similarity Matrix:</h2>\n<table>\n<tr><th></th>\n")
        for i in range(len(sentences)):
            file.write(f"<th>Sentence {i + 1}</th>")
        file.write("</tr>\n")

        for i, row in enumerate(similarity_matrix):
            file.write(f"<tr><th>Sentence {i + 1}</th>")
            for value in row:
                file.write(f"<td>{value:.4f}</td>")  # Định dạng giá trị độ tương đồng
            file.write("</tr>\n")
        file.write("</table>\n")

        # Ghi TF-IDF nếu được cung cấp
        if tfidf_docs is not None and all_words is not None:
            file.write("<h2>TF-IDF Values:</h2>\n<table>\n<tr><th>Sentence</th>\n")

            # Tiêu đề cột là các từ
            for word in all_words:
                file.write(f"<th>{word}</th>")
            file.write("</tr>\n")

            # Ghi giá trị TF-IDF cho từng câu
            for i, tfidf_doc in enumerate(tfidf_docs):
                file.write(f"<tr><th>Sentence {i + 1}</th>")
                for word in all_words:
                    value = tfidf_doc.get(word, 0)
                    file.write(f"<td>{value:.4f}</td>")
                file.write("</tr>\n")
            file.write("</table>\n")

        file.write("</body>\n</html>")


def draw_and_show_graph(graph):
    """
    Vẽ đồ thị độ tương đồng giữa các câu và hiển thị trực tiếp.
    """
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=12, font_weight='bold')
    plt.title("Đồ thị độ tương đồng giữa các câu")
    plt.show()  # Hiển thị đồ thị trong cửa sổ đồ họa


def write_summary(output_path, summary, num_sentences):
    """
    Ghi tóm tắt văn bản vào file HTML mà không có trọng số.
    """
    with open(output_path, 'w', encoding='utf-8') as file:
        # Thêm thông tin ngày giờ xuất file
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        file.write("<html>\n<head>\n<title>Text Summary</title>\n")
        file.write('<meta charset="UTF-8">\n')
        file.write('<style>\n')
        file.write('body {font-family: Arial, sans-serif;}\n')
        file.write('table {border-collapse: collapse; width: 100%; margin-top: 20px;}\n')
        file.write('th, td {border: 1px solid #ddd; padding: 8px; text-align: center;}\n')
        file.write('th {background-color: #f2f2f2;}\n')
        file.write('tr:nth-child(even) {background-color: #f9f9f9;}\n')
        file.write('tr:hover {background-color: #f1f1f1;}\n')
        file.write('</style>\n</head>\n<body>\n')

        file.write(f"<h1>Text Summary (generated on {now})</h1>\n")
        file.write(
            f"<h3>Number of sentences in the original text: {len(sentences)}, Number of sentences in the summary: {num_sentences}</h3>\n")

        # Ghi phần tóm tắt
        file.write("<h2>Summary:</h2>\n")
        for idx, sentence in enumerate(summary):
            file.write(f"<p>{sentence} </p>")  # Ghi câu vào file HTML

        file.write("</body>\n</html>")


# 4. Hàm chính
def main(input_file, output_file, ratio=0.2):
    text = read_and_process_html(input_file)
    sentences = split_sentences(text)
    preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]

    # Tạo vector TF-IDF
    vectors, all_words = create_tfidf_vectors(preprocessed_sentences)
    tfidf_docs = calculate_tfidf(preprocessed_sentences)

    summary, scores, similarity_matrix, graph = summarize_text_textrank_auto(text, ratio=ratio)

    draw_and_show_graph(graph)

    # Truyền thêm tham số TF-IDF
    write_summary_with_matrix(output_file, summary, len(summary), similarity_matrix, sentences, scores, tfidf_docs,
                              all_words)
    print(f"Bản tóm tắt, ma trận độ tương đồng và TF-IDF đã được ghi vào file: {output_file}")


# 5. Hàm xử lý tất cả các file trong thư mục
def process_all_files(input_folder, output_folder, ratio=0.2):
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".html"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name.replace(".html", "_summary.html"))
            main(input_path, output_path, ratio)


# Chạy chương trình
if __name__ == "__main__":
    input_folder = "C:/Users/PC/PycharmProjects/bot2vec/NLP_2/input"
    output_folder = "C:/Users/PC/PycharmProjects/bot2vec/NLP_2/output"
    process_all_files(input_folder, output_folder)
