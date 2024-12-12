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
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import f1_score, precision_score, recall_score
from spacy.lang.am.examples import sentences
from statsmodels.graphics.tukeyplot import results

# Tải danh sách stop words tiếng Anh
stop_words = set(stopwords.words('english'))

# Hàm tách câu từ văn bản
nlp = spacy.load("en_core_web_sm")  # Tải mô hình spaCy


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


def split_sentences(text):
    """
    Hàm tách câu sử dụng spaCy
    """
    doc = nlp(text)  # Phân tích văn bản # Tách câu theo dấu câu - Thư viện Spacy
    # sentences = re.split(r'(?<=[.!?]) +', text) #tách câu theo cơ bản

    sentences = [sent.text for sent in doc.sents]  # Lấy câu từ đối tượng doc

    return sentences


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


# 2. Các hàm tính toán và phân tích
"""
Dưới đây là tóm tắt các thông số của hàm `pagerank`:

1. **graph**: Đồ thị đầu vào (có thể là ma trận kề hoặc cấu trúc liên kết giữa các nút trong đồ thị).
2. **max_iter=100**: Số lần lặp tối đa (mặc định là 100 vòng lặp) để thuật toán thực hiện tính toán PageRank.
3. **d=0.85**: Hệ số giảm dần (damping factor) trong phạm vi [0, 1], mặc định là 0.85, kiểm soát mức độ ảnh hưởng của các liên kết.
4. **tol=1e-6**: Ngưỡng sai số, khi sự thay đổi giữa các vòng lặp nhỏ hơn giá trị này, thuật toán sẽ dừng lại (mặc định là \(1 times 10^{-6}\)).

Các tham số này giúp điều chỉnh cách thức thuật toán PageRank tính toán và dừng khi đạt độ chính xác cần thiết.

"""


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
    # xuất điểm pagerank
    pagerank_values = pagerank(similarity_matrix)

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

            # Ghi điểm PageRank
            file.write("<h2>PageRank Scores:</h2>\n<table>\n<tr><th>Sentence</th><th>PageRank</th></tr>\n")
            for i, score in enumerate(pagerank_values):
                file.write(f"<tr><td>Sentence {i + 1}</td><td>{score:.4f}</td></tr>\n")
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


from sklearn.metrics import f1_score, precision_score, recall_score


def calculate_f1_score(ground_truth_summary, generated_summary):
    """
    Tính F1-score giữa bản tóm tắt gốc và bản tóm tắt được tạo ra
    """
    # Tiền xử lý để so sánh
    ground_truth_processed = [preprocess_text(' '.join(ground_truth_summary))]
    generated_processed = [preprocess_text(' '.join(generated_summary))]

    print(f"Ground Truth Processed: {ground_truth_processed}")
    print(f"Generated Processed: {generated_processed}")

    # Chuyển đổi thành vector nhị phân
    all_words = list(set(ground_truth_processed[0] + generated_processed[0]))

    print(f"All Words: {all_words}")

    # Tạo vector nhị phân
    ground_truth_vector = [1 if word in ground_truth_processed[0] else 0 for word in all_words]
    generated_vector = [1 if word in generated_processed[0] else 0 for word in all_words]

    print(f"Ground Truth Vector: {ground_truth_vector}")
    print(f"Generated Vector: {generated_vector}")

    # Tính F1-score với zero_division để tránh cảnh báo
    f1 = f1_score(ground_truth_vector, generated_vector, zero_division=0)
    precision = precision_score(ground_truth_vector, generated_vector, zero_division=0)
    recall = recall_score(ground_truth_vector, generated_vector, zero_division=0)

    # Tính BLEU score
    reference = [ground_truth_processed[0]]
    candidate = generated_processed[0]
    bleu_score = sentence_bleu(reference, candidate)

    print(f"F1: {f1}, Precision: {precision}, Recall: {recall}, BLEU: {bleu_score}")

    return {
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'bleu_score': bleu_score
    }


def process_all_files_with_f1_score(input_folder, output_folder, ground_truth_folder, ratio=0.2):
    """
    Xử lý tất cả các file và tính F1-score
    """
    f1_results = {}

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".html"):
            # Xử lý file đầu vào (generated summary file)
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name.replace(".html", "_summary.html"))

            # Thực hiện tóm tắt
            text = read_and_process_html(input_path)
            summary, scores, similarity_matrix, graph = summarize_text_textrank_auto(text, ratio=ratio)

            # Ghi file tóm tắt
            write_summary(output_path, summary, len(summary))

            # Tìm file tóm tắt gốc tương ứng (ground truth file)
            ground_truth_filename = file_name.replace("_summary.html", ".html")
            ground_truth_path = os.path.join(ground_truth_folder, ground_truth_filename)

            if os.path.exists(ground_truth_path):
                # Đọc tóm tắt gốc
                ground_truth_text = read_and_process_html(ground_truth_path)
                ground_truth_sentences = split_sentences(ground_truth_text)

                # Tính F1-score
                f1_metrics = calculate_f1_score(ground_truth_sentences, summary)
                f1_results[file_name] = f1_metrics

            else:
                print(f"Ground truth file for {file_name} not found!")

    return f1_results  # Return f1_results to be used outside the function


def write_f1_scores_to_html(f1_results, output_folder):
    """
    ghi F1 scores vao HTML file
    """
    with open(os.path.join(output_folder, 'f1_scores.md.html'), 'w', encoding='utf-8') as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(
            "<html><body><h1>"
            "F1 Score Results</h1><table border='1'><tr><th>File</th><th>F1-score</th><th>Precision</th><th>Recall</th><th>BLEU Score</th></tr>"
            f"<h3>F1-score (generated on {now})</h3>\n"
            "")
        for file_name, metrics in f1_results.items():
            f.write(
                f"<tr><td>{file_name}</td><td>{metrics['f1_score']}</td><td>{metrics['precision']}</td><td>{metrics['recall']}</td><td>{metrics['bleu_score']}</td></tr>")
        f.write("</table></body></html>")


# 4. Hàm chính
def main(input_file, output_file, ratio=0.2):
    text = read_and_process_html(input_file)
    sentences = split_sentences(text)
    preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]
    # print(f"Tiền xử lý{text}")
    # Tạo vector TF-IDF
    vectors, all_words = create_tfidf_vectors(preprocessed_sentences)
    tfidf_docs = calculate_tfidf(preprocessed_sentences)

    summary, scores, similarity_matrix, graph = summarize_text_textrank_auto(text, ratio=ratio)
    # print(f"Tiền xử lý{scores}")
    draw_and_show_graph(graph)
    # tính diểm pagerank , gọi hàm
    # pagerank_values = pagerank(similarity_matrix)
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
    ground_truth_folder = "C:/Users/PC/PycharmProjects/bot2vec/NLP_2/sum_exam"

    # input_folder = os.path.join(os.path.dirname(__file__), 'input')
    # output_folder = os.path.join(os.path.dirname(__file__), 'output')
    # ground_truth_folder = os.path.join(os.path.dirname(__file__), 'sum_exam')
    if os.path.exists(ground_truth_folder):
        print("Ground truth file exists:", ground_truth_folder)
    else:
        print("Ground truth file not found:", ground_truth_folder)
    process_all_files(input_folder, output_folder)

    f1_results = process_all_files_with_f1_score(input_folder, output_folder, ground_truth_folder)

    # f1_results = process_all_files_with_f1_score(input_folder, output_folder, ground_truth_folder)

    # Write F1-score results to HTML file
    write_f1_scores_to_html(f1_results, output_folder)

    print("F1 Scores have been written to f1_scores.md.html")

    print("F1 Scores have been written to f1_scores.md.html")
