import re
import html
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime  # Để lấy thời gian hiện tại
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.am.examples import sentences
import numpy as np

# Hàm tiền xử lý văn bản
from nltk.corpus import stopwords

# Tải danh sách stop words tiếng Anh
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()  # Chuyển thành chữ thường
    text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ ký tự đặc biệt
    words = text.split()  # Tách thành các từ
    words = [word for word in words if word not in stop_words]  # Loại bỏ stop words
    return ' '.join(words)  # Kết hợp lại thành chuỗi



# Hàm tách câu từ văn bản
nlp = spacy.load("en_core_web_sm")  # Hoặc mô hình khác nếu cần


# Hàm tách câu sử dụng spaCy
def split_sentences(text):
    doc = nlp(text)  # Phân tích văn bản
    sentences = [sent.text for sent in doc.sents]  # Lấy câu từ đối tượng doc
    return sentences




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
    # Tạo ma trận khả năng chuyển
    # Sắp xếp lại ma trận (chuyển đổi từ ma trận độ tương đồng)
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
# Hàm tóm tắt văn bản sử dụng TextRank với kiểm tra chi tiết hơn
# def summarize_text_textrank_auto(text, ratio=0.2):
#     sentences = split_sentences(text)
#     preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]
#
#     # Vector hóa câu
#     vectorizer = CountVectorizer().fit_transform(preprocessed_sentences)
#     vectors = vectorizer.toarray()
#
#
#     # Tính ma trận độ tương đồng cosine
#     similarity_matrix = cosine_similarity(vectors)
#
#     # Xây dựng đồ thị
#     graph = nx.from_numpy_array(similarity_matrix)
#
#     # Tính điểm PageRank
#     scores = nx.pagerank(graph)
#
#     # Sắp xếp câu theo điểm PageRank
#     ranked_sentences = sorted(((scores[i], sentence) for i, sentence in enumerate(sentences)), reverse=True)
#
#     # Số lượng câu tóm tắt theo tỷ lệ
#     num_sentences = max(1, int(len(sentences) * ratio))  # Đảm bảo có ít nhất 1 câu tóm tắt
#
#     # Lấy các câu có điểm cao nhất từ PageRank
#     summary = [sentence for _, sentence in ranked_sentences[:num_sentences]]
#
#     return summary, scores, similarity_matrix, graph

# Hàm tóm tắt văn bản sử dụng TextRank
def summarize_text_textrank_auto(text, ratio=0.2):
    sentences = split_sentences(text)
    preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]

    # Vector hóa câu bằng TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english').fit_transform(preprocessed_sentences)
    vectors = vectorizer.toarray()

    # Tính ma trận độ tương đồng cosine
    similarity_matrix = cosine_similarity(vectors)

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




# Hàm đọc và xử lý văn bản từ HTML
def read_and_process_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    clean_text = re.sub(r'<[^>]*>', '', html_content)
    clean_text = html.unescape(clean_text)
    return clean_text

# Hàm ghi tóm tắt ra file HTML với trọng số của câu
# Hàm ghi tóm tắt và ma trận độ tương đồng vào file HTML
def write_summary_with_matrix(output_path, summary, num_sentences, similarity_matrix, sentences, scores):
    with open(output_path, 'w', encoding='utf-8') as file:
        # Thêm thông tin ngày giờ xuất file
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        file.write("<html>\n<head>\n<title>Text Summary</title>\n")
        # Đảm bảo mã hóa đúng UTF-8 và font chữ hiển thị đúng
        file.write('<meta charset="UTF-8">\n')
        file.write('<style>\n')
        file.write('body {font-family: Arial, sans-serif;}\n')
        file.write('table {border-collapse: collapse; width: 100%; margin-top: 20px;}\n')
        file.write('th, td {border: 1px solid #ddd; padding: 8px; text-align: center;}\n')
        file.write('th {background-color: #f2f2f2;}\n')
        file.write('tr:nth-child(even) {background-color: #f9f9f9;}\n')
        file.write('tr:hover {background-color: #f1f1f1;}\n')
        file.write('</style>\n</head>\n<body>\n')

        # Tiêu đề và số lượng câu
        file.write(f"<h1>Text Summary (generated on {now})</h1>\n")
        file.write(
            f"<h3>Number of sentences in the original text: {len(sentences)}, Number of sentences in the summary: {num_sentences}</h3>\n")

        # Ghi phần tóm tắt
        file.write("<h2>Summary:</h2>\n")
        for idx, sentence in enumerate(summary):
            weight = scores[idx]  # Lấy trọng số của câu
            # Đảm bảo weight là kiểu số (float) và không phải là tuple
            if isinstance(weight, (int, float)):
                file.write(f"<p>{sentence} </p>")  # Sử dụng thẻ <p>
            else:
                file.write(f"<p>{sentence} </p>")  # Nếu không phải số thì ghi "Invalid"

        # Ghi ma trận độ tương đồng
        file.write("<h2>Similarity Matrix:</h2>\n<table>\n<tr><th></th>\n")

        # Header của bảng (các câu được đánh chỉ số)
        for i in range(len(sentences)):
            file.write(f"<th>Sentence {i + 1}</th>")
        file.write("</tr>\n")

        # Nội dung ma trận
        for i, row in enumerate(similarity_matrix):
            file.write(f"<tr><th>Sentence {i + 1}</th>")  # Ghi tiêu đề dòng
            for value in row:
                file.write(f"<td>{value:.4f}</td>")  # Định dạng giá trị độ tương đồng (4 chữ số thập phân)
            file.write("</tr>\n")

        file.write("</table>\n</body>\n</html>")


# Hàm vẽ đồ thị và hiển thị trực tiếp
def draw_and_show_graph(graph):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=12, font_weight='bold')
    plt.title("Đồ thị độ tương đồng giữa các câu")
    plt.show()  # Hiển thị đồ thị trong cửa sổ đồ họa


# Hàm ghi tóm tắt ra file HTML, bao gồm thời gian xuất file và trọng số của câu
from datetime import datetime

from datetime import datetime


def write_summary(output_path, summary, num_sentences):
    with open(output_path, 'w', encoding='utf-8') as file:
        # Thêm thông tin ngày giờ xuất file
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        file.write("<html>\n<head>\n<title>Text Summary</title>\n")
        # Đảm bảo mã hóa đúng UTF-8 và font chữ hiển thị đúng
        file.write('<meta charset="UTF-8">\n')
        file.write('<style>\n')
        file.write('body {font-family: Arial, sans-serif;}\n')
        file.write('table {border-collapse: collapse; width: 100%; margin-top: 20px;}\n')
        file.write('th, td {border: 1px solid #ddd; padding: 8px; text-align: center;}\n')
        file.write('th {background-color: #f2f2f2;}\n')
        file.write('tr:nth-child(even) {background-color: #f9f9f9;}\n')
        file.write('tr:hover {background-color: #f1f1f1;}\n')
        file.write('</style>\n')
        file.write("</head>\n<body>\n")
        file.write(f"<h1>Text Summary (generated on {now})</h1>\n")
        file.write(
            f"<h3>Number of sentences in the original text: {num_sentences}, Number of sentences in the summary: {num_sentences}</h3>\n")

        # Ghi từng câu trong danh sách tóm tắt vào file HTML mà không có trọng số
        file.write("<ul>\n")
        for sentence in summary:
            file.write(f"{sentence}.")  # Không thêm trọng số
        file.write("</ul>\n")

        file.write("</body>\n</html>")



# Hàm chính
def main(input_file, output_file, ratio=0.2):
    text = read_and_process_html(input_file)
    summary, scores, similarity_matrix, graph = summarize_text_textrank_auto(text, ratio=ratio)

    # Vẽ đồ thị và hiển thị trực tiếp
    draw_and_show_graph(graph)

    # Ghi tóm tắt và ma trận độ tương đồng vào file HTML
    write_summary_with_matrix(output_file, summary, len(summary), similarity_matrix, split_sentences(text), scores)

    print(f"Bản tóm tắt và ma trận độ tương đồng đã được ghi vào file: {output_file}")



# Hàm xử lý tất cả các file trong thư mục
def process_all_files(input_folder, output_folder, ratio=0.2):
    import os
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".html"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name.replace(".html", "_summary.html"))
            # Gọi hàm main với 3 tham số
            main(input_path, output_path, ratio)


# Chạy chương trình
if __name__ == "__main__":
    input_folder = "C:/Users/PC/PycharmProjects/bot2vec/NLP_2/input"  # Thư mục chứa file đầu vào
    output_folder = "C:/Users/PC/PycharmProjects/bot2vec/NLP_2/output"  # Thư mục chứa file đầu ra
    process_all_files(input_folder, output_folder)
