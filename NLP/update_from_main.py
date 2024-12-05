import re
from collections import Counter
import string
import os
from datetime import datetime
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import networkx as nx

import html  # Thư viện cần thiết cho việc escape HTML


# Bước 1: Tiền xử lý (loại bỏ HTML, tách câu, tính tần suất từ).
# Bước 2: Xây dựng đồ thị câu.
# Bước 3: Xếp hạng các câu bằng thuật toán PageRank.
# Bước 4: Tóm tắt văn bản dựa trên điểm xếp hạng.
# Bước 5: Đánh giá bản tóm tắt.
# Bước 6: Lưu kết quả vào file HTML và vẽ đồ thị.
# Dựa trên mã nguồn, bài toán tóm tắt văn bản này được thực hiện qua 6 bước chính:
#
# 1. Tiền xử lý (load_text, preprocess_text)
# 2. Xây dựng đồ thị câu (create_sentence_graph)
# 3. Xếp hạng các câu bằng thuật toán PageRank (rank_sentences)
# 4. Tóm tắt văn bản bằng cách chọn các câu quan trọng nhất (summarize)
# 5. Đánh giá bản tóm tắt (evaluate)
# 6. Lưu kết quả và vẽ đồ thị (save_to_html, plot_sentence_graph)
#
# Mỗi bước được thực hiện chi tiết trong lớp TextSummarizer.

def sentence_similarity(s1, s2):
    """
    Tính độ tương đồng giữa hai câu.
    Args:
        s1 (str): Câu thứ nhất.
        s2 (str): Câu thứ hai.

    Returns:
        float: Độ tương đồng giữa hai câu.
    """
    words1 = set(s1.lower().split())
    words2 = set(s2.lower().split())
    return len(words1.intersection(words2)) / (len(words1) + len(words2))


def add_sentence_numbers(sentences):
    """
    Thêm số thứ tự vào các câu.

    Args:
        sentences (list): Danh sách các câu.

    Returns:
        list: Danh sách các câu có đánh số.
        :param sentences:
        :param self:
    """
    return [f"[{i + 1}] {sentence}" for i, sentence in enumerate(sentences)]


def get_numbered_summary_text(summary_sentences):
    """
    Trả về bản tóm tắt với số câu.
    """
    return ' '.join(add_sentence_numbers(summary_sentences))


class TextSummarizer:
    def __init__(self):
        self.input_text = ""
        self.sentences = []
        self.word_freq = Counter()
        self.sentence_graph = None

    # Bước 1.1: Đọc văn bản từ file HTML, loại bỏ thẻ HTML và nạp nội dung.
    def load_text_from_file(self, file_path):
        """
        Đọc văn bản từ file, loại bỏ các thẻ <s> và tái tạo văn bản hoàn chỉnh.
        Args:
            file_path (str): Đường dẫn tới file chứa văn bản.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')
                # Lấy tất cả văn bản, bỏ qua thẻ <s>
                cleaned_text = ' '.join(soup.stripped_strings)
                self.load_text(cleaned_text)
            print(f"Đọc file thành công: {file_path}")
        except FileNotFoundError:
            print(f"Không tìm thấy file: {file_path}")

    # Bước 1.2: Tách văn bản thành các câu.
    def load_text(self, text):
        """
        Nạp văn bản và chia thành các câu.

        Args:
            text (str): Văn bản cần nạp.

        Returns:
            int: Số lượng câu trong văn bản.
        """
        self.input_text = text
        # Tách văn bản thành các câu
        self.sentences = re.split('[.!?]', self.input_text)
        self.sentences = [s.strip() for s in self.sentences if s.strip()]
        print(f"Số câu trong văn bản: {len(self.sentences)}")
        return len(self.sentences)

    # Bước 1.3: Tiền xử lý văn bản để tính tần suất từ.
    def preprocess_text(self):
        """
        Tiền xử lý văn bản, bao gồm chuyển sang chữ thường và loại bỏ các ký tự đặc biệt.
        """
        words = self.input_text.lower()
        words = re.sub(r'[{}]'.format(string.punctuation), ' ', words)
        words = words.split()
        self.word_freq = Counter(words)
        print(f"Số lượng từ trong văn bản: {len(self.word_freq)}")

    # Bước 2: Tạo đồ thị câu.
    def create_sentence_graph(self, threshold=0.3):
        """
        Tạo đồ thị câu với ngưỡng độ tương đồng.
        """
        G = nx.Graph()

        # Thêm các đỉnh (nodes)
        for i, sentence in enumerate(self.sentences):
            G.add_node(i, sentence=sentence)

        # Thêm các cạnh (edges) với ngưỡng độ tương đồng
        for i in range(len(self.sentences)):
            for j in range(i + 1, len(self.sentences)):
                weight = sentence_similarity(self.sentences[i], self.sentences[j])
                if weight > threshold:  # Chỉ thêm cạnh nếu độ tương đồng lớn hơn threshold
                    G.add_edge(i, j, weight=weight)

        self.sentence_graph = G
        print(f"Đồ thị câu đã được tạo. Số lượng đỉnh: {len(G.nodes)}, Số lượng cạnh: {len(G.edges)}")

    def filter_graph_by_pagerank(self, top_n=10):
        """
        Lọc đồ thị để chỉ giữ lại các câu quan trọng nhất dựa trên PageRank.
        """
        pagerank = nx.pagerank(self.sentence_graph)  # Tính điểm PageRank
        top_nodes = sorted(pagerank, key=pagerank.get, reverse=True)[:top_n]  # Lấy top N đỉnh

        # Tạo đồ thị con chỉ chứa các đỉnh quan trọng
        filtered_graph = self.sentence_graph.subgraph(top_nodes)
        print(
            f"Đồ thị đã được lọc. Số lượng đỉnh: {len(filtered_graph.nodes)}, Số lượng cạnh: {len(filtered_graph.edges)}")

        return filtered_graph

    def detect_communities(self):
        """
        Phát hiện các cụm trong đồ thị câu.
        """
        import community as community_louvain

        # Phát hiện cụm
        partition = community_louvain.best_partition(self.sentence_graph)

        # Gán thuộc tính cụm cho từng đỉnh
        nx.set_node_attributes(self.sentence_graph, partition, 'community')

        print("Cụm đã được phát hiện.")
        return partition

    # Bước 3: Xếp hạng các câu bằng thuật toán PageRank.
    def rank_sentences(self):
        """
        Xếp hạng câu dựa trên đồ thị.
        """
        if self.sentence_graph is None:
            self.create_sentence_graph()

        scores = nx.pagerank(self.sentence_graph)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [self.sentences[idx] for idx, _ in sorted_scores]

    # Bước 4: Tóm tắt văn bản bằng cách chọn các câu quan trọng nhất.
    def summarize(self, text, ratio=0.3):
        """
        Tóm tắt văn bản.
        Args:
            text (str): Văn bản cần tóm tắt.
            ratio (float, optional): Tỷ lệ số câu trong bản tóm tắt so với văn bản gốc. Mặc định là 0.3.

        Returns:
            str: Bản tóm tắt.
        """
        self.load_text(text)
        self.preprocess_text()
        ranked_sentences = self.rank_sentences()
        summary = ' '.join(ranked_sentences[:int(len(ranked_sentences) * ratio)])
        print(f"Bản tóm tắt: {summary[:200]}...")  # In ra 200 ký tự đầu tiên của tóm tắt
        return summary

    # Bước 5: Đánh giá bản tóm tắt dựa trên tỷ lệ nén và số câu.

    def evaluate(self, original, summary):
        """
        Đánh giá bản tóm tắt.
        """
        original_word_count = len(original.split())
        summary_word_count = len(summary.split())

        # Tránh lỗi chia cho 0
        if original_word_count == 0:
            print("Văn bản gốc rỗng, không thể đánh giá.")
            return {
                'compression_ratio': 0,
                'original_sentences': len(self.sentences),
                'summary_sentences': len(summary.split('.'))
            }

        compression_ratio = summary_word_count / original_word_count
        return {
            'compression_ratio': compression_ratio,
            'original_sentences': len(self.sentences),
            'summary_sentences': len(summary.split('.'))
        }

    # tạo thanành các dữ liệu html
    # Bước 6.1: Lưu kết quả vào file HTML.
    @staticmethod
    def save_to_html(output_file, original_text, summary_text, metrics):
        """
        Lưu văn bản gốc, bản tóm tắt và các chỉ số đánh giá vào file HTML.

        Args:
            output_file (str): Đường dẫn tới file HTML sẽ được tạo.
            original_text (str): Văn bản gốc.
            summary_text (str): Bản tóm tắt.
            metrics (dict): Các chỉ số đánh giá.
        """
        # Escape các ký tự đặc biệt trong văn bản
        original_text = html.escape(original_text)
        summary_text = html.escape(summary_text)

        # Lấy thời gian hiện tại
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Text Summarization</title>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                h1 {{ text-align: center; }}
                .section {{ margin-bottom: 20px; }}
                pre, p {{ text-decoration: none; }}
                .timestamp {{ font-size: 0.9em; color: #555; text-align: right; margin-top: 10px; }}
            </style>
        </head>
        <body>
            <h1>Text Summarization</h1>
            <div class="timestamp">
                <p>Generated on: {current_time}</p>
            </div>
            <div class="section">
                <h2>Original Text</h2>
                <pre style="white-space: pre-wrap;">{original_text}</pre>
            </div>
            <div class="section">
                <h2>Summary</h2>
                <pre style="white-space: pre-wrap;">{summary_text}</pre>
            </div>
            <div class="section">
                <h2>Evaluation</h2>
                <p>Compression Ratio: {metrics['compression_ratio']:.2f}</p>
                <p>Original Sentences: {metrics['original_sentences']}</p>
                <p>Summary Sentences: {metrics['summary_sentences']}</p>
            </div>
        </body>
        </html>
        """

        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Ghi nội dung HTML vào file
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(html_content)

        print(f"HTML output saved to: {output_file}")



    def plot_sentence_graph(self):
        """
        Vẽ đồ thị câu với các nhãn câu và cạnh nối giữa các câu.
        """
        if self.sentence_graph is None:
            self.create_sentence_graph()

        # Kiểm tra các cạnh trong đồ thị
        print("Cạnh trong đồ thị:", list(self.sentence_graph.edges()))

        # Sử dụng spring_layout để tạo vị trí các đỉnh
        pos = nx.spring_layout(self.sentence_graph, seed=42, k=0.15)  # Thử điều chỉnh tham số k

        # Tạo figure
        plt.figure(figsize=(12, 8))

        # Vẽ các đỉnh (nút)
        nx.draw_networkx_nodes(self.sentence_graph, pos, node_size=500, node_color='lightblue')

        # Vẽ các cạnh với độ dày và màu sắc đen
        nx.draw_networkx_edges(self.sentence_graph, pos, width=2, edge_color='black', alpha=0.7)

        # Vẽ các nhãn của các đỉnh (các câu)
        nx.draw_networkx_labels(self.sentence_graph, pos, font_size=10, font_family='sans-serif')

        # Tiêu đề cho đồ thị
        plt.title("Đồ thị câu - Mối quan hệ giữa các câu", fontsize=15)

        # Hiển thị đồ thị
        plt.axis('off')  # Tắt trục tọa độ
        plt.tight_layout()  # Điều chỉnh bố cục để tránh nhãn bị che khuất
        plt.show()

    def get_numbered_original_text(self):
        """
        Trả về văn bản gốc với số câu.
        """
        return ' '.join(add_sentence_numbers(self.sentences))


if __name__ == "__main__":
    input_dir = r"C:\Users\PC\PycharmProjects\bot2vec\NLP\input"
    output_dir = r"C:\Users\PC\PycharmProjects\bot2vec\NLP\output"
    os.makedirs(output_dir, exist_ok=True)

    input_files = [f for f in os.listdir(input_dir) if f.endswith('.html')]

    for input_filename in input_files:
        input_file_path = os.path.join(input_dir, input_filename)
        output_filename = re.sub(r'\.html$', '_summary.html', input_filename)
        output_file_path = os.path.join(output_dir, output_filename)

        try:
            summarizer = TextSummarizer()
            summarizer.load_text_from_file(input_file_path)

            summary = summarizer.summarize(summarizer.input_text, ratio=0.4)
            numbered_original_text = summarizer.get_numbered_original_text()
            numbered_summary_text = get_numbered_summary_text(summary.split('. '))

            metrics = summarizer.evaluate(summarizer.input_text, summary)
            summarizer.save_to_html(output_file_path, numbered_original_text, numbered_summary_text, metrics)
            # Create the sentence graph
            summarizer.create_sentence_graph()

            # Plot the sentence graph
            summarizer.plot_sentence_graph()
            print(f"Processed: {input_filename} -> {output_filename}")


        except Exception as e:
            print(f"Error processing {input_filename}: {e}")
            with open('error_log.txt', 'a') as log_file:
                log_file.write(f"Error processing {input_filename}: {e}\n")
