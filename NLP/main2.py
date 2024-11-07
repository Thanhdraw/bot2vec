import re
from collections import Counter
import string
import os
import matplotlib.pyplot as plt
import networkx as nx


class TextSummarizer:
    def __init__(self):
        self.input_text = ""
        self.sentences = []
        self.word_freq = Counter()
        self.sentence_graph = None

    def load_text_from_file(self, file_path):
        """
        Đọc văn bản từ file.

        Args:
            file_path (str): Đường dẫn tới file chứa văn bản.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.input_text = file.read()
            self.load_text(self.input_text)
        except FileNotFoundError:
            print(f"Không tìm thấy file: {file_path}")

    def load_text(self, text):
        """
        Nạp văn bản và chia thành các câu.

        Args:
            text (str): Văn bản cần nạp.

        Returns:
            int: Số lượng câu trong văn bản.
        """
        self.input_text = text
        self.sentences = re.split('[.!?]', self.input_text)
        self.sentences = [s.strip() for s in self.sentences if s.strip()]
        return len(self.sentences)

    def preprocess_text(self):
        """
        Tiền xử lý văn bản, bao gồm chuyển sang chữ thường và loại bỏ các ký tự đặc biệt.
        """
        words = self.input_text.lower()
        words = re.sub(r'[{}]'.format(string.punctuation), ' ', words)
        words = words.split()
        self.word_freq = Counter(words)

    def sentence_similarity(self, s1, s2):
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

    def create_sentence_graph(self):
        """
        Tạo đồ thị câu.
        """
        G = nx.Graph()
        for i, sentence in enumerate(self.sentences):
            G.add_node(i, sentence=sentence)

        for i in range(len(self.sentences)):
            for j in range(i + 1, len(self.sentences)):
                weight = self.sentence_similarity(self.sentences[i], self.sentences[j])
                if weight > 0:
                    G.add_edge(i, j, weight=weight)

        self.sentence_graph = G

    def rank_sentences(self):
        """
        Xếp hạng câu dựa trên đồ thị.
        """
        if self.sentence_graph is None:
            self.create_sentence_graph()

        scores = nx.pagerank(self.sentence_graph)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [self.sentences[idx] for idx, _ in sorted_scores]

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
        return summary

    def evaluate(self, original, summary):
        """
        Đánh giá bản tóm tắt.

        Args:
            original (str): Văn bản gốc.
            summary (str): Bản tóm tắt.

        Returns:
            dict: Các chỉ số đánh giá, bao gồm tỷ lệ nén, số câu trong văn bản gốc và số câu trong bản tóm tắt.
        """
        compression_ratio = len(summary.split()) / len(original.split())
        return {
            'compression_ratio': compression_ratio,
            'original_sentences': len(self.sentences),
            'summary_sentences': len(summary.split('.'))
        }

    def save_to_html(self, output_file, original_text, summary_text, metrics):
        """
        Lưu văn bản gốc, bản tóm tắt và các chỉ số đánh giá vào file HTML.

        Args:
            output_file (str): Đường dẫn tới file HTML sẽ được tạo.
            original_text (str): Văn bản gốc.
            summary_text (str): Bản tóm tắt.
            metrics (dict): Các chỉ số đánh giá.
        """
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
            </style>
        </head>
        <body>
            <h1>Text Summarization</h1>
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

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(html_content)

        print(f"HTML output saved to: {output_file}")

    def plot_sentence_graph(self):
        """
        Vẽ đồ thị câu.
        """
        if self.sentence_graph is None:
            self.create_sentence_graph()

        pos = nx.spring_layout(self.sentence_graph)
        plt.figure(figsize=(12, 8))
        nx.draw(self.sentence_graph, pos, with_labels=True)
        plt.title("Đồ thị câu")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    file_path = os.path.join("/Users", "dangquocthanh", "bot2vec", "NLP", "input", "d061j.html")
    output_file = os.path.join("/Users", "dangquocthanh", "bot2vec", "NLP", "output", "summary.html")

    summarizer = TextSummarizer()
    summarizer.load_text_from_file(file_path)
    summary = summarizer.summarize(summarizer.input_text, ratio=0.4)
    metrics = summarizer.evaluate(summarizer.input_text, summary)

    summarizer.save_to_html(output_file, summarizer.input_text, summary, metrics)
    summarizer.plot_sentence_graph()
