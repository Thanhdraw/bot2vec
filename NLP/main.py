import matplotlib.pyplot as plt
from collections import Counter
import re
import string


# Định nghĩa class TextSummarizer
class TextSummarizer:
    def __init__(self):
        """Khởi tạo các biến cần thiết"""
        self.input_text = ""
        self.sentences = []
        self.word_freq = Counter()

    def load_text(self, text):
        """Input: Nhận văn bản đầu vào
           Output: Xử lý và lưu văn bản"""
        self.input_text = text
        # Tách câu dựa trên dấu chấm, chấm hỏi, chấm than
        self.sentences = re.split('[.!?]', self.input_text)
        self.sentences = [s.strip() for s in self.sentences if s.strip()]
        return len(self.sentences)

    def preprocess_text(self):
        """Tiền xử lý văn bản:
        - Loại bỏ ký tự đặc biệt
        - Chuyển về chữ thường
        - Tách từ"""
        # Tính tần suất từ
        words = self.input_text.lower()
        words = re.sub(r'[{}]'.format(string.punctuation), ' ', words)
        words = words.split()
        self.word_freq = Counter(words)

    def score_sentences(self):
        """Tính điểm cho từng câu dựa trên:
        - Tần suất từ xuất hiện
        - Vị trí câu trong văn bản
        - Độ dài câu"""
        scores = {}
        for i, sentence in enumerate(self.sentences):
            # Điểm vị trí (câu đầu và cuối quan trọng hơn)
            position_score = 1.0
            if i == 0 or i == len(self.sentences) - 1:
                position_score = 1.2

            # Điểm tần suất từ
            words = sentence.lower().split()
            word_score = sum(self.word_freq[word] for word in words) / len(words)

            # Điểm độ dài (ưu tiên câu độ dài vừa phải)
            length_score = 1.0
            if 10 <= len(words) <= 20:
                length_score = 1.1

            scores[sentence] = position_score * word_score * length_score

        return scores

    def summarize(self, text, ratio=0.3):
        """Tạo bản tóm tắt với tỷ lệ số câu mong muốn"""
        # Kiểm tra tỷ lệ hợp lệ
        ratio = max(0.1, min(ratio, 0.9))

        # Load text trước
        self.load_text(text)

        # Tiền xử lý
        self.preprocess_text()

        # Tính điểm các câu
        scores = self.score_sentences()

        # Chọn các câu có điểm cao nhất
        n_sentences = max(1, int(len(self.sentences) * ratio))
        top_sentences = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_sentences]

        # Sắp xếp lại các câu theo thứ tự xuất hiện trong văn bản gốc
        summary = []
        for sentence, _ in sorted(top_sentences,
                                  key=lambda x: self.sentences.index(x[0])):
            summary.append(sentence)

        return ' '.join(summary)

    def evaluate(self, original, summary):
        """Đánh giá chất lượng bản tóm tắt:
        - Tỷ lệ nén
        - Độ bao phủ thông tin chính
        - Tính mạch lạc"""
        original_words = len(original.split())
        summary_words = len(summary.split())
        compression_ratio = summary_words / original_words
        summary_sentences = re.split('[.!?]', summary)
        summary_sentences = [s.strip() for s in summary_sentences if s.strip()]

        # Hàm vẽ đồ thị từ bản tóm tắt
        def plot_summary_word_freq(summary):
            # Chuyển về chữ thường và loại bỏ ký tự đặc biệt
            words = summary.lower()
            words = re.sub(r'[{}]'.format(string.punctuation), '', words)
            word_list = words.split()

            # Đếm tần suất từ
            word_freq = Counter(word_list)

            # Lấy 10 từ xuất hiện nhiều nhất
            most_common_words = word_freq.most_common(10)
            words, frequencies = zip(*most_common_words)

            # Vẽ đồ thị
            plt.figure(figsize=(10, 6))
            plt.bar(words, frequencies, color='skyblue')
            plt.xlabel('Từ')
            plt.ylabel('Tần suất')
            plt.title('Tần suất các từ trong bản tóm tắt')
            plt.xticks(rotation=45)
            plt.show()

        # Sử dụng hàm này sau khi đã tạo bản tóm tắt
        plot_summary_word_freq(summary)
        return {
            'compression_ratio': compression_ratio,
            'original_sentences': len(self.sentences),
            'summary_sentences': len(summary_sentences)
        }


# Phần chạy chương trình (đặt bên ngoài class)
if __name__ == "__main__":
    # Văn bản mẫu
    text = """
   Nhìn chung, nguyên nhân đau cơ xương khớp phổ biến nhất là về cơ năng, tức mất cân bằng cơ xung quanh ổ khớp. Chỉ có một số nhóm cơ được sử dụng nhiều hơn do tư thế ngồi nhiều và sử dụng máy tính hay điện thoại thường xuyên. Lâu dài, chúng gây lệch trọng tâm khớp, trục khớp, dẫn đến đau khớp, tổn thương sụn khớp, thoái hóa.

"Người dân đi massage, vật lý trị liệu, dùng nhiệt, hồng ngoại hay sử dụng thuốc giảm đau, giãn cơ chỉ làm giãn cơ một phần hoặc toàn bộ cơ thể. Bệnh nhân dễ chịu trong thời gian điều trị nhưng cơn đau sẽ quay trở lại bởi vấn đề mất cân bằng cơ vẫn còn ở đó, thói quen vận động, sinh hoạt không thay đổi", bác sĩ Quang Anh nhấn mạnh.

Ông cũng cho rằng đa số bệnh nhân chỉ đang điều trị phần "ngọn", gốc rễ vấn đề vẫn chưa được giải quyết khiến bệnh tái đi tái lại, thậm chí diễn tiến nặng hơn thành thoái hóa, thoát vị đĩa đệm.
    """

    # Khởi tạo đối tượng TextSummarizer
    summarizer = TextSummarizer()

    # Tạo bản tóm tắt
    summary = summarizer.summarize(text, ratio=0.4)

    # In kết quả
    print("Văn bản gốc:")
    print(text)
    print("\nBản tóm tắt:")
    print(summary)

    # Đánh giá
    metrics = summarizer.evaluate(text, summary)
    print("\nĐánh giá:")
    print(metrics)
