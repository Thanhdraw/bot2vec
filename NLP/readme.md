file_path = os.path.join("/Users", "dangquocthanh", "bot2vec", "NLP", "input", "d061j.html")
    output_file = os.path.join("/Users", "dangquocthanh", "bot2vec", "NLP", "output", "summary.html")




Phương pháp được sử dụng trong code trên là "Extractive Text Summarization" (Tóm tắt văn bản trích xuất)
với phương pháp scoring-based (dựa trên điểm số).

### Yêu cầu
1.Đầu vào là đồ thị
2. Biểu diễn dựa trên đồ thị G(V,E)
3. Xep hạng dựa trện đồ thị
4. Tóm tat




### đã làm
Trong mã nguồn ban đầu, đã thực hiện các bước sau:

1. **Tiền xử lý văn bản**
- Tách văn bản thành các câu
    -> mỗi câu là một từ !
- Chuyển văn bản sang chữ thường
- Loại bỏ ký tự đặc biệt
- Đếm tần suất từ ??? giải thích ?
    -> đếm tần suất từ -> biết từ quan trọng !

2. **Xây dựng đồ thị câu**
- Mỗi câu là một nút
- Tính độ tương đồng giữa các câu bằng phương pháp Jaccard
- Kết nối các câu có độ tương đồng > 0 ????

3. **Xếp hạng câu**
- Sử dụng thuật toán PageRank
- Xác định tầm quan trọng của câu dựa trên kết nối ?để chi ?

4. **Tạo bản tóm tắt**
- Chọn các câu có điểm số cao nhất ?
- Giữ nguyên thứ tự câu trong văn bản gốc

**Hạn chế của phương pháp hiện tại:**
- Độ tương đồng câu đơn giản (chỉ đếm từ chung)
- Chưa sử dụng vector để biểu diễn ngữ nghĩa
- Chưa xem xét mối quan hệ sâu giữa các từ

######
Đây là kết quả của phương thức `evaluate()` trong class `TextSummarizer`,

1. `compression_ratio`: 0.4286 (khoảng 42.86%)
   - Tỷ lệ số từ trong bản tóm tắt so với văn bản gốc
   - Ở đây, bản tóm tắt chứa khoảng 42.86% số từ của văn bản gốc

2. `original_sentences`: 11
   - Tổng số câu trong văn bản gốc

3. `summary_sentences`: 1
   - Số câu trong bản tóm tắt

Các chỉ số này giúp đánh giá mức độ nén và hiệu quả của bản tóm tắt:
- Cho thấy bản tóm tắt khá ngắn gọn
- Chỉ sử dụng 1 câu từ tổng số 11 câu ban đầu
- Giảm được khoảng 57% số từ so với văn bản gốc


/////////////////////////////////////////////////////////////////

### Ví dụ về những gì đã làm và cải tiến:

#### 1. **Tiền xử lý văn bản**
   - **Đã làm**:
     - Tách văn bản thành các câu.
     - Chuyển văn bản sang chữ thường và loại bỏ ký tự đặc biệt.
     - Đếm tần suất từ để xác định từ quan trọng.
   - **Cải tiến**:
     - **Lemmatization** thay vì chỉ **stemming** (vì lemmatization sẽ giúp giữ nguyên ý nghĩa từ, trong khi stemming có thể làm mất thông tin).
     - Loại bỏ **stop words** (các từ không mang nhiều thông tin như "the", "a", "in", ...) để chỉ tập trung vào các từ khóa quan trọng.
     - Sử dụng **TF-IDF (Term Frequency-Inverse Document Frequency)** thay vì chỉ đếm tần suất từ, giúp các từ ít xuất hiện trong toàn bộ văn bản nhưng quan trọng sẽ được ưu tiên hơn.

#### 2. **Xây dựng đồ thị câu**
   - **Đã làm**:
     - Mỗi câu trong văn bản là một nút trong đồ thị.
     - Độ tương đồng giữa các câu được tính bằng phương pháp **Jaccard** (so sánh sự giao nhau và hợp nhất của từ trong các câu).
   - **Cải tiến**:
     - Áp dụng phương pháp tính **cosine similarity** sử dụng **vector hóa câu** thông qua mô hình **Word2Vec**, **GloVe**, hoặc **BERT**, để tính toán độ tương đồng giữa các câu một cách chính xác hơn, giúp nhận diện các câu có nghĩa tương tự mặc dù từ ngữ có thể khác nhau.
     - Cập nhật các **câu quan trọng** và **mối quan hệ ngữ nghĩa** thay vì chỉ so sánh sự giao nhau của từ ngữ.

#### 3. **Xếp hạng câu**
   - **Đã làm**:
     - Sử dụng **PageRank** để xếp hạng câu trong đồ thị, dựa trên kết nối giữa các câu.
   - **Cải tiến**:
     - Thêm **Maximal Marginal Relevance (MMR)** để giảm thiểu sự trùng lặp và đảm bảo bản tóm tắt không chỉ chứa các câu quan trọng mà còn đảm bảo tính đa dạng của các thông tin.
     - Sử dụng các yếu tố khác như **độ dài câu**, **tần suất từ khóa** để bổ sung cho điểm PageRank và cải thiện xếp hạng.

#### 4. **Tạo bản tóm tắt**
   - **Đã làm**:
     - Chọn các câu có điểm số cao nhất để tạo bản tóm tắt.
     - Giữ nguyên thứ tự câu trong văn bản gốc.
   - **Cải tiến**:
     - **Chọn câu** không chỉ dựa trên điểm số, mà còn đảm bảo tính **bao quát** và **tính liên kết** giữa các câu trong văn bản gốc, để bản tóm tắt không chỉ ngắn gọn mà vẫn đầy đủ thông tin.
     - **Thêm mô hình học sâu (Deep Learning)** để tạo ra bản tóm tắt **tự nhiên** và **mạch lạc hơn** (chuyển từ phương pháp trích xuất sang phương pháp tóm tắt sáng tạo).

#### 5. **Sử dụng mô hình học sâu**
   - **Đã làm**:
     - Phương pháp hiện tại chỉ dựa vào kỹ thuật truyền thống như PageRank và Jaccard.
   - **Cải tiến**:
     - Áp dụng các mô hình **BERT**, **T5**, hoặc **GPT** để có thể tạo ra các bản tóm tắt **abstractive** (sáng tạo), thay vì chỉ **extractive** (trích xuất). Điều này giúp tạo ra tóm tắt tự nhiên, mạch lạc và có thể diễn đạt lại thông tin theo cách dễ hiểu hơn.

#### Kết quả của các cải tiến:
   - Bản tóm tắt sẽ **chính xác hơn**, **ngắn gọn hơn**, và **tự nhiên hơn**.
   - Tăng tính **đa dạng** và giảm **trùng lặp** trong bản tóm tắt.
   - **Tốt hơn trong việc xử lý các văn bản phức tạp**, đặc biệt là khi có sự khác biệt về từ ngữ nhưng nội dung tương đồng.
