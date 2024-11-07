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

