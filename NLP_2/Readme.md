Chương trình này thực hiện các bước sau để xử lý và tóm tắt văn bản từ một file HTML:
Tổng kết
Tóm lại, quá trình xây dựng đồ thị trong TextRank có thể được chia thành các bước:

Tiền xử lý văn bản.
Tách câu thành các phần tử riêng biệt.
Vector hóa các câu.
Tính toán ma trận độ tương đồng giữa các câu.
Xây dựng đồ thị từ ma trận độ tương đồng.
Tính toán điểm PageRank cho từng câu.
Chọn ra các câu quan trọng nhất để tóm tắt.

Hy vọng giải thích này giúp bạn hiểu rõ hơn về cách tính toán và xây dựng đồ thị trong thuật toán TextRank.
Chương trình này thực hiện các bước sau:

### 1. **Tiền xử lý văn bản**
   - **Chức năng**: Chương trình bắt đầu bằng việc tiền xử lý văn bản, bao gồm:
     - **Chuyển văn bản sang chữ thường**: Đảm bảo rằng việc phân tích văn bản không phân biệt giữa chữ hoa và chữ thường.
     - **Loại bỏ ký tự đặc biệt**: Sử dụng biểu thức chính quy để loại bỏ các ký tự không phải chữ cái, chữ số hoặc khoảng trắng.

### 2. **Tách câu trong văn bản**
   - **Chức năng**: Văn bản được tách thành các câu riêng biệt. Điều này giúp chương trình xử lý từng câu riêng biệt khi áp dụng thuật toán tóm tắt.

### 3. **Tóm tắt văn bản sử dụng thuật toán TextRank**
   - **Thuật toán TextRank**: 
     - **Vector hóa câu**: Mỗi câu trong văn bản được chuyển thành một vector (dãy số) bằng cách sử dụng `CountVectorizer`. Điều này giúp tạo ra ma trận đại diện cho các câu trong văn bản.
     - **Tính độ tương đồng giữa các câu**: Chương trình tính toán độ tương đồng giữa các câu sử dụng phép toán cosine similarity. Ma trận tương đồng giữa các câu được xây dựng.
     - **Xây dựng đồ thị và tính điểm PageRank**: Mỗi câu được coi như một đỉnh trong đồ thị, và độ tương đồng giữa các câu được coi là trọng số của các cạnh. Thuật toán PageRank được sử dụng để tính toán độ quan trọng của từng câu.
     - **Sắp xếp câu**: Các câu được xếp hạng theo độ quan trọng và chọn ra những câu có điểm PageRank cao nhất để tạo thành bản tóm tắt.

### 4. **Vẽ đồ thị thể hiện mối quan hệ giữa các câu**
   - **Đồ thị vector**: Chương trình sử dụng `networkx` để tạo ra đồ thị thể hiện mối quan hệ giữa các câu trong văn bản. Các đỉnh trong đồ thị tương ứng với các câu, và các cạnh thể hiện mức độ tương đồng giữa các câu.
   - **Vẽ đồ thị**: Đồ thị này sau đó được vẽ ra bằng thư viện `matplotlib`, với các nút là các câu trong văn bản, và các cạnh nối các câu có độ tương đồng cao.

### 5. **Đọc văn bản từ file HTML**
   - **Chức năng**: Văn bản được đọc từ một file HTML, sau đó loại bỏ tất cả các thẻ HTML và giải mã các ký tự đặc biệt như `&lt;`, `&gt;`, `&quot;` để có được nội dung văn bản thuần túy.

### 6. **Ghi tóm tắt ra file HTML**
   - **Chức năng**: Sau khi tóm tắt văn bản, bản tóm tắt (một danh sách các câu đã được chọn) được ghi vào một file HTML mới. Mỗi câu trong bản tóm tắt được hiển thị dưới dạng một mục trong danh sách (`<ul>` trong HTML).

### 7. **Hàm chính (main)**
   - **Chức năng**: Hàm chính tích hợp toàn bộ quy trình, bao gồm:
     - Đọc và xử lý văn bản từ file HTML.
     - Tóm tắt văn bản sử dụng thuật toán TextRank.
     - Vẽ đồ thị thể hiện mối quan hệ giữa các câu.
     - Ghi bản tóm tắt ra một file HTML.
     - In thông báo cho người dùng biết rằng bản tóm tắt đã được ghi vào file.

### 8. **Đầu vào và đầu ra của chương trình**
   - **Đầu vào**: Chương trình nhận một file HTML chứa văn bản cần tóm tắt và số lượng câu muốn tóm tắt.
   - **Đầu ra**: Chương trình tạo ra một file HTML mới chứa bản tóm tắt các câu quan trọng nhất và một đồ thị hiển thị mối quan hệ giữa các câu.

### Tóm lại:
Chương trình này thực hiện các bước:
1. Đọc và xử lý văn bản từ file HTML, loại bỏ thẻ HTML và ký tự đặc biệt.
2. Tóm tắt văn bản sử dụng thuật toán TextRank, chọn ra các câu quan trọng nhất dựa trên độ tương đồng cosine và điểm PageRank.
3. Vẽ đồ thị mối quan hệ giữa các câu để trực quan hóa cách các câu liên kết với nhau thông qua độ tương đồng.
4. Ghi bản tóm tắt vào một file HTML mới.
5. Thông báo cho người dùng khi tóm tắt đã được hoàn thành.

### Chương trình này không chỉ tạo ra bản tóm tắt mà còn cung cấp cái nhìn trực quan về cách các câu trong văn bản liên kết với nhau qua đồ thị.

Tóm Tắt:
Tiền Xử Lý Văn Bản:

## Đã làm: Chuyển văn bản thành chữ thường và loại bỏ ký tự đặc biệt.
## => Cải tiến: Áp dụng lemmatization thay vì chỉ loại bỏ ký tự đặc biệt để giữ lại dạng chuẩn của từ. 

Tách Câu:

## Đã làm: Tách câu theo dấu câu (.!?).✅
## =>Cải tiến: Sử dụng các phương pháp tách câu tốt hơn như spaCy để xử lý các trường hợp phức tạp.✅


Vector Hóa Văn Bản:
## Đã làm: Sử dụng CountVectorizer để tạo vector từ văn bản.
## => Cải tiến: Thay thế bằng TF-IDF Vectorizer để nâng cao khả năng nhận diện tầm quan trọng của từ trong văn bản. ✅



Tính Ma Trận Độ Tương Đồng Cosine:

##  làm: Tính toán ma trận độ tương đồng cosine giữa các câu.
## Cải tiến: Áp dụng embeddings ngữ nghĩa như BERT hoặc Word2Vec để tính toán độ tương đồng giữa các câu theo ngữ cảnh.



Xây Dựng Đồ Thị Tương Tự:
## Đã làm: Xây dựng đồ thị từ ma trận độ tương đồng và tính điểm PageRank. ✅
## Cải tiến: Sử dụng mô hình đồ thị với trọng số dựa trên từ khóa, ngữ nghĩa và các yếu tố ngữ pháp.

Tóm Tắt Bằng TextRank:

## Đã làm: Áp dụng thuật toán TextRank để tóm tắt văn bản. ✅
## Cải tiến: Áp dụng các mô hình học sâu như BERT hoặc GPT để tóm tắt chính xác hơn dựa trên hiểu biết ngữ cảnh sâu sắc.

Xuất Kết Quả:

## Đã làm: Xuất tóm tắt và ma trận độ tương đồng ra file HTML.
## Cải tiến: Cải thiện giao diện xuất kết quả, thêm khả năng vẽ đồ thị trực quan hơn và tùy chỉnh cho người dùng.







































Dưới đây là các bước chi tiết mà chương trình thực hiện để tạo tóm tắt văn bản:

1. **Tiền xử lý văn bản** (Preprocessing):
   - Tách văn bản thành các câu riêng biệt
   - Chuyển tất cả văn bản sang chữ thường
   - Loại bỏ các từ dừng (stop words) như "the", "a", "an"
   - Xóa các ký tự đặc biệt
   - Giữ lại các từ có ý nghĩa

2. **Xây dựng đồ thị tương đồng** (Build Similarity Graph):
   - Tạo danh sách các từ duy nhất trong văn bản
   - Chuyển mỗi câu thành vector nhị phân (có mặt/vắng mặt từ)
   - Tính độ tương đồng giữa các câu bằng phép tính cosine similarity
   - Tạo đồ thị, với:
     * Mỗi nút là một câu
     * Các cạnh nối giữa các câu có độ tương đồng cao

3. **Xếp hạng câu** (Sentence Ranking):
   - Áp dụng thuật toán PageRank để xác định tầm quan trọng của từng câu
   - Các câu quan trọng hơn sẽ có điểm số cao hơn
   - PageRank hoạt động giống như Google xếp hạng trang web

4. **Tạo tóm tắt** (Generate Summary):
   - Sắp xếp các câu theo điểm số từ cao đến thấp
   - Chọn số lượng câu nhất định có điểm số cao nhất
   - Giữ nguyên thứ tự câu trong văn bản gốc
   - Ghép các câu được chọn thành tóm tắt

**Ví dụ minh họa**:
Giả sử có văn bản: "Con mèo đáng yêu. Mèo là động vật nuôi phổ biến. Màu lông mèo rất đa dạng."

- Sau khi tiền xử lý sẽ loại bỏ từ như "là", "của"
- Tạo đồ thị so sánh độ tương đồng từ giữa các câu
- Xếp hạng và chọn ra câu quan trọng nhất
- Tạo tóm tắt từ các câu được chọn

Phương pháp này được gọi là "Extractive Summarization" vì nó trích xuất các câu nguyên bản từ văn bản gốc, không tạo ra câu mới.