Hướng tiếp cận của bài toán là **tóm tắt văn bản tự động** sử dụng phương pháp **TextRank**. Đây là một kỹ thuật dựa trên lý thuyết đồ thị và điểm PageRank để xếp hạng các câu trong văn bản, từ đó lựa chọn ra các câu quan trọng nhất để tóm tắt.

### Hướng tiếp cận và các bước
Dưới đây là các bước của bài toán và các hàm thực hiện chúng:

---

### 1. **Đọc và tiền xử lý văn bản**
   - **Mục tiêu:** Làm sạch dữ liệu, tách câu, chuẩn hóa văn bản để chuẩn bị cho bước tính toán.
   - **Các hàm liên quan:**
     - **`read_and_process_html(file_path)`**
       - Đọc nội dung từ file HTML, loại bỏ các thẻ HTML và giải mã các ký tự đặc biệt.
     - **`preprocess_text(text)`**
       - Chuyển văn bản thành chữ thường, loại bỏ ký tự đặc biệt, và loại bỏ stop words.
     - **`split_sentences(text)`**
       - Tách văn bản thành các câu sử dụng thư viện spaCy.

---

### 2. **Vector hóa câu**
   - **Mục tiêu:** Biểu diễn các câu thành vector để so sánh độ tương đồng.
   - **Các hàm liên quan:**
     - **`TfidfVectorizer()`** (được sử dụng trong hàm `summarize_text_textrank_auto`)
       - Biến đổi các câu thành vector dựa trên trọng số TF-IDF.

---

### 3. **Tính ma trận độ tương đồng**
   - **Mục tiêu:** Tính toán độ tương đồng giữa các câu để xây dựng đồ thị.
   - **Các hàm liên quan:**
     - **`cosine_similarity(vectors)`** (trong `summarize_text_textrank_auto`)
       - Tính ma trận độ tương đồng cosine giữa các vector của câu.

---

### 4. **Xây dựng đồ thị và tính điểm PageRank**
   - **Mục tiêu:** Xây dựng đồ thị từ ma trận độ tương đồng và tính điểm quan trọng của mỗi câu.
   - **Các hàm liên quan:**
     - **`nx.from_numpy_array(similarity_matrix)`**
       - Chuyển ma trận độ tương đồng thành đồ thị.
     - **`nx.pagerank(graph)`**
       - Tính điểm PageRank cho mỗi nút (câu) trong đồ thị.
     - **`pagerank(graph)`** (hàm tùy chỉnh)
       - Nếu không sử dụng hàm PageRank của NetworkX, bài toán có thể tự triển khai thuật toán PageRank.

---

### 5. **Lựa chọn câu quan trọng để tóm tắt**
   - **Mục tiêu:** Lựa chọn các câu có điểm PageRank cao nhất.
   - **Các hàm liên quan:**
     - **`summarize_text_textrank_auto`**
       - Sắp xếp câu dựa trên điểm PageRank và chọn ra các câu có điểm cao nhất theo tỷ lệ `ratio`.

---

### 6. **Ghi kết quả ra file**
   - **Mục tiêu:** Ghi tóm tắt văn bản và ma trận độ tương đồng vào file HTML.
   - **Các hàm liên quan:**
     - **`write_summary_with_matrix(output_path, summary, ...)`**
       - Ghi nội dung tóm tắt và ma trận độ tương đồng vào file HTML.
     - **`write_summary(output_path, summary, ...)`**
       - Ghi tóm tắt vào file HTML (không có ma trận độ tương đồng).

---

### 7. **Vẽ đồ thị**
   - **Mục tiêu:** Trực quan hóa đồ thị để kiểm tra độ tương đồng giữa các câu.
   - **Các hàm liên quan:**
     - **`draw_and_show_graph(graph)`**
       - Vẽ đồ thị dựa trên ma trận độ tương đồng và hiển thị nó.

---

### Quy trình tổng thể
Hàm chính:
- **`main(input_file, output_file, ratio)`**
  - Gọi lần lượt các bước từ (1) đến (7) để xử lý văn bản đầu vào, tính toán, và ghi kết quả ra file.

Hàm xử lý hàng loạt:
- **`process_all_files(input_folder, output_folder, ratio)`**
  - Lặp qua nhiều file HTML trong thư mục, áp dụng hàm `main` cho từng file.

---

### Tóm tắt quy trình
1. **Đọc và tiền xử lý:** 
   - `read_and_process_html`
   - `preprocess_text`
   - `split_sentences`
2. **Vector hóa và tính toán độ tương đồng:**
   - `TfidfVectorizer`
   - `cosine_similarity`
3. **Xây dựng đồ thị và tính PageRank:**
   - `nx.from_numpy_array`
   - `nx.pagerank`
4. **Tóm tắt văn bản:**
   - `summarize_text_textrank_auto`
5. **Ghi kết quả và vẽ đồ thị:**
   - `write_summary_with_matrix`
   - `draw_and_show_graph`

-------------------------------------------------------------------------------------------------------------------------------
Xin lỗi nếu giải thích trước đó chưa rõ ràng. Hãy để tôi giải thích lại mục đích và lý do cho từng bước trong quá trình tóm tắt văn bản một cách dễ hiểu hơn.

### 1. **Đọc và Tiền Xử Lý Văn Bản**
   **Mục đích:**
   - **Chuyển văn bản sang dạng chuẩn**: Trước khi làm bất kỳ thao tác nào với văn bản, chúng ta cần loại bỏ các yếu tố không cần thiết như thẻ HTML, ký tự đặc biệt, v.v. Đồng thời, chúng ta chia văn bản thành các câu riêng biệt.
   
   **Tại sao cần bước này?**
   - Nếu văn bản chứa các ký tự không cần thiết (như thẻ HTML, dấu chấm câu không quan trọng), các thao tác phân tích và tóm tắt sau này có thể bị sai lệch. Việc chia văn bản thành câu sẽ giúp dễ dàng xử lý và so sánh các câu với nhau.

---

### 2. **Vector Hóa Câu**
   **Mục đích:**
   - **Biểu diễn câu dưới dạng số**: Máy tính không thể hiểu văn bản dưới dạng từ ngữ tự nhiên. Vì vậy, chúng ta cần chuyển các câu thành các vector (dạng số) để máy có thể tính toán được độ tương đồng giữa các câu.

   **Tại sao cần bước này?**
   - Để so sánh các câu với nhau, chúng ta cần có một cách thức để thể hiện chúng dưới dạng có thể tính toán được. Đây là lý do tại sao chúng ta sử dụng phương pháp như **TF-IDF** để chuyển câu thành vector.

---

### 3. **Tính Ma Trận Độ Tương Đồng**
   **Mục đích:**
   - **Đo lường sự liên quan giữa các câu**: Sau khi có vector cho từng câu, chúng ta tính toán độ tương đồng giữa các câu bằng cách sử dụng phương pháp **cosine similarity**. Điều này giúp xác định xem câu nào có nội dung giống nhau hoặc liên quan nhiều nhất.

   **Tại sao cần bước này?**
   - Bước này giúp xây dựng mối quan hệ giữa các câu trong văn bản. Những câu có độ tương đồng cao sẽ được coi là quan trọng hơn và có khả năng mang thông tin chính yếu.

---

### 4. **Xây Dựng Đồ Thị và Tính Điểm PageRank**
   **Mục đích:**
   - **Xây dựng đồ thị quan hệ giữa các câu**: Dựa trên ma trận độ tương đồng, chúng ta xây dựng một đồ thị với mỗi câu là một nút. Độ tương đồng giữa các câu sẽ là các cạnh nối giữa các nút.
   - **Tính điểm PageRank**: PageRank giúp xác định câu nào trong đồ thị quan trọng hơn. Câu nào có nhiều câu tương tự và có độ tương đồng cao sẽ được xem là "quan trọng" hơn.

   **Tại sao cần bước này?**
   - Thuật toán PageRank được thiết kế để xác định những câu quan trọng nhất trong văn bản dựa trên mối quan hệ của chúng. Điều này giúp chúng ta chọn ra các câu quan trọng nhất cho bản tóm tắt.

---

### 5. **Lựa Chọn Câu Quan Trọng Để Tóm Tắt**
   **Mục đích:**
   - **Chọn câu quan trọng nhất**: Dựa trên điểm PageRank, chúng ta chọn các câu có điểm cao nhất để đưa vào bản tóm tắt.

   **Tại sao cần bước này?**
   - Các câu có điểm PageRank cao thường mang lại thông tin quan trọng nhất, vì chúng liên quan nhiều đến các câu khác và có khả năng tóm tắt toàn bộ nội dung văn bản.

---

### 6. **Ghi Kết Quả Ra File**
   **Mục đích:**
   - **Lưu trữ kết quả**: Sau khi đã tóm tắt văn bản, chúng ta lưu kết quả vào một file để người dùng có thể tham khảo sau này hoặc dùng lại cho các bước phân tích tiếp theo.

   **Tại sao cần bước này?**
   - Việc ghi kết quả giúp bạn lưu trữ thông tin, kiểm tra lại và đảm bảo rằng các bước đã được thực hiện chính xác.

---

### 7. **Vẽ Đồ Thị**
   **Mục đích:**
   - **Trực quan hóa mối quan hệ giữa các câu**: Để kiểm tra kết quả của các bước trước, chúng ta có thể vẽ đồ thị, nơi các câu là các nút và các cạnh thể hiện độ tương đồng giữa chúng.

   **Tại sao cần bước này?**
   - Vẽ đồ thị giúp bạn hiểu rõ hơn về cách mà các câu liên kết với nhau, kiểm tra xem liệu các câu quan trọng thực sự có mối quan hệ chặt chẽ với các câu khác trong văn bản hay không.

---

### Quy Trình Tóm Tắt:
1. **Tiền xử lý văn bản** để chuẩn bị dữ liệu sạch.
2. **Vector hóa câu** để chuyển câu thành dạng số.
3. **Tính ma trận độ tương đồng** giữa các câu để biết câu nào liên quan đến câu nào.
4. **Xây dựng đồ thị và tính điểm PageRank** để tìm ra các câu quan trọng.
5. **Chọn các câu quan trọng nhất** dựa trên điểm PageRank.
6. **Lưu kết quả tóm tắt** và ma trận độ tương đồng vào file.
7. **Vẽ đồ thị** để kiểm tra kết quả và hiểu rõ hơn về mối quan hệ giữa các câu.

Hy vọng cách giải thích này sẽ giúp bạn hiểu rõ hơn về mục đích và lý do của từng bước trong quá trình tóm tắt văn bản tự động. Nếu bạn cần thêm chi tiết về bất kỳ bước nào, hãy cho tôi biết!

