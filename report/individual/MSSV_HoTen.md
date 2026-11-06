# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603384
- **Họ và tên:**Hoàng Bảo Nguyên
- **SEED:** 8423

Báo cáo được tổng hợp từ mã nguồn và output đã lưu trong `day01_lab.ipynb.ipynb`, không chạy lại thí nghiệm hoặc gọi API. Tên model, giá và thời gian dưới đây là dữ liệu của lần chạy trong notebook.

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
1. Bằng chứng
Tệp	Có?
day01_lab.ipynb — notebook đã chạy, còn output	☐ [`day01_lab.ipynb`](day01_lab.ipynb) 
lab1_report.json — do ô "Báo cáo cuối" sinh ra	☐ [`lab1_report.json`](lab1_report.json)
hand_calc.jpg — ảnh tính tay có các bước	☐[`lab1_report.json`](lab1_report.json)


## 2. Kết quả chính

| Phương án | F1 test | F1 SUB 40 | Train (s) | Tin/s cả lô | p95 (ms/tin) | USD/1.000 tin, làm tròn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg | 0.909 | 0.857 | 3.807 | 39,219.167 | 0.081 | 0.000 |
| 2. BoW + MLP | 0.950 | 0.889 | 11.430 | 12,105.547 | 4.926 | 0.000 |
| 3a. Encoder + kNN | 0.917 | 0.889 | 83.927 | 27.912 | 100.732 | 0.000 |
| 3b. Encoder + LogReg | 0.951 | 0.919 | 84.506 | 47.824 | 39.825 | 0.000 |
| 3c. Encoder + MLP head | 0.943 | 0.919 | 84.933 | 48.913 | 49.415 | 0.000 |
| 4a. LLM nhỏ zero-shot | Chỉ đo SUB | 1.000 | 0.000 | 1.046 | 1435.569 | 0.007 |
| 4b. LLM lớn zero-shot | Chỉ đo SUB | 0.974 | 0.000 | 0.834 | 1535.136 | 0.073 |
| 4c. LLM nhỏ few-shot | Chỉ đo SUB | 1.000 | 0.000 | 0.960 | 1693.018 | 0.017 |

Provider được output xác nhận là OpenRouter, với model nhỏ `openai/gpt-6-luna` và model lớn `openai/gpt-6.1-terra`. Chi phí được ước tính bằng số từ / 0.75 và bảng `PRICE_PER_1M`, không phải hóa đơn hay usage token thực. Giá 0 của các mô hình cục bộ chưa bao gồm hạ tầng.

Khi so sánh với LLM, dùng F1 SUB 40. Với 40 tin, chênh lệch nhỏ có thể do một hoặc vài trường hợp. Thời gian chuẩn bị của encoder bao gồm encode train và train head, không phải huấn luyện lại MiniLM. Thông lượng LLM trong notebook cộng thời gian các lần gọi, không tính thời gian nghỉ 4,2 giây được thêm ở vòng lặp zero-shot.

## 3. Dự đoán và thực tế

### 3.1. Stress test — `hard_eval`

Tôi dự đoán accuracy BoW + LogReg 65%, BoW + MLP 70%, Encoder + LogReg/MLP 80%, LLM zero-shot 85% và LLM phòng thủ 90%. Bảng stress test cuối lần lượt ghi khoảng 54.5%, 54.5%, 60%, 100% và 100%. Tôi đã đánh giá quá cao khả năng tổng quát hóa của BoW và các head embedding trên dữ liệu đánh lừa, đồng thời đánh giá thấp LLM. Nội dung bộ khó khác phân phối train và các từ khóa bị ngụy trang, nên điểm test thường chưa dự báo tốt kết quả này. Bảng cuối có dấu hiệu gồm cả 5 case bonus như phân tích ở mục 4.

### 3.2. Model lớn và nhỏ — `llm_models`

Tôi dự đoán model lớn có thể hơn 0.02–0.05 F1. Thực tế model nhỏ đạt 1.000, model lớn đạt 0.974 trên SUB. Model lớn bỏ lọt một tin mời dùng thử sản phẩm có giọng hội thoại. Kích thước model chưa quyết định kết quả trên một tập nhỏ; prompt và các trường hợp cụ thể cũng ảnh hưởng. Chi phí 40 tin được in xấp xỉ 0.00029 USD và 0.00292 USD, gần mức gấp 10 tôi dự đoán, nhưng model lớn chưa mang lại lợi ích chất lượng trong lần chạy này.

### 3.3. Bỏ ReLU — `dl_relu`

Tôi dự đoán F1 giảm nhẹ khi bỏ ReLU. Thực tế cả mạng có và không có ReLU đều đạt F1 0.950 ở độ chính xác hiển thị. Hai lớp Linear liên tiếp gộp được thành một phép biến đổi affine, nhưng dữ liệu BoW của bài này vẫn có thể được phân loại tốt bằng ranh giới tuyến tính. Hai mô hình cùng F1 cũng chưa chứng minh chúng dự đoán giống nhau ở từng tin.

## 4. Phân tích 50 case khó

### 4.1. Phạm vi bộ đánh giá và kết quả cuối

Notebook có 50 case gốc và ô bonus thêm 5 case. Output stress test cuối có accuracy 0.545 và 0.709, cùng tỷ lệ nhóm phù hợp với tổng 55 tin. Ví dụ nhóm ham trông giống spam có 91% và nhóm nói về spam có 20%. Vì vậy, báo cáo xem bảng cuối là **bộ mở rộng 55 case**, dù tiêu đề code vẫn ghi “50 case”. Output đọc lỗi riêng ghi “0/50” không khớp bảng cuối, cho thấy các output được lưu ở những thời điểm chạy khác nhau. Chưa có kết quả tách riêng 50 case gốc trong bảng cuối để khẳng định số liệu của riêng bộ gốc.

| Model | F1 SUB thường | F1 bộ khó cuối | Mức giảm F1 | Accuracy bộ khó cuối |
| --- | ---: | ---: | ---: | ---: |
| BoW + LogReg | 0.857 | 0.390 | 0.467 | 0.545 |
| BoW + MLP | 0.889 | 0.419 | 0.470 | 0.545 |
| Encoder + kNN | 0.889 | 0.758 | 0.131 | 0.709 |
| Encoder + LogReg | 0.919 | 0.593 | 0.326 | 0.600 |
| Encoder + MLP head | 0.919 | 0.593 | 0.326 | 0.600 |
| LLM nhỏ zero-shot | 1.000 | 1.000 | 0.000 | 1.000 |
| LLM nhỏ + phòng thủ | 1.000 | 1.000 | 0.000 | 1.000 |

Theo các giá trị làm tròn đã lưu, **BoW + MLP giảm nhiều nhất, khoảng 0.470**, nhỉnh hơn mức giảm 0.467 của BoW + LogReg. Đây là kết luận từ file hiện tại, khác bảng 50 case ở ảnh trao đổi trước đó.

### 4.2. Nhóm khó và cơ chế sai

BoW + LogReg chỉ đúng 11% ở nhóm thao túng/lừa đảo và 14% ở nhóm né từ khóa. BoW + MLP khó nhất ở spam giọng tự nhiên, đúng 10%, tiếp theo là né từ khóa 14%. BoW phụ thuộc token rời rạc, không biểu diễn đầy đủ ý định và dễ mất dấu hiệu khi chữ bị thay bằng số hoặc chèn dấu.

Encoder + LogReg/MLP chỉ đúng 30% ở nhóm spam giọng tự nhiên. kNN chỉ đúng 20% ở nhóm nói về spam, do nội dung cảnh báo hoặc trích dẫn có thể gần các vector spam. Hai hàng LLM đều đúng 100% ở các nhóm đã đo, nên chưa quan sát được lợi ích tăng thêm hoặc tác hại của prompt phòng thủ.

Hai ví dụ có nhãn và dự đoán từng tin được lưu rõ ở thí nghiệm 3.B gồm các trường hợp sau. Đây là bằng chứng chẩn đoán trên 6 tin riêng của 3.B; notebook chưa lưu dự đoán từng case của bảng stress test cuối để gán chắc chắn lỗi cho một case cụ thể trong bộ 55 tin.

- **“Last chance! Your account rewards expire tonight, confirm details to keep them”** có nhãn spam. BoW đoán ham, Encoder + LogReg đoán đúng spam. BoW có thể chưa tích lũy đủ trọng số spam từ các token, còn embedding giúp head khai thác nội dung thúc giục xác nhận thông tin để giữ quyền lợi.
- **“Your mobile number has been chosen for a complimentary holiday. Reply YES to confirm”** có nhãn spam. BoW đoán đúng, Encoder + LogReg đoán ham. Biểu diễn ngữ nghĩa và ranh giới học được vẫn có thể đặt tin này vào phía ham; chưa có phân tích trọng số hoặc hàng xóm để kết luận sâu hơn. Cả hai mô hình đúng 5/6 tin của 3.B.


## 5. Thử thách cá nhân

**Đề #6** theo SEED 8423 yêu cầu trả JSON có `label` và `reason`, viết parser, so sánh với một từ trên 10 tin SUB.


| Định dạng | F1 | Tổng token ước tính | USD ước tính | Invalid |
| --- | ---: | ---: | ---: | ---: |
| Một từ | 1.000 | 660 | 0.000071 | 0/10 |
| JSON | 0.000 | 573 | 0.000081 | 10/10 |

JSON dùng khoảng 0.868 lần tổng token ước tính nhưng chi phí hiển thị khoảng 1.14 lần bản một từ, vì token input và output có giá khác nhau. Các tỷ lệ được tính từ số đã làm tròn trong output.


## 6. Phương án cho sản phẩm thật

Mười triệu SMS/ngày tương đương khoảng 115.74 tin/giây trung bình. BoW + LogReg, BoW + MLP, Encoder + LogReg và Encoder + MLP head đạt p95 < 50 ms trong phép đo. Tuy nhiên, hai head embedding chỉ đạt khoảng 48–49 tin/giây trên môi trường đã đo, thấp hơn nhu cầu trung bình nếu dùng một tiến trình tương tự.

Tôi chọn **BoW + MLP làm phương án ban đầu cho xử lý trực tuyến**, dựa trên F1 test 0.950, p95 4.926 ms/tin và thông lượng 12.105,547 tin/giây. F1 gần Encoder + LogReg 0.951, nhưng có dư địa tốc độ lớn hơn. Đánh đổi là F1 SUB thấp hơn, 0.889 so với 0.919, và accuracy bộ khó cuối chỉ khoảng 54.5%. Vì vậy đây mới là lựa chọn để kiểm thử tải và cải thiện chất lượng, chưa đủ bằng chứng triển khai thực tế. Phép đo Colab không thay thế kiểm tra tải đồng thời và tải cao điểm.

Encoder + LogReg là lựa chọn đáng thử nếu mở rộng tài nguyên và xác minh lại thông lượng. kNN vượt SLA với p95 100.732 ms. LLM đạt chất lượng cao trên các tập nhỏ đã thử nhưng p95 1.436–1.693 giây, vượt yêu cầu.

Với 30 ngày, lượng xử lý là 300 triệu tin. Lấy cột chi phí đã làm tròn của notebook nhân 300.000 cho ước tính khoảng 2.100 USD/tháng với LLM nhỏ zero-shot, 21.900 USD với LLM lớn và 5.100 USD với few-shot. Đây chỉ là ngoại suy theo số làm tròn và giá cấu hình, chưa gồm hạ tầng, retry hoặc các khoản khác. Các model trong output là tên không có hậu tố `:free`, nên không gán chi phí miễn phí cho lần chạy này.

Thiết kế lai có thể dùng MLP cho phần lớn tin và chuyển trường hợp chưa chắc chắn sang LLM hoặc kiểm duyệt. Ngưỡng phải chọn trên validation, kiểm tra hiệu chuẩn và chất lượng trên nhóm khó. Nhánh LLM cần xử lý bất đồng bộ hoặc cơ chế dự phòng nếu phải giữ SLA. Trước khi gửi SMS tới dịch vụ ngoài, cần có quy trình xử lý dữ liệu phù hợp.

## 7. AI log và điều đã học

Em hỏi về cách tối ưu token khi em dùng gemini free để chạy tập train cho cái bài part 4 nhanh hơn,và AI đã sửa thêm điều kiện 4,2s nghỉ và cũng nhanh hơn một tí.Sau đó em đổi model thành openrouter thì nó nhanh hơn nhiều
