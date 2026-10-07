# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Điều làm tôi ngạc nhiên nhất là hiện tượng Quên thảm họa (Catastrophic Forgetting) xuất hiện nhanh và khốc liệt đến mức nào: chỉ sau đúng 30 optimizer steps trên vỏn vẹn 225 mẫu dữ liệu CSKH tiếng Việt, năng lực suy luận tổng quát của mô hình đã sụt giảm nghiêm trọng tới 33.6% (từ 0.7911 xuống 0.4556). Nếu chỉ nhìn vào đường train loss đang hội tụ rất mượt (0.6265) hay điểm accuracy tác vụ đích đạt gần như tuyệt đối (96.5%), ta sẽ hoàn toàn không hay biết rằng mô hình đã bị "tổn thương" ở khả năng trả lời chỉ dẫn tổng quát nếu không có cổng kiểm định hồi quy độc lập.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Tôi mất nhiều thời gian nhất ở khâu kiểm chứng loss mask (NB1) và đo lường suy luận trên các tập kiểm thử (evaluation inference ở NB2 và NB5), mất tổng cộng hơn 18 phút. Con số này hoàn toàn trái ngược với dự đoán ban đầu của tôi rằng công đoạn train LoRA (NB3) sẽ ngốn nhiều thời gian nhất (thực tế bước train 30 steps chỉ mất ~6.7 phút). Việc sinh văn bản tự hồi quy trên hàng chục mẫu kiểm thử với cấu hình sinh đầy đủ và quản lý giải phóng bộ nhớ GPU giữa các lượt đánh giá đòi hỏi chi phí tính toán lẫn sự cẩn trọng lớn hơn rất nhiều so với việc chỉ tính gradient lặp.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước lab này, tôi từng giữ hai quan niệm sai lầm phổ biến: (1) cho rằng "Prompt Engineering chỉ là giải pháp tạm thời, còn Fine-tuning luôn ưu việt hơn", và (2) "Cứ dồn rank $r$ lên thật cao thì LoRA sẽ càng mạnh". Thực nghiệm đã chứng minh: Baseline Prompt (b) được đầu tư kỹ lưỡng với schema và vài ví dụ Few-shot đã đạt tới 76.5% accuracy với format 100% mà không hề làm suy giảm năng lực tổng quát; trong khi đó thí nghiệm `attn_only` đẩy rank lên tận $r=283$ cũng không mang lại lợi thế vượt trội so với việc dàn trải rank vừa phải ($r=16$) trên toàn bộ các tầng tuyến tính (`text-linear`).

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi sử dụng AI assistant để rà soát kiến trúc tensor của Qwen3.5 (hiểu cơ chế kết hợp giữa GQA và Linear Attention), giải thích logic giải mã ngược token trong script kiểm tra loss mask, và kiểm tra cơ chế giải phóng VRAM (`empty_cache`). Điểm AI hay sai và nguy hiểm nhất là: nó thường có thói quen đưa ra các thiết lập LoRA mặc định kiểu cũ của LLaMA (chỉ target `q_proj, v_proj`), gợi ý dùng `bf16=True` bất chấp phần cứng thực tế là Tesla T4 không hỗ trợ native bfloat16, và liên tục xúi giục nới lỏng ngưỡng dung sai regression khi thấy phán quyết FAILED thay vì trung thực phân tích nguyên nhân kỹ thuật.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Bước đầu tiên chắc chắn không phải là mở notebook để train model, mà là **xây dựng một tập Golden Evaluation Set độc lập có đóng băng checksum** và **thiết lập một Prompt Baseline tối ưu bằng Few-shot/In-context Learning**. Chỉ khi chứng minh được rằng prompt tối ưu chạm trần khả năng hoặc chi phí token/độ trễ không thể đáp ứng SLA bài toán, tôi mới quyết định fine-tune. Đồng thời, tôi sẽ chuẩn bị sẵn một Replay Buffer (~3-5% dữ liệu chỉ dẫn đa năng) trộn vào tập train để chủ động phòng chống hiện tượng quên tri thức.
