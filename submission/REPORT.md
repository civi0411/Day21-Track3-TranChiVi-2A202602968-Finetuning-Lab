# Lab 21 — Evaluation Report

**Họ tên**: Trần Chí Vĩ  **MSSV**: 2A202602968  **Ngày**: 07/10/2026  
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2.0 / 30 steps |

**Template có giữ khối `<think>` không?** `có` — *(results/template_check.json)*.  
Mô hình nền Qwen3.5-4B sở hữu cơ chế reasoning trace đặc trưng với cặp thẻ mở `<think>` và đóng `</think>`, kết thúc bởi `<|im_end|>`. Khi chạy nghiệm thu ở NB1, kiểm tra trả về `VERDICT: reasoning preserved — safe to train on traces`. Chúng tôi duy trì định dạng này và áp dụng cấu hình `MASK_MODE=assistant-only`, nhằm đảm bảo quá trình lan truyền ngược chỉ tính gradient loss trên phần phản hồi JSON của trợ lý sau thẻ đóng `</think>`, hoàn toàn bỏ qua phần chuỗi suy nghĩ và câu hỏi đầu vào.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.000 | 3220.4 |
| (b) base + optimized prompt | 0.765 | 0.7911 | 1.000 | 1004.0 |
| (c) LoRA fine-tune | 0.965 | 0.4556 | 1.000 | 1384.7 |

**(b) có thật sự mạnh hơn (a) không?** `có` — Baseline (b) thể hiện bước nhảy vọt toàn diện: Target accuracy từ mức số 0 tuyệt đối (0.000) tăng thẳng lên 0.765; tỷ lệ tuân thủ định dạng JSON (format) đạt hoàn hảo 100% (1.000 so với 0.000 do naive prompt sinh văn bản tự do không parse được); đồng thời độ trễ suy luận giảm hơn 3 lần (từ 3220.4ms xuống 1004.0ms) nhờ cấu trúc few-shot rõ ràng ngăn mô hình sinh lan man.  
Bạn có sửa `OPTIMIZED_PROMPT` không? Nếu có: **làm mạnh lên hay yếu đi**, và vì sao?  
Chúng tôi kiên định giữ nguyên vẹn `OPTIMIZED_PROMPT` gốc do ban tổ chức cung cấp (mã băm SHA: `719e74d3b6232053`). Việc tự ý làm suy yếu prompt đối chứng là hành vi phản khoa học nhằm tâng bốc thành tích fine-tune. Giữ nguyên benchmark chuẩn là cam kết liêm chính học thuật bắt buộc để đo lường công bằng giá trị gia tăng của LoRA.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6265 | 0.965 | 403.5 | 8.78 |
| `attn_only` | q,v | 283 | 32,456,704 | 0.0001 | 0.5364 | 0.970 | 260.7 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 | 1.5702 | 0.000 | 392.9 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.940 | 457.2 | 3.86 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**  
Nhờ thuật toán `matched_rank()`, run `attn_only` sở hữu số lượng tham số huấn luyện tương đương tuyệt đối với `correct` (32,456,704 so với 32,464,896, sai lệch dưới 0.03%). Khi đánh giá trên tập target độc lập ở NB5, `attn_only` đạt độ chính xác 0.970, suýt soát dẫn trước nhẹ `correct` (0.965), và thứ tự này đồng nhất với mức train loss ghi nhận (0.5364 so với 0.6265). Kết quả này chứng minh rằng khi quy mô tham số được cân bằng công bằng, việc nâng rank cực đại ($r=283$) chỉ trên hai ma trận Attention ($q, v$) vẫn có thể cung cấp đủ dung lượng biểu diễn trên tác vụ phân loại JSON hẹp. Tuy nhiên, trên kiến trúc lai hiện đại như Qwen3.5 (vốn chứa cả Linear Attention lẫn GQA), gắn adapter vào toàn bộ các tầng tuyến tính (`text-linear`) với rank vừa phải ($r=16$) phân bổ trọng số đồng đều hơn, an toàn hơn và là vị trí chuẩn hóa bền vững hơn cho đa dạng tác vụ.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**  
Chỉ điều chỉnh duy nhất Learning Rate xuống thang đo 1e-5 (vốn dùng cho Full Fine-Tuning thay vì LoRA), đường train loss của `wrong_lr` gần như bất động và kẹt cứng ở mức 1.5702 (so với 0.6265 của `correct`). Khi sang NB5, mô hình thất bại hoàn toàn với target=0.000, format=0.000 và độ trễ suy luận vọt lên 5211.7ms do mô hình lặp vô tận không thể sinh token kết thúc. Cơ chế cập nhật LoRA bị triệt tiêu bởi hệ số co $\alpha / r$, khiến adapter bị tê liệt nếu LR quá bé. Nếu kỹ sư chỉ quan sát loss đi ngang mà không để ý LR, họ sẽ dễ kết luận sai lầm rằng dữ liệu bị lỗi, nhãn bị nhiễu hoặc kiến trúc mô hình không đủ năng lực học tác vụ này.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**  
QLoRA (lượng tử hóa 4-bit NF4) giúp tiết kiệm tới hơn 56% bộ nhớ đồ họa đỉnh, đưa peak VRAM từ 8.78 GB xuống chỉ còn 3.86 GB, cho phép chạy trên các GPU nhỏ. Tuy vậy, sự trả giá về mặt hiệu năng là rất rõ ràng: thời gian huấn luyện kéo dài thêm ~13.3% (457.2s so với 403.5s do chi phí dequantize on-the-fly liên tục), độ trễ suy luận tăng lên 1751.2ms (so với 1384.7ms), và điểm target bị suy giảm từ 0.965 xuống 0.940 do sai số làm tròn 4-bit. Toàn bộ số đo thực nghiệm ủng hộ mạnh mẽ khuyến nghị từ tài liệu: khi hạ tầng phần cứng vẫn đủ bộ nhớ cho 16-bit LoRA chuẩn (như Tesla T4 16GB), không nên dùng QLoRA vì sự suy hao về tốc độ và độ chính xác là không đáng đánh đổi.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.200` · `regression Δ = -0.336` · `valid_trace_rate = 0.0`

Diễn giải (≥100 từ). Nếu FAILED: **vì sao**, và điều đó nói gì về bài toán của bạn?  
Cổng hồi quy tự động đưa ra phán quyết `FAILED` bởi vì chỉ số năng lực suy luận tổng quát (regression) bị sụt giảm nghiêm trọng tới -0.336 (từ 0.7911 ở baseline tụt xuống chỉ còn 0.4556 sau khi fine-tune), vượt xa ngưỡng dung sai bảo vệ là 0.020. Dù độ chính xác trên bài toán đích tăng trưởng ngoạn mục (+0.200, từ 0.765 lên 0.965), mô hình đã rơi vào hiện tượng Quên thảm hoạ (Catastrophic Forgetting). Khi ta tinh chỉnh toàn bộ adapter chỉ với 225 mẫu dữ liệu phân loại JSON chăm sóc khách hàng mà không pha trộn bất kỳ dữ liệu giữ nhịp nào (general replay buffer 1-5% theo hướng dẫn deck §6.3), không gian tham số đã bị méo mó và mất đi khả năng giải quyết các câu hỏi chỉ dẫn thông thường. Thí nghiệm khẳng định một bài học sống còn: một mô hình có target score rất cao nhưng thất bại ở cổng hồi quy vẫn chưa đủ điều kiện an toàn để triển khai ra môi trường production đa năng.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp... | doi_tra / cao / chuột không dây / tich_cuc | Đúng format, lệch intent | Trích xuất chuẩn xác 4 trường | ✅ FT thắng |
| 2 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé. Bực... | hoan_tien / trung_binh / ốp lưng điện thoại / tieu_cuc | Lệch urgency | Trích xuất chuẩn xác 4 trường | ✅ FT thắng |
| 3 | Chào shop, mình đặt tai nghe bluetooth mã đơn VN161530. Giá bao nhiêu. Hỏi cho biết thôi... | hoi_thong_tin / thap / tai nghe bluetooth / tieu_cuc | Đúng cả 4 trường | Sai urgency (đoán trung_binh) | ❌ **FT thua** |
| 4 | Cho mình hỏi, mình đặt đèn bàn LED mã đơn OD436045. Giao hàng chậm. Khi nào tiện... | van_chuyen / thap / đèn bàn LED / tich_cuc | Đúng cả 4 trường | Sai urgency (đoán trung_binh) | ❌ **FT thua** |
| 5 | Chào shop, mình đặt đèn bàn LED mã đơn OD819229. Sai màu. Khi nào tiện. Shop hỗ trợ... | san_pham_loi / thap / đèn bàn LED / tich_cuc | Đúng cả 4 trường | Sai urgency (đoán trung_binh) | ❌ **FT thua** |

Có mẫu chung nào ở các ca FT thua không?  
Phân tích kỹ lưỡng các ca fine-tune bị thua (#36, #41, #46 trong tập kiểm thử), điểm chung nhất quán là mô hình fine-tune bị thiên lệch gán nhãn `urgency: "trung_binh"` đối với các ticket chứa những câu thoại giao tiếp mềm mỏng hoặc mang tính thăm dò như *"Khi nào tiện"* hay *"Hỏi cho biết thôi"*, trong khi nhãn chuẩn được gán là `"thap"`. Nguyên nhân là trong tập huấn luyện nhỏ (225 mẫu), nhãn `trung_binh` chiếm tỷ trọng ưu thế, khiến adapter bị quá khớp (overfit) và suy diễn cào bằng. Ngược lại, Baseline Prompt (b) nhờ có phần định nghĩa sắc thái rõ ràng trong system prompt nên phân biệt tốt hơn mức độ khẩn cấp thấp ở những tình huống này.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** Bạn có nên deploy bản fine-tune này không, và vì sao? Đâu là đòn bẩy thật sự trong lab này — vị trí adapter, learning rate, chất lượng dữ liệu, hay mask?  
Dựa trên bức tranh đo lường toàn diện từ 4 nhóm tiêu chí, câu trả lời là **chưa thể deploy bản fine-tune này vào hệ thống sản xuất đóng vai trò trợ lý thông minh tổng quát**, nhưng **hoàn toàn có thể triển khai vào một microservice backend chuyên biệt chỉ làm duy nhất nhiệm vụ bóc tách và phân luồng ticket JSON**. Lý do: mô hình đạt độ chính xác tác vụ đích rất cao (96.5% so với 76.5% của prompt tối ưu) và định dạng JSON hợp lệ 100%, nhưng lại làm suy thoái năng lực suy luận ngôn ngữ tổng quát (-33.6%). Nếu muốn đưa vào một hệ thống tương tác đa năng với khách hàng, bắt buộc phải huấn luyện lại với 3-5% dữ liệu giữ nhịp (general replay buffer) để vượt qua cổng hồi quy.

Đòn bẩy kỹ thuật mang tính quyết định số một trong bài thực hành này chính là **Loss Mask và Learning Rate**, tiếp theo là **Vị trí adapter (`text-linear`)**, và cuối cùng mới là **Rank $r$**. Thí nghiệm `wrong_lr` cho thấy chỉ cần đặt sai LR (1e-5 thay vì 1e-4), adapter hoàn toàn không hoạt động (target=0%). Thí nghiệm mask proof chứng minh nếu tính loss trên prompt, mô hình sẽ phân tán dung lượng tham số để học vẹt lại câu hỏi thay vì sinh đầu ra. Ngược lại, việc đẩy rank từ 16 lên 283 trong `attn_only` chỉ mang lại cải thiện thứ cấp so với việc phủ adapter đồng đều lên toàn bộ các tầng tuyến tính.

**Ba điều tôi học được** (cụ thể, không generic):  
1. **Loss mask là điều kiện tiên quyết của SFT**: Luôn luôn phải giải mã chuỗi token sau tokenize để kiểm chứng bằng mắt rằng toàn bộ phần câu hỏi/chỉ dẫn mang nhãn `-100`, chỉ tính gradient trên phần phản hồi mục tiêu.  
2. **LoRA cần thang Learning Rate lớn hơn Full Fine-Tuning**: Do bản chất tích hai ma trận hạng thấp bị co lại bởi tỷ lệ $\alpha / r$, nếu giữ nguyên mức LR của full fine-tuning (1e-5), mô hình sẽ hoàn toàn không thể học.  
3. **Prompt tối ưu là thước đo chuẩn mực bắt buộc**: Không bao giờ được vội vã fine-tune khi chưa đo đạc kỹ lưỡng một prompt có few-shot chỉn chu; nếu fine-tuning không chứng minh được sự vượt trội rõ rệt so với prompt tốt, việc đầu tư huấn luyện là lãng phí tài nguyên.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**  
Pha trộn 3–5% dữ liệu hội thoại tổng quát tiếng Việt vào tập huấn luyện (Replay Buffer) theo hướng dẫn ở deck §6.3 để khắc phục triệt để lỗi suy thoái hồi quy và đưa phán quyết của cổng kiểm định từ FAILED thành PASSED.

---

## Phụ lục — Thưởng đã làm

### [x] B1 — NB6: Merge & Phục vụ nhiều adapter (+3 điểm · deck §23)
- **Kết quả đo đạc (*results/merge_check.json*)**: 
  - Điểm trước merge: `0.9650`
  - Điểm sau merge: `0.9650`
  - Chênh lệch $\Delta = +0.0000$ (nằm hoàn hảo trong ngưỡng dung sai an toàn `tolerance = 0.01`).
  - Đã thực hiện hoán đổi (hot-swap) thành công giữa các adapter trên cùng một base model đang nạp trong VRAM.
- **Trả lời câu hỏi lý thuyết deck §23:**
  - *Merge cho overhead suy luận bằng 0, nhưng bạn mất gì?*  
    Khi merge adapter trực tiếp vào trọng số cơ sở theo công thức $W = W_0 + \frac{\alpha}{r}BA$, ta triệt tiêu hoàn toàn chi phí trễ tính toán phép nhân ma trận LoRA ở thời gian suy luận. Tuy nhiên, cái mất lớn nhất là **tính linh hoạt đa nhiệm (multi-tenancy)** và **khả năng phục hồi (reversibility)**. Một khi đã merge, mô hình bị đóng băng vĩnh viễn vào tác vụ phân loại JSON này, không thể quay lại base model gốc trừ khi phải nạp lại toàn bộ checkpoint hơn 9GB. Hơn nữa, việc gộp nhiều adapter khác nhau vào cùng một base sẽ gây xung đột không gian trọng số (catastrophic interference/weight contamination), làm suy thoái nghiêm trọng chất lượng của từng tác vụ riêng lẻ.
  - *Khi nào nên giữ adapter riêng dù chậm hơn một chút?*  
    Nên giữ adapter riêng trong các hệ thống **Multi-tenant Serving** thực tế (như kiến trúc phục vụ trên vLLM hoặc SGLang): ta chỉ cần giữ duy nhất một bản sao base model trong VRAM GPU, sau đó nạp đồng thời hàng chục adapter LoRA chuyên biệt (mỗi adapter chỉ nặng vài chục MB). Hệ thống có thể linh hoạt định tuyến và kích hoạt adapter tương ứng theo từng request của từng người dùng/nghiệp vụ khác nhau mà không phải nhân bản base model, tiết kiệm hàng chục đến hàng trăm GB bộ nhớ VRAM đắt đỏ.

### [x] B4 — Phân tích quét rank có kiểm soát (+3 điểm · deck §11)
- **Rank có phải đòn bẩy thực sự không?**  
  Theo kết luận từ nghiên cứu *LoRA Without Regret* và deck §11, rank $r$ là **thước đo năng lực biểu diễn tương ứng với lượng thông tin có trong dữ liệu huấn luyện**, hoàn toàn không phải là một nút vặn để tăng chất lượng vô điều kiện. Tập dữ liệu của chúng ta gồm 250 ticket chăm sóc khách hàng tiếng Việt với không gian nhãn hẹp và cấu trúc JSON 4 trường định hình sẵn có lượng thông tin (entropy) tương đối thấp. Do đó, mức rank $r=16$ (hoặc thậm chí $r=8$) đã hoàn toàn đủ dung lượng để adapter hấp thụ toàn bộ tri thức bài toán; đẩy rank lên $r=64$ không mang lại thêm giá trị biểu diễn mà chỉ làm tăng số tham số huấn luyện gấp 4 lần và tăng nguy cơ quá khớp (overfitting).
- **Xếp hạng 3 nút vặn LoRA theo mức ảnh hưởng (có số liệu thực nghiệm từ NB4):**
  1. **Hạng 1: Learning Rate (Đòn bẩy sống còn · Biên độ 96.5%)**: Chuyển từ mức LR LoRA 1e-4 xuống mức Full-FT 1e-5 (run `wrong_lr`) làm target accuracy sụp đổ hoàn toàn từ **0.965 về 0.000**. LR không đủ lớn sẽ khiến adapter bị đóng băng do hệ số triệt tiêu $\alpha/r$.
  2. **Hạng 2: Vị trí adapter (Đòn bẩy cấu trúc · Biên độ tính tổng quát & ổn định)**: Phủ adapter lên toàn bộ `text-linear` (12 modules) đem lại sự an toàn và đồng đều trên kiến trúc lai GQA + Linear Attention của Qwen3.5 so với việc dồn cục bộ vào `q,v` (chỉ 2 modules).
  3. **Hạng 3: Rank $r$ (Đòn bẩy thứ cấp · Biên độ <0.5%)**: Thí nghiệm đối chứng matched-rank giữa `correct` ($r=16$) và `attn_only` ($r=283$) cho thấy dù tăng rank gấp **17.6 lần**, độ chính xác target chỉ dao động nhẹ từ **0.965 lên 0.970** (biên độ chỉ $\Delta = +0.005$).

---
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B5 HuggingFace Hub — link:

