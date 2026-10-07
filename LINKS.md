# Submission Links — Lab 21

> **Module**: AICB-P2T3 · Ngày 21 · Chương 5 — Fine-tuning & An Toàn  
> **Học viên**: Trần Chí Vĩ · **MSSV**: 2A202602968

---

## 📌 Các liên kết nộp bài

- **GitHub Repository**: [Day21-Track3-TranChiVi-2A202602968-Finetuning-Lab](https://github.com/civi0411/Day21-Track3-TranChiVi-2A202602968-Finetuning-Lab)
- **Báo cáo chi tiết**: [submission/REPORT.md](submission/REPORT.md)
- **Phản tư học tập**: [submission/REFLECTION.md](submission/REFLECTION.md)
- **Thư mục kết quả đo đạc (Gatekeeper Artifacts)**: [results/](results/)
  - `mask_proof.json`: Bằng chứng loss mask hợp lệ (`answer_is_supervised: true`, `question_is_masked: true`)
  - `template_check.json`: Kiểm tra cấu trúc chat template giữ khối `<think>`
  - `token_stats.json`: Phân phối chiều dài chuỗi token (p95 = 98)
  - `baselines_frozen.json`: 3 baseline đo trước huấn luyện (a=0.000, b=0.765, c=0.965)
  - `runs.csv`: Bảng đối chứng 4 cấu hình cùng 30 optimizer steps (`correct`, `attn_only`, `wrong_lr`, `qlora`)
  - `verdict.json`: Phán quyết kiểm tra hồi quy (Catastrophic Forgetting gatekeeper)
  - `autopsy.json`: Điểm số đối chứng 4 cấu hình trên tập target test
  - `qualitative.json`: Phân tích định tính (đầy đủ các trường hợp mô hình thắng và thua)
  - `merge_check.json`: Đo đạc kiểm định sau khi merge adapter LoRA (delta = +0.0000)

---

## 📦 Trọng số Adapter
- Trọng số LoRA đã huấn luyện: `adapters/correct/` (`adapter_model.safetensors`, `adapter_config.json`).
