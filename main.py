from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from google import genai
from google.genai import types # Phải import thêm types để cấu hình JSON
import json
import os

# 1. CẤU HÌNH API KEY (NHỚ THAY KEY MỚI VÀO ĐÂY)
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

# 2. KHỞI TẠO FASTAPI
app = FastAPI(title="Hotel Review AI Analyzer")

# 3. ĐỊNH NGHĨA DỮ LIỆU ĐẦU VÀO (Pydantic)
class ReviewItem(BaseModel):
    id: int
    content: str

class ReviewRequest(BaseModel):
    reviews: List[ReviewItem]

# 4. TẠO API ENDPOINT
@app.post("/analyze")
async def analyze_reviews(request: ReviewRequest):
    try:
        # Chuyển đổi danh sách review thành chuỗi để đưa vào Prompt
        reviews_data = [{"id": r.id, "content": r.content} for r in request.reviews]
        
        prompt = f"""
        Bạn là một Giám đốc vận hành khách sạn cực kỳ khắt khe và chú trọng chi tiết.

        Dưới đây là danh sách đánh giá của khách hàng:
        {json.dumps(reviews_data, ensure_ascii=False)}

        ====================
        PHẦN 1: PHÂN TÍCH TỪNG ĐÁNH GIÁ
        ====================
        Trả về mảng "reviews_analysis", mỗi object phải có cấu trúc CHÍNH XÁC:

        - "id": Giữ nguyên ID
        - "sentiment": Chỉ được chọn 1 trong 3: "Tích cực", "Tiêu cực", "Trung lập"
        - "tags": Mảng từ khóa (ví dụ: ["Vệ sinh", "Lễ tân", "Giá cả", "Tiện ích", "Phong cảnh", "Nội thất", "Dịch vụ"])
        - "summary": Tóm tắt ngắn gọn (1 câu)
        - "action_needed":
            + Nếu có vấn đề: ghi rõ hành động cần sửa (cụ thể, thực tế)
            + Nếu không có vấn đề: ghi "Không có vấn đề, tiếp tục phát huy"

        ====================
        PHẦN 2: TỔNG KẾT TOÀN BỘ
        ====================
        Tạo object "overall_summary" gồm:

        - "main_issues": Danh sách vấn đề lớn lặp lại nhiều lần
        - "priority_actions": Danh sách hành động cần ưu tiên xử lý ngay
        - "positive_points": Những điểm mạnh cần duy trì
        - "general_recommendation": Nhận định tổng thể (1-2 câu)

        ====================
        FORMAT TRẢ VỀ (BẮT BUỘC)
        ====================
        CHỈ trả về JSON object duy nhất:

        {{
          "reviews_analysis": [...],
          "overall_summary": {{...}}
        }}

        KHÔNG thêm text, KHÔNG markdown, KHÔNG giải thích.
        """

        # Gọi AI bằng cú pháp mới nhất kèm JSON Mode
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        
        # Tiền xử lý chuỗi trả về (đề phòng AI vẫn nhét thẻ markdown)
        result_text = response.text.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]

        # Parse chuỗi thành JSON và trả về cho Laravel
        return json.loads(result_text)

    except Exception as e:
        import traceback
        print("=== LỖI TỪ AI HOẶC PYTHON ===")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý AI: {str(e)}")