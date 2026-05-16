"""Gemini agent with tool-calling — Tara v2 (Ported)."""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator
from datetime import date

import google.generativeai as genai

from .config import Config
from .tools.serpapi import search_flights, search_shopping

# ── Cấu hình ─────────────────────────────────────────────────────────

genai.configure(api_key=Config.gemini_api_key)

SYSTEM_PROMPT = """Bạn là Tara Bot — agent thông minh chuyên tìm vé máy bay và săn giá đồ.

NGUYÊN TẮC QUAN TRỌNG KHI GỌI TOOL:
1. Tool search_flights: BẮT BUỘC sử dụng mã sân bay IATA. Ví dụ:
   - Hà Nội -> HAN
   - Sài Gòn/Hồ Chí Minh -> SGN
   - Đà Nẵng -> DAD
   - Phú Quốc -> PQC
   - Nha Trang -> CXR
   - Hải Phòng -> HPH
   Định dạng ngày bay (outbound_date, return_date) bắt buộc là YYYY-MM-DD.
2. Trả lời bằng tiếng Việt tự nhiên, thân thiện.
3. Sau khi tool trả kết quả, chuyển tiếp NGUYÊN VĂN kết quả đó cho user, chỉ thêm 1-2 câu ngắn. KHÔNG tự ý reformat lại kết quả từ tool.
4. Có thể nói chuyện thông thường — không cần gọi tool.

Mặc định cho câu hỏi mơ hồ về thời gian:
- "cuối tuần" → thứ Sáu tuần gần nhất (không quá khứ)
- "tuần sau" → tuần tiếp theo"""

ALL_TOOLS = [search_flights, search_shopping]
MAX_TOOL_ITERATIONS = 5

# ── Agent ─────────────────────────────────────────────────────────────

class Agent:
    def __init__(self):
        # Sử dụng gemini-2.0-flash cho tốc độ, độ ổn định và tối ưu hạn mức Free Tier
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=SYSTEM_PRO
