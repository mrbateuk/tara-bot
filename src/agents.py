"""Gemini agent with tool-calling — Tara v2 (Ported).

Sự thay đổi:
- Chuyển sang google.generativeai (Gemini 1.5 Flash - Tốc độ cao, miễn phí).
- Lược bỏ JSON schema: Gemini tự đọc docstring của hàm Python làm công cụ.
- Quản lý bộ nhớ: Tự động qua object chat.send_message().
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator
from datetime import date

import google.generativeai as genai
from google.generativeai.types import content_types

from .config import Config
from .tools.serpapi import search_flights, search_shopping

# ── Cấu hình ─────────────────────────────────────────────────────────

# Lưu ý: Cần thêm gemini_api_key vào class Config trong file config.py của bạn
genai.configure(api_key=Config.gemini_api_key)

SYSTEM_PROMPT = """Bạn là Tara Bot — agent thông minh chuyên tìm vé máy bay và săn giá đồ.

NGUYÊN TẮC:
- Trả lời bằng tiếng Việt tự nhiên, thân thiện.
- Khi user hỏi vé máy bay, gọi tool search_flights.
- Khi user hỏi giá sản phẩm, gọi tool search_shopping.
- Sau khi tool trả kết quả, chuyển tiếp NGUYÊN VĂN kết quả đó cho user, chỉ thêm 1-2 câu ngắn.
- KHÔNG reformat lại kết quả từ tool.
- Có thể nói chuyện thông thường — không cần gọi tool.

Mặc định cho câu hỏi mơ hồ về thời gian:
- "cuối tuần" → thứ Sáu tuần gần nhất (không quá khứ)
- "tuần sau" → tuần tiếp theo"""

# Với Gemini, ta nạp trực tiếp function vào list, hệ thống sẽ tự phân tích biến số
ALL_TOOLS = [search_flights, search_shopping]
MAX_TOOL_ITERATIONS = 5

# ── Agent ─────────────────────────────────────────────────────────────

class Agent:
    def __init__(self):
        # Sử dụng model Flash cho tốc độ vượt trội trong các tác vụ chat real-time
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=SYSTEM_PROMPT,
            tools=ALL_TOOLS,
        )
        # Gemini tự lưu lịch sử trong object này
        self.chat_session = self.model.start_chat(history=[])

    def _with_date(self, user_message: str) -> str:
        """Inject ngày hôm nay vào user message."""
        today = date.today().strftime("%A, %d/%m/%Y")
        return f"[Hôm nay: {today}]\n{user_message}"

    def chat(self, user_message: str) -> str:
        """Sync chat — tool-use loop, trả về text cuối cùng."""
        injected = self._with_date(user_message)

        for iteration in range(MAX_TOOL_ITERATIONS):
            response = self.chat_session.send_message(injected)

            if response.function_call:
                # Trích xuất tên tool và tham số
                part = response.parts[0]
                func_name = part.function_call.name
                func_args = {k: v for k, v in part.function_call.args.items()}

                print(f"[iter {iteration + 1}] Gọi hàm: {func_name} với tham số {func_args}")

                # Thực thi tool
                if func_name == "search_flights":
                    result = search_flights(**func_args)
                elif func_name == "search_shopping":
                    result = search_shopping(**func_args)
                else:
                    result = "Lỗi: Không tìm thấy tool."

                # Gói kết quả để gửi lại cho mô hình trong vòng lặp tiếp theo
                injected = content_types.Part.from_function_response(
                    name=func_name,
                    response={"result": str(result)}
                )
            else:
                # Nếu không gọi tool, trả về văn bản
                return response.text

        return "Xin lỗi, em không thể xử lý yêu cầu này. Thử lại với câu hỏi đơn giản hơn nhé!"

    async def stream_chat(self, user_message: str) -> AsyncGenerator[str | dict, None]:
        """Async generator stream cho Telegram fake-streaming."""
        injected = self._with_date(user_message)

        for iteration in range(MAX_TOOL_ITERATIONS):
            response = self.chat_session.send_message(injected, stream=True)

            has_tool_call = False
            func_name = None
            func_args = {}

            # Duyệt qua các chunk trả về
            for chunk in response:
                # Nếu LLM quyết định gọi tool
                if chunk.function_call:
                    has_tool_call = True
                    func_name = chunk.function_call.name
                    func_args = {k: v for k, v in chunk.function_call.args.items()}
                    # Yield pill trạng thái cho Telegram
                    yield {"type": "tool_use", "name": func_name}
                
                # Nếu LLM sinh ra văn bản
                if chunk.text:
                    yield chunk.text

            # Xử lý kết quả tool SAU KHI stream của iteration này kết thúc
            if has_tool_call:
                print(f"[stream iter {iteration + 1}] Chạy: {func_name}")
                if func_name == "search_flights":
                    result = search_flights(**func_args)
                elif func_name == "search_shopping":
                    result = search_shopping(**func_args)
                else:
                    result = "Lỗi khi chạy tool."

                # Nạp kết quả vào để chuẩn bị cho iteration tiếp theo phân tích
                injected = content_types.Part.from_function_response(
                    name=func_name,
                    response={"result": str(result)}
                )
                continue
            else:
                break
