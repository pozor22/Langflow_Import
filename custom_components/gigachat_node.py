import json
import time
import requests
from typing import Any, Dict, Optional

from lfx.custom.custom_component.component import Component
from lfx.io import (
    DropdownInput,
    IntInput,
    FloatInput,
    MessageTextInput,
    MultilineInput,
    Output,
    SecretStrInput,
    SliderInput,
    StrInput,
)
from lfx.schema import Message


class GigaChatProxyPolling(Component):
    display_name = "GigaChat"
    description = "Generate text using GigaChat"
    icon = "brain-circuit"
    name = "GigaChatProxyPolling"

    model_map = {
        "GigaChat-2-Lite": "gigachat/GigaChat-2",
        "GigaChat-2-Pro": "gigachat/GigaChat-2-Pro",
        "GigaChat-2-Max": "gigachat/GigaChat-2-Max",
    }

    gigachat_scopes = ["GIGACHAT_API_PERS", "GIGACHAT_API_B2B", "GIGACHAT_API_CORP"]

    model_keys = list(model_map.keys())

    timeout_for_requests = 30

    url_proxy_llm_endpoint = "llm/v1/chat/completions"

    url_proxy_task = "general/task"

    inputs = [
        StrInput(
            name="proxy_url",
            display_name="URL Proxy",
            value="",
            required=True,
            advanced=True,
            info="URL прокси-сервера.",
        ),
        SecretStrInput(
            name="master_key",
            display_name="Master key for LiteLLM",
            value="",
            required=False,
            advanced=True,
            info="Ключ авторизации для LiteLLM.",
        ),
        DropdownInput(
            name="model",
            display_name="Model",
            options=model_keys,
            value=model_keys[0],
            required=True,
            info="Модель для генерации ответа.",
        ),
        SecretStrInput(
            name="authorization_key",
            display_name="Authorization Key",
            value="",
            required=True,
            info="Ключ авторизации для получения access token в GigaChat.",
        ),
        DropdownInput(
            name="gigachat_scope",
            display_name="GigaChat Scope",
            options=gigachat_scopes,
            value=gigachat_scopes[0],
            required=True,
            advanced=True,
            info="Scope для OAuth GigaChat.",
        ),
        MessageTextInput(
            name="input_value",
            display_name="Input",
            required=True,
            info="Текст пользовательского сообщения.",
        ),
        MultilineInput(
            name="system_message",
            display_name="System Message",
            value="You are a helpful assistant.",
            required=False,
            info="Системный промпт. В GigaChat должен быть первым сообщением и только один.",
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.5,
            range_spec={"min": 0.01, "max": 1, "step": 0.01},
            required=False,
            advanced=True,
            info="Степень случайности: меньше — стабильнее, больше — разнообразнее.",
        ),
        SliderInput(
            name="top_p",
            display_name="top_p",
            value=1.0,
            range_spec={"min": 0.01, "max": 1, "step": 0.01},
            advanced=True,
            info="Альтернатива temperature. Задает вероятностную массу токенов, которые должна учитывать модель..",
        ),
        IntInput(
            name="max_tokens",
            display_name="Max tokens",
            value=0,
            range_spec={"min": 0, "max": 32768, "step": 1},
            advanced=True,
            info="Ограничение длины генерации ответа. Если установлено 0, лимит определяется настройками самой модели по умолчанию.",
        ),
        SliderInput(
            name="repetition_penalty",
            display_name="Repetition penalty",
            value=1.0,
            range_spec={"min": 0.1, "max": 2, "step": 0.01},
            advanced=True,
            info="Количество повторений слов. 1.0 — нейтрально.",
        ),
        StrInput(
            name="x_client_id",
            display_name="X-Client-ID",
            value="",
            advanced=True,
            info="Произвольный ID клиента для логирования.",
        ),
        StrInput(
            name="x_request_id",
            display_name="X-Request-ID",
            value="",
            advanced=True,
            info="Произвольный ID запроса для логирования/трассировки.",
        ),
        StrInput(
            name="x_session_id",
            display_name="X-Session-ID",
            value="",
            advanced=True,
            info="Произвольный ID сессии для логирования.",
        ),
        MultilineInput(
            name="messages_json",
            display_name="Messages (JSON override)",
            value="",
            advanced=True,
            info="JSON-массив messages; если заполнен, заменяет Input/System Message.",
        ),
        StrInput(
            name="functions_state_id",
            display_name="Functions state id",
            value="",
            advanced=True,
            info="Идентификатор состояния функций; актуально для сценариев с function calling.",
        ),
        MultilineInput(
            name="attachments_json",
            display_name="attachments (JSON array)",
            value="",
            advanced=True,
            info="Массив file_id для вложений. Ограничения: до 10 изображений на запрос, 1 изображение на сообщение, размер запроса с медиа < 80 МБ.",
        ),
        StrInput(
            name="function_call_value",
            display_name='Function call ("none" | "auto" | JSON)',
            value="",
            advanced=True,
            info='Режим функций: "none", "auto" или объект {"name":"..."} для принудительного вызова.',
        ),
        MultilineInput(
            name="functions_json",
            display_name="Functions (JSON)",
            value="",
            advanced=True,
            info="JSON-массив описаний пользовательских функций (name/description/parameters/...).",
        ),
        FloatInput(
            name="poll_timeout_sec",
            display_name="Polling timeout seconds",
            value=120.0,
            range_spec={"min": 1.0, "max": 600.0, "step": 1.0},
            advanced=True,
            info="Максимальное время ожидания ответа от прокси.",
        ),
        FloatInput(
            name="poll_interval_sec",
            display_name="Polling interval seconds",
            value=1.0,
            range_spec={"min": 1.0, "max": 15.0, "step": 0.5},
            advanced=True,
            info="Частота проверки готовности ответа.",
        ),
    ]

    outputs = [
        Output(display_name="Message", name="message", method="run_model"),
    ]

    @staticmethod
    def _raise_http_error(resp: requests.Response, action: str) -> None:
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            details = ""
            try:
                body = resp.json()
                details = (
                              body.get("error", {}).get("message")
                              if isinstance(body.get("error"), dict)
                              else body.get("error")
                          ) or body.get("detail") or str(body)
            except Exception:
                details = resp.text.strip() or "No details"

            raise ValueError(
                f"{action} failed: HTTP {resp.status_code}. {details}"
            ) from e

    @staticmethod
    def _parse_optional_json(raw: str, field: str) -> Optional[Any]:
        if raw == "":
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"{field} must be valid JSON") from e

    def run_model(self) -> Message:
        base_url = self.proxy_url

        enqueue_url = f"{base_url}/{self.url_proxy_llm_endpoint}"

        headers: Dict[str, str] = {"Content-Type": "application/json"}

        if self.master_key != "":
            headers["master-key"] = f"{self.master_key}"

        if self.x_client_id != "":
            headers["X-Client-ID"] = self.x_client_id

        if self.x_request_id != "":
            headers["X-Request-ID"] = self.x_request_id

        if self.x_session_id != "":
            headers["X-Session-ID"] = self.x_session_id

        messages_override = self._parse_optional_json(self.messages_json, "messages_json")
        if messages_override is not None:
            if not isinstance(messages_override, list):
                raise ValueError("messages_json must be JSON array")
            messages = messages_override
        else:
            messages = []

            if self.system_message != "":
                messages.append({"role": "system", "content": self.system_message})

            user_message: Dict[str, Any] = {
                "role": "user",
                "content": self.input_value,
            }

            if self.functions_state_id != "":
                user_message["functions_state_id"] = self.functions_state_id

            attachments = self._parse_optional_json(self.attachments_json, "attachments_json")
            if attachments is not None:
                if not isinstance(attachments, list):
                    raise ValueError("attachments_json must be JSON array")
                user_message["attachments"] = attachments

            messages.append(user_message)

        payload: Dict[str, Any] = {
            "model": self.model_map.get(self.model),
            "messages": messages,
            "stream": False,
            "api_key": self.authorization_key,
            "scope": self.gigachat_scope,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }

        if self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens

        if self.function_call_value != "":
            if self.function_call_value in {"none", "auto"}:
                payload["function_call"] = self.function_call_value
            else:
                try:
                    payload["function_call"] = json.loads(self.function_call_value)
                except json.JSONDecodeError as e:
                    raise ValueError("function_call must be 'none', 'auto' or valid JSON") from e

        functions = self._parse_optional_json(self.functions_json, "functions_json")
        if functions is not None:
            if not isinstance(functions, list):
                raise ValueError("functions_json must be JSON array")
            payload["functions"] = functions

        start_resp = requests.post(enqueue_url, json=payload, headers=headers, timeout=self.timeout_for_requests)
        self._raise_http_error(start_resp, "Queue request to proxy")
        queued = start_resp.json()

        task_id = queued.get("task_id")
        if not task_id:
            raise ValueError(f"Proxy did not return task_id: {queued}")

        task_url = f"{base_url}/{self.url_proxy_task}/{task_id}"
        deadline = time.time() + self.poll_timeout_sec

        while time.time() < deadline:
            status_resp = requests.get(task_url, headers=headers, timeout=self.timeout_for_requests)
            self._raise_http_error(status_resp, "Polling task status")
            task = status_resp.json()

            state = str(task.get("status", "")).upper()
            if state in {"PENDING", "RECEIVED", "STARTED", "RETRY", "PROGRESS", "QUEUED"}:
                time.sleep(self.poll_interval_sec)
                continue

            if state != "SUCCESS":
                raise ValueError(f"Task {task_id} finished with state={state}: {task}")

            result = task.get("result") or {}
            if result.get("status") != "success":
                raise ValueError(f"LiteLLM error: {result}")

            data = result.get("data") or {}
            choices = data.get("choices") or []
            if not choices:
                return Message(text=json.dumps(data, ensure_ascii=False))

            message = choices[0].get("message") or {}
            content = message.get("content", "")
            tool_calls = message.get("tool_calls") or []
            finish_reason = choices[0].get("finish_reason")

            if tool_calls:
                return Message(
                    text=json.dumps(
                        {
                            "type": "tool_calls",
                            "finish_reason": finish_reason,
                            "tool_calls": tool_calls,
                        },
                        ensure_ascii=False,
                    )
                )

            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False)

            return Message(text=str(content))

        requests.delete(task_url, timeout=self.timeout_for_requests)

        raise TimeoutError(f"Polling timeout ({self.poll_timeout_sec}s), task_id={task_id}")