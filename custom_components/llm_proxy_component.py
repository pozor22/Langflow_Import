import requests
import time
from lfx.custom.custom_component.component import Component
from lfx.io import MessageTextInput, Output, DropdownInput
from lfx.schema.data import Data


class LLMProxyComponent(Component):
    display_name = "LLM Proxy Client"
    description = "Sends prompts to LLM proxy"
    icon = "bot"
    name = "LLMProxyComponent"

    inputs = [
        MessageTextInput(
            name="proxy_address",
            display_name="Proxy Address",
            value="http://host.docker.internal:8000",
            info="Base URL of your proxy service",
        ),
        DropdownInput(
            name="provider",
            display_name="Provider",
            options=["ollama"],
            value="ollama",
        ),
        MessageTextInput(
            name="model_name",
            display_name="Model Name",
            value="llama3",
            info="e.g. llama3, mistral",
        ),
        MessageTextInput(
            name="input_prompt",
            display_name="Prompt",
            info="The user prompt",
            tool_mode=True,
        ),
    ]

    outputs = [
        Output(display_name="Output Text", name="output", method="build_output"),
    ]

    def build_output(self) -> Data:
        proxy_url = self.proxy_address.rstrip('/')

        # Формируем имя модели (для liteLLM часто используется формат provider/model)
        full_model = f"{self.provider}/{self.model_name}" if self.provider else self.model_name

        payload = {
            "model": full_model,
            "messages": [
                {"role": "user", "content": self.input_prompt}
            ]
        }

        try:
            # 1. Отправка задачи
            response = requests.post(f"{proxy_url}/llm/v1/chat/completions", json=payload)
            response.raise_for_status()
            task_data = response.json()
            task_id = task_data.get("task_id")

            if not task_id:
                raise ValueError(f"No task_id returned: {task_data}")

            # 2. Polling результата
            result = self._poll_result(proxy_url, task_id)

            # 3. Извлечение текста ответа LLM
            # Структура: result = {"status": "success", "status_code": 200, "data": {...}, ...}
            # В data находится OpenAI-совместимый ответ с choices
            if result.get("status") == "success":
                data = result.get("data", {})
                choices = data.get("choices", [])

                if choices and len(choices) > 0:
                    content = choices[0].get("message", {}).get("content", "")
                    if content:
                        self.status = content
                        return Data(value=content)

                # Если структура отличается, возвращаем весь data
                self.status = str(data)
                return Data(value=str(data))
            else:
                # Обработка ошибки от задачи
                error_info = result.get("error", result)
                raise Exception(f"LLM task failed: {error_info}")

        except Exception as e:
            error_msg = f"Error calling LLM Proxy: {str(e)}"
            self.status = error_msg
            raise ValueError(error_msg)

    def _poll_result(self, base_url, task_id, timeout=300):
        start_time = time.time()
        while time.time() - start_time < timeout:
            resp = requests.get(f"{base_url}/general/task/{task_id}")
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status")

                if status == "SUCCESS":
                    # Возвращаем result из ответа polling
                    return data.get("result", {})
                elif status == "FAILURE":
                    raise Exception(f"Task failed: {data.get('result')}")

            time.sleep(1)  # Ждем 1 секунду перед следующим опросом

        raise TimeoutError("Task polling timed out")