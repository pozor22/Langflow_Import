import requests
import time
from lfx.custom.custom_component.component import Component
from lfx.io import MessageTextInput, Output, DropdownInput, DictInput, IntInput
from lfx.schema.data import Data


class APIProxyComponent(Component):
    display_name = "API Proxy Client"
    description = "Makes external API proxy."
    icon = "globe"
    name = "APIProxyComponent"

    inputs = [
        MessageTextInput(
            name="proxy_address",
            display_name="Proxy Address",
            value="http://host.docker.internal:8000",
            info="Base URL of your proxy service",
        ),
        MessageTextInput(
            name="api_url",
            display_name="Target API URL",
            value="https://api.example.com/data",
        ),
        DropdownInput(
            name="method",
            display_name="HTTP Method",
            options=["GET", "POST", "PUT", "DELETE", "PATCH"],
            value="GET",
        ),
        DictInput(
            name="headers",
            display_name="Headers",
            is_list=True,  # Позволяет добавлять пары ключ-значение
        ),
        DictInput(
            name="json_body",
            display_name="JSON Body",
            info="Payload for POST/PUT requests",
            is_list=True,
        ),
        IntInput(
            name="timeout",
            display_name="Timeout (sec)",
            value=30,
        ),
    ]

    outputs = [
        Output(display_name="API Response", name="output", method="build_output"),
    ]

    def build_output(self) -> Data:
        proxy_url = self.proxy_address.rstrip('/')

        # Формируем тело запроса согласно API_PROXY_EXAMPLES.md
        payload = {
            "method": self.method,
            "url": self.api_url,
            "headers": self.headers or {},
            "timeout": self.timeout
        }

        # Добавляем JSON тело только если оно есть и метод подходящий
        if self.json_body and self.method in ["POST", "PUT", "PATCH"]:
            payload["json"] = self.json_body

        try:
            # 1. Отправка в очередь
            response = requests.post(f"{proxy_url}/api/v1/proxy", json=payload)
            response.raise_for_status()
            task_data = response.json()
            task_id = task_data.get("task_id")

            if not task_id:
                raise ValueError(f"No task_id returned: {task_data}")

            # 2. Polling
            result = self._poll_result(proxy_url, task_id)

            # result содержит: { "status": "success", "status_code": 200, "data": {...}, ... }
            self.status = result

            # Возвращаем данные. Можно вернуть result['data'] если нужен только payload
            return Data(value=result)

        except Exception as e:
            error_msg = f"Error calling API Proxy: {str(e)}"
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
                    return data.get("result")
                elif status == "FAILURE":
                    raise Exception(f"Task failed: {data.get('result')}")

            time.sleep(1)

        raise TimeoutError("Task polling timed out")