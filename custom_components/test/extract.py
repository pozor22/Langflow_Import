from ollama import Client
from langflow.custom import Component
from langflow.io import DataInput, MessageInput, Output
from langflow.schema import Message


class ExtractScript54(Component):
    display_name = "Extract Script for 5.4"
    description = "Processes TZ objects and extracts suitable tests with explanations. Robust input/output parsing."
    icon = "extract"
    name = "ExtractScript54"
    inputs = [
        MessageInput(
            name="message",
            display_name="TZ Content",
            info="TZ content message",
        ),
        MessageInput(
            name="ollama_api_base",
            display_name="Ollama API Base",
            info="Base URL for Ollama server",
            required=True,
        ),
        MessageInput(
            name="model_name",
            display_name="Model name",
            info="Ollama model",
            required=True,
        ),
        MessageInput(
            name="prompt_template",
            display_name="prompt_template",
            info="prompt_template",
        ),
        MessageInput(
            name="pm1_template",
            display_name="pm1_template",
            info="pm1_template",
        ),
    ]
    outputs = [
        Output(display_name="Extracted Data", name="output", method="build_output"),
    ]

    def create_ollama_client(self) -> Client:
        """Creates and returns an Ollama client instance."""
        try:
            return Client(host=self.ollama_api_base.text, headers={'x-some-header': 'some-value'})
        except Exception as e:
            self.status = f"Ollama client failed: {e}"
            return f"__OLLAMA_CLIENT_ERROR__: {e}"

    def send_ollama_request(self, client: Client, prompt: str) -> str:
        """Ask Ollama and return a text result (or error string)."""
        try:
            model = self.model_name.text

            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            return response.message.content
        except Exception as e:
            self.status = f"Ollama request failed: {e}"
            return f"__OLLAMA_ERROR__: {e}"

    def build_output(self) -> Message:
        tz_content = self.message.text
        pm1_template = "*ШАБЛОН ПМ1*" + self.pm1_template.text

        client = self.create_ollama_client()
        if isinstance(client, str):
            return client

        prompt = f"{self.prompt_template.text}\n\n{tz_content}\n\n{pm1_template}"

        result = self.send_ollama_request(client, prompt)

        return result