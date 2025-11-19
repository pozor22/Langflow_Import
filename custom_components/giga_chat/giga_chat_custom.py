from gigachat import GigaChat
from langflow.custom import Component
from langflow.schema import Message
from langflow.io import DataInput, MessageInput, Output, DropdownInput


class GigaChatCustom(Component):
    display_name = "GigaChatCustom"
    description = "Runs a language model given a specified provider."
    documentation: str = "https://docs.langflow.org/components-models"
    icon = "brain-circuit"

    model_options = ["GigaChat", "GigaChat-Pro", "GigaChat-Max", "GigaChat-2", "GigaChat-2-Pro", "GigaChat-2-Max"]

    inputs = [
        MessageInput(name="input", display_name="Input", required=True),
        MessageInput(name="prompt", display_name="Prompt", required=True),
        MessageInput(name="credentials", display_name="Credentials", required=True),
        DropdownInput(name="model", display_name="Model", options=model_options, required=True),
    ]
    outputs = [
        Output(display_name="Output answer GigaChat", name="output", method="build_output"),
    ]

    def create_gigachat_client(self) -> GigaChat:
        client = GigaChat(
            credentials=self.credentials.text,
            model=self.model,
            verify_ssl_certs=False,
        )

        return client

    def senf_gigachat_request(self, giga: GigaChat) -> str:
        response = giga.chat(self.prompt.text)

        return response.choices[0].message.content

    def build_output(self) -> Message:
        gigachat_client = self.create_gigachat_client()

        answer = self.senf_gigachat_request(gigachat_client)

        return answer
