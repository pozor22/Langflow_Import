from typing import Dict, Any

from gigachat.settings import BASE_URL, AUTH_URL, SCOPE
from langflow.base.embeddings.model import LCEmbeddingsModel
from langflow.base.models.openai_constants import OPENAI_EMBEDDING_MODEL_NAMES
from langflow.field_typing import Embeddings
from langflow.io import BoolInput, DictInput, DropdownInput, FloatInput, IntInput, MessageTextInput, SecretStrInput, StrInput
from langchain_gigachat import GigaChatEmbeddings
from pydantic.v1 import SecretStr


class GigaChatEmbeddingsComponent(LCEmbeddingsModel):
    display_name = "GigaChat Embeddings"
    description = "Generate embeddings using GigaChat models."
    name = "GigaChatEmbeddings"

    inputs = [
        StrInput(
            name="model",
            display_name="Model Name",
            advanced=False,
            value="EmbeddingsGigaR",
        ),
        StrInput(
            name="base_url",
            display_name="GigaChat API Base",
            advanced=True,
            value=BASE_URL,
            info="The base URL of the GigaChat API. "
                 f"Defaults to {BASE_URL}. "
        ),
        StrInput(
            name="auth_url",
            display_name="GigaChat Auth URL",
            advanced=True,
            value=AUTH_URL,
            info="The auth URL of the GigaChat API. "
                 f"Defaults to {AUTH_URL}. "
        ),
        StrInput(
            name="scope",
            display_name="GigaChat Scope",
            advanced=False,
            value=SCOPE,
            info="The scope of the GigaChat API. "
                 f"Defaults to {SCOPE}. "
        ),
        SecretStrInput(
            name="credentials",
            display_name="GigaChat Credentials",
            info="The GigaChat API Key to use for the GigaChat model.",
            advanced=False,
            value=None,
            required=False,
        ),
        StrInput(
            name="user",
            display_name="GigaChat User",
            info="The GigaChat API Username to use.",
            advanced=True,
            value="USERNAME",
            required=False,
        ),
        SecretStrInput(
            name="password",
            display_name="GigaChat Password",
            info="The GigaChat API Password to use.",
            advanced=True,
            value=None,
            required=False,
        ),
        BoolInput(
            name="verify_ssl_certs",
            display_name="verify_ssl_certs",
            value=False,
            advanced=True,
            info="Проверка SSL сертов"
        ),
        IntInput(
            name="timeout",
            display_name="Timeout",
            info="The timeout for requests to GigaChat API.",
            advanced=True,
            value=700,
        ),
    ]

    def build_embeddings(self) -> Embeddings:
        parameters: Dict[str, Any] = {
            "base_url": self.base_url,
            "auth_url": self.auth_url,
            "credentials": SecretStr(self.credentials).get_secret_value() if self.credentials else None,
            "scope": self.scope,
            "model": self.model,
            "user": self.user,
            "password": SecretStr(self.password).get_secret_value() if self.password else None,
            "timeout": self.timeout,
            "verify_ssl_certs": self.verify_ssl_certs,
        }
        return GigaChatEmbeddings(**parameters)
