from typing import Any, Dict

from langchain_gigachat import GigaChat
from pydantic.v1 import SecretStr
from gigachat.settings import BASE_URL, AUTH_URL, SCOPE

from langflow.base.constants import STREAM_INFO_TEXT
from langflow.base.models.model import LCModelComponent
from langflow.field_typing import LanguageModel
from langflow.field_typing.range_spec import RangeSpec
from langflow.inputs.inputs import BoolInput, DictInput, IntInput, SecretStrInput, SliderInput, StrInput, DropdownInput
from langflow.logging import logger


class GigaChatModelComponent(LCModelComponent):
    display_name = "GigaChat"
    description = "Generates text using GigaChat LLMs."
    name = "GigaChatModel"
    icon = "brain-circuit"

    model_options = [
        "GigaChat",
        "GigaChat-preview",
        "GigaChat-Plus",
        "GigaChat-Pro-preview"
        "GigaChat-Pro",
        "GigaChat-Max-preview",
        "GigaChat-Max",
        "GigaChat-2",
        "GigaChat-2-Pro",
        "GigaChat-2-Max",
        "Embeddings",
        "Embeddings-2",
        "EmbeddingsGigaR",
    ]

    inputs = [
        *LCModelComponent._base_inputs,
        BoolInput(name="stream", display_name="Stream", info=STREAM_INFO_TEXT, advanced=False),
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            advanced=True,
            info="The maximum number of tokens to generate. Set to 0 for unlimited tokens.",
            range_spec=RangeSpec(min=0, max=128000),
        ),
        DictInput(
            name="model_kwargs",
            display_name="Model Kwargs",
            advanced=True,
            info="Additional keyword arguments to pass to the model.",
        ),
        DropdownInput(
            name="model",
            display_name="Model Name",
            options=model_options,
            advanced=False,
            value="GigaChat-2-Max",
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
        StrInput(
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
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.1,
            range_spec=RangeSpec(min=0, max=1, step=0.01),
            show=True,
        ),
        IntInput(
            name="timeout",
            display_name="Timeout",
            info="The timeout for requests to GigaChat completion API.",
            advanced=True,
            value=700,
        ),
        BoolInput(
            name="profanity_check",
            display_name="profanity_check",
            value=True,
            advanced=True,
            info="Параметр цензуры",
        ),
        BoolInput(
            name="verify_ssl_certs",
            display_name="verify_ssl_certs",
            value=False,
            advanced=True,
            info="Проверка SSL сертов"
        )
    ]

    def build_model(self) -> LanguageModel:  # type: ignore[type-var]
        logger.debug(f"Executing request with model: {self.model}")
        parameters: Dict[str, Any] = {
            "base_url": self.base_url,
            "auth_url": self.auth_url,
            "credentials": SecretStr(self.credentials).get_secret_value() if self.credentials else None,
            "scope": self.scope,
            "model": self.model,
            "profanity_check": self.profanity_check,
            "user": self.user,
            "password": SecretStr(self.password).get_secret_value() if self.password else None,
            "timeout": self.timeout,
            "verify_ssl_certs": self.verify_ssl_certs,
        }
        output = GigaChat(**parameters)

        return output

    def _get_exception_message(self, e: Exception):
        """Get a message from an OpenAI exception.

        Args:
            e (Exception): The exception to get the message from.

        Returns:
            str: The message from the exception.
        """
        try:
            from openai import BadRequestError
        except ImportError:
            return None
        if isinstance(e, BadRequestError):
            message = e.body.get("message")
            if message:
                return message
        return None