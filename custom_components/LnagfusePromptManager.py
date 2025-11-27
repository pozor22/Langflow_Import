import ssl
import requests
import json
import base64
from typing import Any, Dict, List, Optional

from langflow.custom import Component
from langflow.inputs import DropdownInput, StrInput
from langflow.io import Output
from langflow.schema.message import Message

ssl._create_default_https_context = ssl._create_unverified_context

class LangfusePromptComponent(Component):
    display_name = "Langfuse Prompt Manager"
    description = "Fetch Langfuse prompts and return normalized messages"
    icon = "prompts"

    inputs = [
        DropdownInput(
            name="selected_prompt",
            display_name="Select Prompt",
            options=[],
            refresh_button=True,
            advanced=False
        ),
        DropdownInput(
            name="label",
            display_name="Label",
            options=["production", "staging", "latest"],
            value="production",
            advanced=False
        ),
        StrInput(
            name="host",
            display_name="Host",
            advanced=False
        ),
        StrInput(
            name="public_key",
            display_name="Public Key",
            advanced=False
        ),
        StrInput(
            name="secret_key",
            display_name="Secret Key",
            advanced=False
        ),
    ]

    outputs = [
        Output(display_name="Prompt Content", name="content", method="get_prompt_content"),
        Output(display_name="Raw Messages (JSON)", name="raw_messages", method="get_raw_messages")
    ]

    # --- auth helpers ---------------------------------------------------
    def _get_auth_header(self) -> Dict[str, str]:
        """Support both Bearer (sk_...) and Basic(public:secret)."""
        sk = getattr(self, "secret_key", None) or ""
        pk = getattr(self, "public_key", "") or ""

        if sk.startswith("sk_") or sk.startswith("bearer_"):  # heuristic
            return {"Authorization": f"Bearer {sk}"}
        if pk and sk:
            creds = f"{pk}:{sk}"
            token = base64.b64encode(creds.encode("utf-8")).decode("utf-8")
            return {"Authorization": f"Basic {token}"}
        if sk:
            # fallback to try bearer anyway
            return {"Authorization": f"Bearer {sk}"}
        raise ValueError("No secret_key provided for Langfuse API access")

    # --- UI / build config hook -----------------------------------------
    def update_build_config(self, build_config: dict, field_value: str, field_name: str | None = None):
        """Update dropdown options with 'name||id' strings for easy parsing."""
        if field_name == "selected_prompt":
            opts = self._fetch_prompts_list()
            build_config["selected_prompt"]["options"] = opts
        return build_config

    # --- API fetch helpers ----------------------------------------------
    def _fetch_prompts_list(self) -> List[str]:
        """Return options in format 'name||id' so we can request by id later."""
        try:
            headers = self._get_auth_header()
        except Exception as e:
            print(f"[auth error] {e}")
            return ["Error: set secret_key"]

        url = f"{self.host.rstrip('/')}/api/public/v2/prompts"
        try:
            r = requests.get(url, headers=headers, params={"page_size": 200}, timeout=10)
            r.raise_for_status()
            payload = r.json()
            # try common shapes
            candidates = []
            if isinstance(payload, dict):
                if "data" in payload and isinstance(payload["data"], list):
                    candidates = payload["data"]
                elif "items" in payload and isinstance(payload["items"], list):
                    candidates = payload["items"]
                elif "results" in payload and isinstance(payload["results"], list):
                    candidates = payload["results"]
                else:
                    if "data" in payload and isinstance(payload["data"], dict):
                        candidates = [payload["data"]]
            elif isinstance(payload, list):
                candidates = payload

            opts = []
            for p in candidates:
                name = p.get("name") or p.get("title") or p.get("id")
                pid = p.get("id") or p.get("prompt_id") or name
                if name and pid:
                    opts.append(f"{name}")
            return opts or ["No prompts found"]
        except Exception as e:
            print(f"[fetch prompts error] {e}")
            return [f"Error loading prompts: {str(e)}"]

    # --- robust parser for different response shapes --------------------
    def _extract_prompt_messages(self, obj: Any) -> List[Dict[str, str]]:
        """
        Normalize prompt content to a list of messages like:
        [{"role":"system","content":"..."}, {"role":"user","content":"..."}]
        Accepts many Langfuse shapes:
        - {"prompt": [ {role, content}, ... ]}
        - {"data": {"prompt": ...}}
        - {"prompt": "plain string"}
        - list of strings or list of dicts
        """
        def find_prompt_node(d):
            if isinstance(d, dict):
                for k in ("prompt", "messages", "chat", "content"):
                    if k in d:
                        return d[k]
                for v in d.values():
                    node = find_prompt_node(v)
                    if node is not None:
                        return node
            return None

        node = find_prompt_node(obj)
        if node is None:
            node = obj

        messages: List[Dict[str, str]] = []

        if isinstance(node, str):
            messages.append({"role": "system", "content": node})
            return messages

        if isinstance(node, list):
            for item in node:
                if isinstance(item, dict):
                    role = item.get("role") or item.get("actor") or item.get("author") or "system"
                    content = item.get("content") or item.get("text") or item.get("message") or ""
                    messages.append({"role": role, "content": content})
                else:
                    messages.append({"role": "system", "content": str(item)})
            return messages

        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, str):
                    messages.append({"role": k, "content": v})
                elif isinstance(v, dict) and "content" in v:
                    messages.append({"role": k, "content": v.get("content", "")})
            if messages:
                return messages

        try:
            txt = json.dumps(obj, ensure_ascii=False)
        except Exception:
            txt = str(obj)
        messages.append({"role": "system", "content": txt})
        return messages

    # --- fetch single prompt by id/name --------------------------------
    def _get_prompt_content_raw(self, prompt_identifier: str) -> Optional[Any]:
        """
        prompt_identifier may be "name||id" or id or name.
        We'll try to extract id if format 'name||id' else use it raw.
        """
        try:
            headers = self._get_auth_header()
        except Exception as e:
            print(f"[auth error] {e}")
            return None

        if "||" in prompt_identifier:
            _, pid = prompt_identifier.split("||", 1)
            pid = pid.strip()
        else:
            pid = prompt_identifier.strip()

        url = f"{self.host.rstrip('/')}/api/public/v2/prompts/{pid}"
        try:
            r = requests.get(url, headers=headers, params={"label": getattr(self, "label", None)}, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[get prompt raw error] {e}")
            # fallback: list all prompts and find by name
            try:
                allp = requests.get(f"{self.host.rstrip('/')}/api/public/v2/prompts", headers=headers, timeout=10)
                allp.raise_for_status()
                payload = allp.json()
                items = payload.get("data") or payload.get("items") or payload.get("results") or []
                for p in items:
                    if p.get("name") == prompt_identifier or p.get("id") == prompt_identifier:
                        return p
            except Exception as e2:
                print(f"[fallback list error] {e2}")
            return None

    # --- Outputs: content, raw_messages --------------------------------
    async def get_prompt_content(self) -> Message:
        """Return the content as a single Message with role=system (most compatible)."""
        identifier = getattr(self, "selected_prompt", None)
        if not identifier:
            return Message(text="No prompt selected")

        raw = self._get_prompt_content_raw(identifier)
        if raw is None:
            return Message(text="Error fetching prompt (see logs)")

        messages = self._extract_prompt_messages(raw)

        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        system_text = "\n\n".join(system_parts).strip() if system_parts else ""

        if system_text:
            try:
                return Message(role="system", text=system_text)
            except TypeError:
                return Message(text=system_text)

        merged = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return Message(text=merged)

    async def get_raw_messages(self) -> Message:
        """Return JSON string of messages (for connecting to LLM nodes that accept messages list)."""
        identifier = getattr(self, "selected_prompt", None)
        if not identifier:
            return Message(text="[]")
        raw = self._get_prompt_content_raw(identifier)
        if raw is None:
            return Message(text="[]")
        messages = self._extract_prompt_messages(raw)
        return Message(text=json.dumps(messages, ensure_ascii=False))
