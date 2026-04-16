"""Backwards-compatibility shim for the legacy ``AssistantModel`` class.

Re-exports the ``AssistantModel`` class and module-level constants that
used to live in the pre-rewrite ``pinecone-plugin-assistant`` distribution.
Preserved to keep pre-rewrite callers working. New code should use the
Pinecone client assistants namespace directly.

:meta private:
"""

import json
import time
from io import BytesIO
from typing import Any, Iterable, List, Optional, Union
from urllib.parse import urljoin

import requests  # noqa: TID251

from pinecone.models.assistant.chat import ChatCitation, ChatUsage
from pinecone.models.assistant.streaming import StreamContentDelta
from pinecone_plugins.assistant.models.chat import (
    ChatResponse,
    ContextOptions,
    StreamChatResponseCitation,
    StreamChatResponseContentDelta,
    StreamChatResponseMessageEnd,
    StreamChatResponseMessageStart,
)
from pinecone_plugins.assistant.models.chat_completion import (
    ChatCompletionResponse,
    StreamingChatCompletionChunk,
)
from pinecone_plugins.assistant.models.context_responses import ContextResponse
from pinecone_plugins.assistant.models.file_model import FileModel
from pinecone_plugins.assistant.models.list_files_response import ListFilesResponse
from pinecone_plugins.assistant.models.shared import Message

HOST_SUFFIX = "assistant"
MODELS = [
    "gpt-4o",
    "gpt-4.1",
    "o4-mini",
    "claude-3-5-sonnet",
    "claude-3-7-sonnet",
    "gemini-2.5-pro",
]
API_VERSION = "2026-04"

RawMessage = dict  # type: ignore[type-arg]
RawMessages = Union[List[Message], List[RawMessage]]


class AssistantModel:
    """Legacy AssistantModel from the pre-rewrite ``pinecone-plugin-assistant`` plugin.

    Wraps an assistant API response and provides methods for file management,
    chat, and context retrieval via the legacy plugin-style interface.
    """

    def __init__(self, assistant: Any, client_builder: Any, config: Any) -> None:
        self.assistant = assistant
        self.host = assistant.host
        self.host = urljoin(self.host, HOST_SUFFIX)
        self.config = config or {}

        self._assistant_data_api = client_builder(host=self.host)

        # Store key attributes so property access is fast
        self.name = self.assistant.name
        self.created_at = self.assistant.created_at
        self.updated_at = self.assistant.updated_at
        self.metadata = self.assistant.metadata
        self.status = self.assistant.status

    def __str__(self) -> str:
        return str(self.assistant)

    def __repr__(self) -> str:
        return repr(self.assistant)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.assistant, attr)

    # -----------------------------------------------------------------------
    # File upload
    # -----------------------------------------------------------------------

    def upload_file(
        self,
        file_path: str,
        metadata: Optional[dict[str, Any]] = None,
        multimodal: Optional[bool] = None,
        timeout: Optional[int] = None,
        file_id: Optional[str] = None,
    ) -> FileModel:
        """Upload a file from the filesystem to this assistant."""
        try:
            with open(file_path, "rb") as file:
                return self._upload_file_stream(file, metadata, multimodal, timeout, file_id)
        except FileNotFoundError:
            raise Exception(f"Error: The file at {file_path} was not found.")  # noqa: B904
        except IOError:  # noqa: UP024
            raise Exception(f"Error: Could not read the file at {file_path}.")  # noqa: B904

    def upload_bytes_stream(
        self,
        stream: BytesIO,
        file_name: str,
        metadata: Optional[dict[str, Any]] = None,
        multimodal: Optional[bool] = None,
        timeout: Optional[int] = None,
        file_id: Optional[str] = None,
    ) -> FileModel:
        """Upload a bytes stream as a file to this assistant."""
        stream.name = file_name
        return self._upload_file_stream(stream, metadata, multimodal, timeout, file_id)

    def _upload_file_stream(
        self,
        file_stream: Any,
        metadata: Optional[dict[str, Any]] = None,
        multimodal: Optional[bool] = None,
        timeout: Optional[int] = None,
        file_id: Optional[str] = None,
    ) -> FileModel:
        kwargs: dict[str, Any] = {
            "assistant_name": self.assistant.name,
            "file": file_stream,
        }
        if metadata:
            kwargs["metadata"] = metadata
        if multimodal is not None:
            kwargs["multimodal"] = str(multimodal).lower()

        if file_id is not None:
            kwargs["assistant_file_id"] = file_id
            operation = self._assistant_data_api.upsert_file(**kwargs)
        else:
            operation = self._assistant_data_api.upload_file(**kwargs)

        # timeout == -1: return immediately without waiting
        if timeout == -1:
            return self.describe_file(operation.file_id)

        # Poll operation until terminal state
        operation = self._poll_operation(operation.id, timeout)

        if operation.status == "Failed":
            raise Exception(f"File processing failed. Error: {operation.error_message}")

        return self.describe_file(operation.file_id)

    def _poll_operation(self, operation_id: str, timeout: Optional[int]) -> Any:
        """Poll describe_operation until terminal state. Returns final operation."""
        operation = self._assistant_data_api.describe_operation(
            assistant_name=self.name, operation_id=operation_id
        )
        if timeout is None:
            while operation.status == "Processing":
                time.sleep(5)
                operation = self._assistant_data_api.describe_operation(
                    assistant_name=self.name, operation_id=operation_id
                )
            return operation
        while operation.status == "Processing" and timeout >= 0:
            time.sleep(5)
            timeout -= 5
            operation = self._assistant_data_api.describe_operation(
                assistant_name=self.name, operation_id=operation_id
            )
        if timeout < 0 and operation.status == "Processing":
            raise TimeoutError(
                "File operation timed out. "
                f"Please check the operation status for operation {operation_id}."
            )
        return operation

    # -----------------------------------------------------------------------
    # File management
    # -----------------------------------------------------------------------

    def describe_file(self, file_id: str, include_url: Optional[bool] = False) -> FileModel:
        """Describe a file attached to this assistant."""
        if include_url:
            file = self._assistant_data_api.describe_file(
                assistant_name=self.name,
                assistant_file_id=file_id,
                include_url=str(include_url).lower(),
            )
        else:
            file = self._assistant_data_api.describe_file(
                assistant_name=self.name,
                assistant_file_id=file_id,
            )
        return FileModel.from_openapi(file)

    def list_files(
        self,
        filter: Optional[dict[str, Any]] = None,
    ) -> List[FileModel]:
        """List all files attached to this assistant (auto-paginates)."""
        all_files: list[Any] = []
        pagination_token = None
        while True:
            page = self.list_files_paginated(filter=filter, pagination_token=pagination_token)
            all_files.extend(page.files)
            if page.next_token is None:
                break
            pagination_token = page.next_token
        return all_files

    def list_files_paginated(
        self,
        filter: Optional[dict[str, Any]] = None,
        limit: Optional[int] = None,
        pagination_token: Optional[str] = None,
    ) -> ListFilesResponse:
        """List one page of files attached to this assistant."""
        kwargs: dict[str, Any] = {"assistant_name": self.name}
        if filter:
            kwargs["filter"] = json.dumps(filter)
        if limit is not None:
            kwargs["limit"] = limit
        if pagination_token is not None:
            kwargs["pagination_token"] = pagination_token

        resp = self._assistant_data_api.list_files(**kwargs)
        return ListFilesResponse.from_openapi(resp)

    def delete_file(self, file_id: str, timeout: Optional[int] = None) -> None:
        """Delete a file from this assistant."""
        operation = self._assistant_data_api.delete_file(
            assistant_name=self.name, assistant_file_id=file_id
        )

        if timeout == -1:
            return

        operation = self._poll_operation(operation.id, timeout)

        if operation.status == "Failed":
            raise Exception(f"File deletion failed. Error: {operation.error_message}")

    # -----------------------------------------------------------------------
    # Message parsing
    # -----------------------------------------------------------------------

    @classmethod
    def _parse_messages(cls, messages: Union[List[Message], List[RawMessage]]) -> List[Message]:
        """Convert raw dict messages to Message objects."""
        return [
            Message.from_dict(message) if isinstance(message, dict) else message
            for message in messages
        ]

    # -----------------------------------------------------------------------
    # Chat
    # -----------------------------------------------------------------------

    def chat(
        self,
        messages: Union[List[Message], List[RawMessage]],
        filter: Optional[dict[str, Any]] = None,
        stream: bool = False,
        model: Union[str, None] = None,
        temperature: Optional[float] = None,
        json_response: bool = False,
        include_highlights: bool = False,
        context_options: Optional[Union[ContextOptions, dict[str, Any]]] = None,
    ) -> Union[ChatResponse, Iterable[Any]]:
        """Chat with this assistant."""
        if model is None:
            model = "gpt-4o"
        if json_response and stream:
            raise ValueError("Cannot use json_response with streaming")

        messages = self._parse_messages(messages)
        context_options = (
            ContextOptions.from_dict(context_options)
            if isinstance(context_options, dict)
            else context_options
        )  # type: ignore[attr-defined]

        if stream:
            return self._chat_streaming(
                messages=messages,
                model=model,
                filter=filter,
                include_highlights=include_highlights,
                context_options=context_options,
                temperature=temperature,
            )
        return self._chat_single(
            messages=messages,
            model=model,
            filter=filter,
            json_response=json_response,
            include_highlights=include_highlights,
            context_options=context_options,
            temperature=temperature,
        )

    def _chat_single(
        self,
        messages: List[Message],
        model: str = "gpt-4o",
        temperature: Optional[float] = None,
        filter: Optional[dict[str, Any]] = None,
        json_response: bool = False,
        include_highlights: bool = False,
        context_options: Optional[ContextOptions] = None,
    ) -> ChatResponse:
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]

        kwargs: dict[str, Any] = {
            "messages": msg_dicts,
            "model": model,
            "json_response": json_response,
            "include_highlights": include_highlights,
        }
        if filter:
            kwargs["filter"] = filter
        if temperature is not None:
            kwargs["temperature"] = temperature
        if context_options is not None:
            options: dict[str, Any] = {}
            if context_options.top_k is not None:
                options["top_k"] = context_options.top_k
            if context_options.snippet_size is not None:
                options["snippet_size"] = context_options.snippet_size
            if context_options.multimodal is not None:
                options["multimodal"] = context_options.multimodal
            if context_options.include_binary_content is not None:
                options["include_binary_content"] = context_options.include_binary_content
            if options:
                kwargs["context_options"] = options

        return self._assistant_data_api.chat_assistant(  # type: ignore[return-value]
            assistant_name=self.name, **kwargs
        )

    def _chat_streaming(
        self,
        messages: List[Message],
        model: str = "gpt-4o",
        temperature: Optional[float] = None,
        filter: Optional[dict[str, Any]] = None,
        include_highlights: bool = False,
        context_options: Optional[ContextOptions] = None,
    ) -> Iterable[Any]:
        api_key = self.config.api_key
        base_url = f"{self.host}/chat/{self.name}"
        headers = {"api-key": api_key, "Content-Type": "application/json"}
        content: dict[str, Any] = {
            "messages": [vars(message) for message in messages],
            "stream": True,
            "model": model,
            "include_highlights": include_highlights,
        }

        if filter:
            content["filter"] = filter
        if temperature is not None:
            content["temperature"] = temperature
        if context_options is not None:
            options = {}
            if context_options.top_k is not None:
                options["top_k"] = context_options.top_k
            if context_options.snippet_size is not None:
                options["snippet_size"] = context_options.snippet_size
            if context_options.multimodal is not None:
                options["multimodal"] = context_options.multimodal
            if context_options.include_binary_content is not None:
                options["include_binary_content"] = context_options.include_binary_content
            if options:
                content["context_options"] = options

        try:
            response = requests.post(
                base_url, headers=headers, json=content, timeout=60, stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = line.decode("utf-8")
                    if data.startswith("data:"):  # noqa: FURB188
                        data = data[5:]

                    json_data = json.loads(data)

                    res = None
                    if json_data.get("type") == "message_start":
                        res = StreamChatResponseMessageStart(
                            model=json_data.get("model", ""),
                            role=json_data.get("role", ""),
                        )
                    elif json_data.get("type") == "content_chunk":
                        delta_d = json_data.get("delta") or {}
                        res = StreamChatResponseContentDelta(
                            id=json_data.get("id", ""),
                            delta=StreamContentDelta(content=delta_d.get("content", "")),
                            model=json_data.get("model"),
                        )
                    elif json_data.get("type") == "citation":
                        citation_d = json_data.get("citation") or {}
                        res = StreamChatResponseCitation(
                            id=json_data.get("id", ""),
                            citation=ChatCitation(
                                position=citation_d.get("position", 0),
                                references=citation_d.get("references", []),
                            ),
                            model=json_data.get("model"),
                        )
                    elif json_data.get("type") == "message_end":
                        usage_d = json_data.get("usage") or {}
                        res = StreamChatResponseMessageEnd(
                            id=json_data.get("id", ""),
                            usage=ChatUsage(
                                prompt_tokens=usage_d.get("prompt_tokens", 0),
                                completion_tokens=usage_d.get("completion_tokens", 0),
                                total_tokens=usage_d.get("total_tokens", 0),
                            ),
                            model=json_data.get("model"),
                        )

                    yield res
        except Exception as e:
            raise ValueError(f"Error in chat completions streaming: {e}") from e

    # -----------------------------------------------------------------------
    # Chat completions (OpenAI-compatible)
    # -----------------------------------------------------------------------

    def chat_completions(
        self,
        messages: Union[List[Message], List[RawMessage]],
        filter: Optional[dict[str, Any]] = None,
        stream: bool = False,
        model: Union[str, None] = None,
        temperature: Optional[float] = None,
    ) -> Union[ChatCompletionResponse, Iterable[StreamingChatCompletionChunk]]:
        """Chat completions with this assistant (OpenAI-compatible format)."""
        if model is None:
            model = "gpt-4o"
        messages = self._parse_messages(messages)

        if stream:
            return self._chat_completions_streaming(
                messages=messages, model=model, filter=filter, temperature=temperature
            )
        return self._chat_completions_single(
            messages=messages, model=model, filter=filter, temperature=temperature
        )

    def _chat_completions_single(
        self,
        messages: List[Message],
        model: str = "gpt-4o",
        filter: Optional[dict[str, Any]] = None,
        temperature: Optional[float] = None,
    ) -> ChatCompletionResponse:
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]

        kwargs: dict[str, Any] = {"messages": msg_dicts, "model": model}
        if filter:
            kwargs["filter"] = filter
        if temperature is not None:
            kwargs["temperature"] = temperature

        result = self._assistant_data_api.chat_completion_assistant(
            assistant_name=self.name, **kwargs
        )
        return ChatCompletionResponse.from_openapi(result)  # type: ignore[return-value]

    def _chat_completions_streaming(
        self,
        messages: List[Message],
        model: str = "gpt-4o",
        filter: Optional[dict[str, Any]] = None,
        temperature: Optional[float] = None,
    ) -> Iterable[StreamingChatCompletionChunk]:
        api_key = self.config.api_key
        base_url = f"{self.host}/chat/{self.name}/chat/completions"
        headers = {"api-key": api_key, "Content-Type": "application/json"}
        content: dict[str, Any] = {
            "messages": [vars(message) for message in messages],
            "stream": True,
            "model": model,
        }
        if filter:
            content["filter"] = filter
        if temperature is not None:
            content["temperature"] = temperature

        try:
            response = requests.post(
                base_url, headers=headers, json=content, timeout=60, stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = line.decode("utf-8")
                    data = data.removeprefix("data:")

                    json_data = json.loads(data)
                    res = StreamingChatCompletionChunk.from_dict(json_data)

                    yield res
        except Exception as e:
            raise ValueError(f"Error in chat completions streaming: {e}") from e

    # -----------------------------------------------------------------------
    # Context (RAG)
    # -----------------------------------------------------------------------

    def context(
        self,
        query: Optional[str] = None,
        messages: Optional[Union[List[Message], List[RawMessage]]] = None,
        filter: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
        snippet_size: Optional[int] = None,
        multimodal: Optional[bool] = None,
        include_binary_content: Optional[bool] = None,
    ) -> ContextResponse:
        """Retrieve context snippets from this assistant."""
        if not ((not query and messages) or (not messages and query)):
            return ValueError("Invalid Inputs: Exactly one of query or messages must be inputted.")  # type: ignore[return-value]

        kwargs: dict[str, Any] = {}
        if messages:
            parsed = self._parse_messages(messages)
            kwargs["messages"] = [{"role": m.role, "content": m.content} for m in parsed]
        else:
            kwargs["query"] = query

        if filter:
            kwargs["filter"] = filter
        if top_k is not None:
            kwargs["top_k"] = top_k
        if snippet_size is not None:
            kwargs["snippet_size"] = snippet_size
        if multimodal is not None:
            kwargs["multimodal"] = multimodal
        if include_binary_content is not None:
            kwargs["include_binary_content"] = include_binary_content

        raw_response = self._assistant_data_api.context_assistant(
            assistant_name=self.name,
            **kwargs,
        )
        return ContextResponse.from_openapi(raw_response)


__all__ = ["API_VERSION", "HOST_SUFFIX", "MODELS", "AssistantModel"]
