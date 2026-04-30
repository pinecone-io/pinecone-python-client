"""Priority-5 smoke test — async assistants surface.

Mirror of ``test_assistants_sync.py`` against ``AsyncPinecone``.

Punchlist coverage (async): full AsyncAssistants surface.
"""

from __future__ import annotations

import pytest

from pinecone import AsyncPinecone
from tests.smoke.conftest import (
    SMOKE_PREFIX,
    async_cleanup_resource,
    async_poll_until,
    unique_name,
)
from tests.smoke.helpers import (
    SAMPLE_PDF_SMALL,
    SAMPLE_TEXT_FILE,
)


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_assistants_smoke_async(api_key: str) -> None:
    pc = AsyncPinecone(api_key=api_key)
    name = unique_name(f"{SMOKE_PREFIX}-asst-async")
    try:
        # ----- create -----
        created = await pc.assistants.create(
            name=name,
            instructions="You are an async smoke-test assistant. Reply briefly.",
        )
        assert created.name == name

        # ----- poll until Ready -----
        async def status_check() -> str:
            d = await pc.assistants.describe(name=name)
            return d.status

        await async_poll_until(
            status_check,
            lambda s: s == "Ready",
            timeout=120,
            interval=3,
            description=f"async assistant {name}",
        )

        # ----- describe -----
        described = await pc.assistants.describe(name=name)
        assert described.status == "Ready"

        # ----- list (paginator) -----
        listing = await pc.assistants.list().to_list()
        assert any(a.name == name for a in listing)

        # ----- list_page -----
        first_page = await pc.assistants.list_page(page_size=20)
        # confirm structure even if our name isn't on the first page
        assert hasattr(first_page, "assistants")

        # ----- update -----
        updated = await pc.assistants.update(
            name=name,
            instructions="You are an updated async smoke-test assistant.",
        )
        assert updated.name == name

        # ----- file uploads -----
        text_file = await pc.assistants.upload_file(
            assistant_name=name,
            file_path=str(SAMPLE_TEXT_FILE),
        )
        assert text_file.status == "Available"

        pdf_file = await pc.assistants.upload_file(
            assistant_name=name,
            file_path=str(SAMPLE_PDF_SMALL),
        )
        assert pdf_file.status == "Available"

        multimodal_file = await pc.assistants.upload_file(
            assistant_name=name,
            file_path=str(SAMPLE_PDF_SMALL),
            multimodal=True,
        )
        assert multimodal_file.status == "Available"

        # ----- describe_file / list_files / list_files_page -----
        d_file = await pc.assistants.describe_file(
            assistant_name=name, file_id=text_file.id
        )
        assert d_file.id == text_file.id

        all_files = await pc.assistants.list_files(assistant_name=name).to_list()
        assert len(all_files) >= 3

        files_page = await pc.assistants.list_files_page(assistant_name=name)
        assert len(files_page.files) >= 1

        # ----- context -----
        ctx = await pc.assistants.context(
            assistant_name=name,
            query="What's in the test files?",
            top_k=3,
        )
        assert hasattr(ctx, "snippets")

        # ----- chat (non-streaming) -----
        chat_resp = await pc.assistants.chat(
            assistant_name=name,
            messages=[{"content": "Reply with the single word OK."}],
        )
        assert chat_resp.message is not None
        assert chat_resp.message.content

        # ----- chat (streaming) -----
        chunks = []
        stream = await pc.assistants.chat(
            assistant_name=name,
            messages=[{"content": "Count to three."}],
            stream=True,
        )
        async for chunk in stream:
            chunks.append(chunk)
        assert len(chunks) > 0

        # ----- chat_completions (non-streaming) -----
        cc_resp = await pc.assistants.chat_completions(
            assistant_name=name,
            messages=[{"content": "Reply with the single word OK."}],
        )
        assert cc_resp.choices is not None
        assert len(cc_resp.choices) > 0

        # ----- chat_completions (streaming) -----
        cc_chunks = []
        cc_stream = await pc.assistants.chat_completions(
            assistant_name=name,
            messages=[{"content": "Count to three."}],
            stream=True,
        )
        async for chunk in cc_stream:
            cc_chunks.append(chunk)
        assert len(cc_chunks) > 0

        # ----- evaluate_alignment -----
        align = await pc.assistants.evaluate_alignment(
            question="What is the capital of France?",
            answer="Paris.",
            ground_truth_answer="Paris is the capital of France.",
        )
        assert align.scores is not None

        # ----- pc.assistant proxy (no warning currently emitted) -----
        via_proxy = await pc.assistant.describe(name=name)
        assert via_proxy.name == name
        via_call = await pc.assistant(name)
        assert via_call.name == name

        # ----- delete_file -----
        await pc.assistants.delete_file(
            assistant_name=name, file_id=text_file.id
        )
    finally:
        await async_cleanup_resource(
            lambda: pc.assistants.delete(name=name),
            name,
            "assistant",
        )
        await pc.assistants.close()
        await pc.close()
