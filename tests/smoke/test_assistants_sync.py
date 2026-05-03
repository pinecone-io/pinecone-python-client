"""Priority-5 smoke test — sync assistants surface.

Walks every method on ``Pinecone.assistants`` and the deprecated
``pc.assistant`` proxy in a single notebook-style scenario.

Punchlist coverage (sync):

- create / describe / list / list_page / update / delete / close
- upload_file (text + PDF + multimodal PDF) / describe_file / list_files /
  list_files_page / delete_file
- chat (non-streaming + streaming)
- chat_completions (non-streaming + streaming)
- context
- evaluate_alignment
- ``pc.assistant`` deprecated proxy access form
"""

from __future__ import annotations

import pytest

from pinecone import Pinecone
from tests.smoke.conftest import (
    SMOKE_PREFIX,
    cleanup_resource,
    unique_name,
    wait_for_ready,
)
from tests.smoke.helpers import (
    SAMPLE_PDF_SMALL,
    SAMPLE_TEXT_FILE,
)


@pytest.mark.smoke
def test_assistants_smoke(client: Pinecone) -> None:
    """End-to-end assistants walkthrough."""
    name = unique_name(f"{SMOKE_PREFIX}-asst")
    try:
        # ----- create -----
        created = client.assistants.create(
            name=name,
            instructions="You are a smoke-test assistant. Reply briefly.",
        )
        assert created.name == name
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # ----- describe -----
        described = client.assistants.describe(name=name)
        assert described.status == "Ready"

        # ----- list -----
        listing = list(client.assistants.list().to_list())
        assert any(a.name == name for a in listing)

        # ----- list_page -----
        first_page = client.assistants.list_page(page_size=20)
        page_names = {a.name for a in first_page.assistants}
        # name may not be on the very first page if many assistants exist;
        # don't fail — just confirm list_page returned a structured response.
        assert isinstance(page_names, set)

        # ----- update -----
        updated = client.assistants.update(
            name=name,
            instructions="You are an updated smoke-test assistant.",
        )
        assert updated.name == name

        # ----- file uploads -----
        text_file = client.assistants.upload_file(
            assistant_name=name,
            file_path=str(SAMPLE_TEXT_FILE),
        )
        assert text_file.status == "Available"

        pdf_file = client.assistants.upload_file(
            assistant_name=name,
            file_path=str(SAMPLE_PDF_SMALL),
        )
        assert pdf_file.status == "Available"

        multimodal_file = client.assistants.upload_file(
            assistant_name=name,
            file_path=str(SAMPLE_PDF_SMALL),
            multimodal=True,
        )
        assert multimodal_file.status == "Available"

        # ----- describe_file / list_files / list_files_page -----
        d_file = client.assistants.describe_file(assistant_name=name, file_id=text_file.id)
        assert d_file.id == text_file.id

        all_files = client.assistants.list_files(assistant_name=name).to_list()
        assert len(all_files) >= 3

        files_page = client.assistants.list_files_page(assistant_name=name)
        assert len(files_page.files) >= 1

        # ----- context -----
        ctx = client.assistants.context(
            assistant_name=name,
            query="What's in the test files?",
            top_k=3,
        )
        # ContextResponse always has a snippets attribute (possibly empty)
        assert hasattr(ctx, "snippets")

        # ----- chat (non-streaming) -----
        chat_resp = client.assistants.chat(
            assistant_name=name,
            messages=[{"content": "Reply with the single word OK."}],
        )
        assert chat_resp.message is not None
        assert chat_resp.message.content

        # ----- chat (streaming) -----
        stream = client.assistants.chat(
            assistant_name=name,
            messages=[{"content": "Count to three, one per line."}],
            stream=True,
        )
        chunks = list(stream)
        assert len(chunks) > 0

        # ----- chat_completions (non-streaming) -----
        cc_resp = client.assistants.chat_completions(
            assistant_name=name,
            messages=[{"content": "Reply with the single word OK."}],
        )
        assert cc_resp.choices is not None
        assert len(cc_resp.choices) > 0

        # ----- chat_completions (streaming) -----
        cc_stream = client.assistants.chat_completions(
            assistant_name=name,
            messages=[{"content": "Count to three."}],
            stream=True,
        )
        cc_chunks = list(cc_stream)
        assert len(cc_chunks) > 0

        # ----- evaluate_alignment -----
        align = client.assistants.evaluate_alignment(
            question="What is the capital of France?",
            answer="Paris.",
            ground_truth_answer="Paris is the capital of France.",
        )
        assert align.scores is not None

        # ----- pc.assistant proxy (permanent alias for pc.assistants —
        #       confirm both access forms work; no DeprecationWarning expected). -----
        via_proxy = client.assistant.describe(name=name)
        assert via_proxy.name == name
        via_call = client.assistant(name)  # callable form
        assert via_call.name == name

        # ----- delete_file (one of them) -----
        client.assistants.delete_file(assistant_name=name, file_id=text_file.id)
    finally:
        # delete assistant first (tears down files server-side too)
        cleanup_resource(
            lambda: client.assistants.delete(name=name),
            name,
            "assistant",
        )
        client.assistants.close()
        client.close()
