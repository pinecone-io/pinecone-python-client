"""Unit tests for the Admin Projects namespace."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import ADMIN_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.admin.projects import Projects
from pinecone.errors.exceptions import NotFoundError, PineconeError, ValidationError
from pinecone.models.admin.project import ProjectList, ProjectModel

BASE_URL = "https://api.test.pinecone.io"


def _project_response(
    *,
    id: str = "proj-abc123",
    name: str = "my-project",
    max_pods: int = 5,
    force_encryption_with_cmek: bool = False,
    organization_id: str = "org-abc123",
    created_at: str = "2025-01-01T00:00:00Z",
) -> dict[str, Any]:
    return {
        "id": id,
        "name": name,
        "max_pods": max_pods,
        "force_encryption_with_cmek": force_encryption_with_cmek,
        "organization_id": organization_id,
        "created_at": created_at,
    }


@pytest.fixture
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, ADMIN_API_VERSION)


@pytest.fixture
def projects(http_client: HTTPClient) -> Projects:
    return Projects(http=http_client)


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------


@respx.mock
def test_list_projects(projects: Projects) -> None:
    respx.get(f"{BASE_URL}/admin/projects").mock(
        return_value=httpx.Response(
            200,
            json={"data": [_project_response()]},
        ),
    )

    result = projects.list()

    assert isinstance(result, ProjectList)
    assert len(result) == 1
    project = result[0]
    assert isinstance(project, ProjectModel)
    assert project.id == "proj-abc123"
    assert project.name == "my-project"
    assert project.max_pods == 5
    assert project.force_encryption_with_cmek is False
    assert project.organization_id == "org-abc123"
    assert project.created_at == "2025-01-01T00:00:00Z"


@respx.mock
def test_list_projects_empty(projects: Projects) -> None:
    respx.get(f"{BASE_URL}/admin/projects").mock(
        return_value=httpx.Response(200, json={"data": []}),
    )

    result = projects.list()

    assert isinstance(result, ProjectList)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------


@respx.mock
def test_create_project(projects: Projects) -> None:
    route = respx.post(f"{BASE_URL}/admin/projects").mock(
        return_value=httpx.Response(
            200,
            json=_project_response(name="new-proj"),
        ),
    )

    result = projects.create(name="new-proj")

    assert isinstance(result, ProjectModel)
    assert result.name == "new-proj"

    # Verify request body contains only name
    request = route.calls[0].request
    expected_body = httpx.Request("POST", "/", json={"name": "new-proj"})
    assert request.content == expected_body.content


@respx.mock
def test_create_project_with_all_options(projects: Projects) -> None:
    route = respx.post(f"{BASE_URL}/admin/projects").mock(
        return_value=httpx.Response(
            200,
            json=_project_response(
                name="new-proj",
                max_pods=10,
                force_encryption_with_cmek=True,
            ),
        ),
    )

    result = projects.create(
        name="new-proj",
        max_pods=10,
        force_encryption_with_cmek=True,
    )

    assert isinstance(result, ProjectModel)
    assert result.name == "new-proj"
    assert result.max_pods == 10
    assert result.force_encryption_with_cmek is True

    # Verify all fields in request body
    request = route.calls[0].request
    expected_body = httpx.Request(
        "POST",
        "/",
        json={
            "name": "new-proj",
            "max_pods": 10,
            "force_encryption_with_cmek": True,
        },
    )
    assert request.content == expected_body.content


def test_create_project_requires_name(projects: Projects) -> None:
    with pytest.raises(ValidationError, match="name"):
        projects.create(name="")


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


@respx.mock
def test_describe_project(projects: Projects) -> None:
    respx.get(f"{BASE_URL}/admin/projects/proj-abc123").mock(
        return_value=httpx.Response(200, json=_project_response()),
    )

    result = projects.describe(project_id="proj-abc123")

    assert isinstance(result, ProjectModel)
    assert result.id == "proj-abc123"
    assert result.name == "my-project"


def test_describe_requires_project_id(projects: Projects) -> None:
    with pytest.raises(ValidationError, match="project_id"):
        projects.describe(project_id="")


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------


@respx.mock
def test_update_project(projects: Projects) -> None:
    route = respx.patch(f"{BASE_URL}/admin/projects/proj-abc123").mock(
        return_value=httpx.Response(
            200,
            json=_project_response(name="updated", max_pods=10),
        ),
    )

    result = projects.update(
        project_id="proj-abc123",
        name="updated",
        max_pods=10,
    )

    assert isinstance(result, ProjectModel)
    assert result.name == "updated"
    assert result.max_pods == 10

    # Verify request body
    request = route.calls[0].request
    expected_body = httpx.Request("PATCH", "/", json={"name": "updated", "max_pods": 10})
    assert request.content == expected_body.content


@respx.mock
def test_update_partial(projects: Projects) -> None:
    route = respx.patch(f"{BASE_URL}/admin/projects/p1").mock(
        return_value=httpx.Response(
            200,
            json=_project_response(id="p1", name="new"),
        ),
    )

    result = projects.update(project_id="p1", name="new")

    assert isinstance(result, ProjectModel)
    assert result.name == "new"

    # Verify only name is in request body (no max_pods or force_encryption_with_cmek)
    request = route.calls[0].request
    expected_body = httpx.Request("PATCH", "/", json={"name": "new"})
    assert request.content == expected_body.content


def test_update_requires_project_id(projects: Projects) -> None:
    with pytest.raises(ValidationError, match="project_id"):
        projects.update(project_id="", name="new")


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


@respx.mock
def test_delete_project(projects: Projects) -> None:
    respx.delete(f"{BASE_URL}/admin/projects/proj-abc123").mock(
        return_value=httpx.Response(202),
    )

    result = projects.delete(project_id="proj-abc123")

    assert result is None


def test_delete_requires_project_id(projects: Projects) -> None:
    with pytest.raises(ValidationError, match="project_id"):
        projects.delete(project_id="")


# ---------------------------------------------------------------------------
# describe_by_name()
# ---------------------------------------------------------------------------


@respx.mock
def test_describe_by_name_single_match(projects: Projects) -> None:
    respx.get(f"{BASE_URL}/admin/projects").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    _project_response(id="p1", name="other"),
                    _project_response(id="p2", name="target"),
                ]
            },
        ),
    )

    result = projects.describe_by_name(name="target")

    assert isinstance(result, ProjectModel)
    assert result.id == "p2"
    assert result.name == "target"


@respx.mock
def test_describe_by_name_no_match(projects: Projects) -> None:
    respx.get(f"{BASE_URL}/admin/projects").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    _project_response(id="p1", name="alpha"),
                    _project_response(id="p2", name="beta"),
                ]
            },
        ),
    )

    with pytest.raises(NotFoundError):
        projects.describe_by_name(name="missing")


@respx.mock
def test_describe_by_name_multiple_matches(projects: Projects) -> None:
    respx.get(f"{BASE_URL}/admin/projects").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    _project_response(id="p1", name="dup"),
                    _project_response(id="p2", name="dup"),
                ]
            },
        ),
    )

    with pytest.raises(PineconeError, match="Multiple projects"):
        projects.describe_by_name(name="dup")


def test_describe_by_name_empty_name(projects: Projects) -> None:
    with pytest.raises(ValidationError, match="name"):
        projects.describe_by_name(name="")


# ---------------------------------------------------------------------------
# exists()
# ---------------------------------------------------------------------------


@respx.mock
def test_exists_by_id_true(projects: Projects) -> None:
    respx.get(f"{BASE_URL}/admin/projects/proj-abc123").mock(
        return_value=httpx.Response(200, json=_project_response()),
    )

    assert projects.exists(project_id="proj-abc123") is True


@respx.mock
def test_exists_by_id_false(projects: Projects) -> None:
    respx.get(f"{BASE_URL}/admin/projects/proj-gone").mock(
        return_value=httpx.Response(404, json={"error": {"message": "not found"}}),
    )

    assert projects.exists(project_id="proj-gone") is False


@respx.mock
def test_exists_by_name_true(projects: Projects) -> None:
    respx.get(f"{BASE_URL}/admin/projects").mock(
        return_value=httpx.Response(
            200,
            json={"data": [_project_response(name="my-project")]},
        ),
    )

    assert projects.exists(name="my-project") is True


@respx.mock
def test_exists_by_name_false(projects: Projects) -> None:
    respx.get(f"{BASE_URL}/admin/projects").mock(
        return_value=httpx.Response(200, json={"data": []}),
    )

    assert projects.exists(name="missing") is False


@respx.mock
def test_exists_by_name_multiple_matches_returns_true(projects: Projects) -> None:
    respx.get(f"{BASE_URL}/admin/projects").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    _project_response(id="proj-1", name="dup"),
                    _project_response(id="proj-2", name="dup"),
                ]
            },
        ),
    )

    assert projects.exists(name="dup") is True


def test_exists_requires_one_param(projects: Projects) -> None:
    with pytest.raises(ValidationError):
        projects.exists(project_id="abc", name="xyz")

    with pytest.raises(ValidationError):
        projects.exists()


# ---------------------------------------------------------------------------
# keyword-only enforcement
# ---------------------------------------------------------------------------


def test_all_methods_keyword_only(projects: Projects) -> None:
    with pytest.raises(TypeError):
        projects.describe("proj-abc123")  # type: ignore[misc]

    with pytest.raises(TypeError):
        projects.create("new-proj")  # type: ignore[misc]

    with pytest.raises(TypeError):
        projects.update("proj-abc123")  # type: ignore[misc]

    with pytest.raises(TypeError):
        projects.delete("proj-abc123")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


def test_repr(projects: Projects) -> None:
    assert repr(projects) == "Projects()"
