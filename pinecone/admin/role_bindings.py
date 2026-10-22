"""RoleBindings namespace: list, create, describe, delete."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from urllib.parse import quote

from pinecone._internal.adapters.admin_adapter import AdminAdapter
from pinecone._internal.role_bindings import binding_to_payload
from pinecone._internal.validation import require_non_empty, require_one_of
from pinecone.errors.exceptions import ValidationError
from pinecone.models.admin.role_binding import (
    PrincipalType,
    ResourceType,
    RoleBindingInput,
    RoleBindingModel,
    RoleName,
)
from pinecone.models.pagination import Page, Paginator

if TYPE_CHECKING:
    from pinecone._internal.http_client import HTTPClient

logger = logging.getLogger(__name__)

_VALID_PRINCIPAL_TYPES = [p.value for p in PrincipalType]
_VALID_RESOURCE_TYPES = [r.value for r in ResourceType]
_VALID_ROLE_NAMES = [r.value for r in RoleName]


class RoleBindings:
    """The whole of Pinecone's authorization model.

    A role binding grants one ``role`` to one principal (a user, service
    account, API key, or pending invite) at one scope, either the organization
    or a single project. Nothing else confers permissions, so this namespace is
    where a principal's access is read and changed. The other admin namespaces
    carry no role bindings in their models; :meth:`list` with
    ``principal_type`` and ``principal_id`` is how to read a principal's
    access. Reach it as ``admin.role_bindings``; do not construct it directly.

    Bindings are immutable: there is no update. To change a principal's role,
    :meth:`create` the new binding first and then :meth:`delete` the old one.
    In the other order the server refuses the delete when it is the user's
    last organization role and they hold other bindings, and removes the user
    from the organization when it is their only binding.

    The server decides which role may be bound to which scope and principal
    type, and which roles an organization's plan includes. The SDK checks only
    that a value is one this release knows about and that the filter
    co-requirements hold; the server's error message names the role, the
    scope, and, for plan limits, the plan.

    See
    `Error handling <https://sdk.pinecone.io/python/guides/error-handling.html>`__
    for the exceptions every operation here
    can raise.

    Examples:
        >>> from pinecone import Admin
        >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
        >>> for binding in admin.role_bindings.list():
        ...     print(binding.principal_type, binding.role, binding.resource_type)
        user OrgMember organization

    .. seealso::
       - :class:`~pinecone.admin.users.Users`,
         :class:`~pinecone.admin.service_accounts.ServiceAccounts`, and
         :class:`~pinecone.admin.invites.Invites`: the principals bindings point
         at. Each is identified here by its own ``id`` as ``principal_id``, and
         ``principal_type`` says which kind it is.
    """

    def __init__(self, *, http: HTTPClient) -> None:
        self._http = http
        self._adapter = AdminAdapter()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "RoleBindings()"

    def list(
        self,
        *,
        principal_type: str | PrincipalType | None = None,
        principal_id: str | None = None,
        resource_type: str | ResourceType | None = None,
        resource_id: str | None = None,
        role: str | RoleName | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> Paginator[RoleBindingModel]:
        """List the organization's role bindings, with lazy pagination.

        Every supplied filter is combined with **AND**, so
        ``list(principal_type="user", role="OrgOwner")`` returns the bindings
        that are both. With no filters at all it walks every binding the caller
        is allowed to see, which for an org owner is the organization's entire
        authorization state.

        No request is sent until the returned paginator is iterated; see
        `Pagination <https://sdk.pinecone.io/python/guides/pagination.html>`__.
        The filters and *limit* are carried onto every
        later page, so a cursor is always replayed with the query that produced
        it.

        Args:
            principal_type (str | PrincipalType | None): Restrict to one kind of
                principal: ``"user"``, ``"service_account"``, ``"api_key"``, or
                ``"invite"``. Required whenever *principal_id* is given, since an
                ID alone is ambiguous across principal kinds. Omitted when
                ``None``.
            principal_id (str | None): Restrict to one principal's bindings, by
                the principal's UUID. Requires *principal_type*. Omitted when
                ``None``.
            resource_type (str | ResourceType | None): Restrict to one scope
                kind: ``"organization"`` or ``"project"``. Required whenever
                *resource_id* is given. Omitted when ``None``.
            resource_id (str | None): Restrict to one organization or project.
                Requires *resource_type*. Omitted when ``None``.
            role (str | RoleName | None): Restrict to one role, spelled as the
                wire name (``"ProjectOwner"``, not ``"project_owner"``).
                :class:`~pinecone.models.admin.role_binding.RoleName` members are
                accepted interchangeably. Omitted when ``None``.
            limit (int | None): Number of bindings the server returns **per
                page**. It caps each page, not how many bindings the paginator
                yields in total; the paginator keeps following cursors until the
                pages run out, so use :func:`itertools.islice` to cap the total.
                When ``None`` the server chooses the page size.
            pagination_token (str | None): Cursor from a previous paginator's
                ``pagination_token``, to resume where that iteration stopped.
                Reuse it with the same filters and *limit*.

        Returns:
            :class:`~pinecone.models.pagination.Paginator` yielding
            :class:`~pinecone.models.admin.role_binding.RoleBindingModel`
            objects, each carrying the ``id`` that :meth:`delete` needs.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *principal_id* is given without *principal_type*, or
                *resource_id* without *resource_type*; or if *principal_type*,
                *resource_type*, or *role* names a value this SDK release does
                not know. Raised before any network call.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> for binding in admin.role_bindings.list():
            ...     print(binding.principal_type, binding.role, binding.resource_type)
            user OrgMember organization

            Each binding is one principal, one role, and one scope. Filters
            narrow which bindings come back:

            >>> everything_one_service_account_can_do = admin.role_bindings.list(
            ...     principal_type="service_account",
            ...     principal_id="f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
            ... ).to_list()
            >>> project_owners = admin.role_bindings.list(
            ...     resource_type="project",
            ...     resource_id="a2f7dddb-1597-4eff-9f71-535fde243f58",
            ...     role="ProjectOwner",
            ... ).to_list()

            Page-level access exposes the cursor, which is ``None`` once there
            is no further page to fetch:

            >>> for page in admin.role_bindings.list(limit=25).pages():
            ...     print(len(page.items), page.pagination_token)
            1 None

        .. seealso::
           - :meth:`delete`: takes the ``id`` of a binding found here; there is
             no way to revoke by principal, scope, and role.
        """
        if principal_id is not None and principal_type is None:
            raise ValidationError(
                "principal_id requires principal_type: a principal ID is ambiguous on its "
                "own, so the server rejects the pair. Pass principal_type as well, one of "
                f"{', '.join(repr(p) for p in _VALID_PRINCIPAL_TYPES)}."
            )
        if resource_id is not None and resource_type is None:
            raise ValidationError(
                "resource_id requires resource_type: a resource ID is ambiguous on its own, "
                "so the server rejects the pair. Pass resource_type as well, one of "
                f"{', '.join(repr(r) for r in _VALID_RESOURCE_TYPES)}."
            )

        if principal_type is not None:
            require_one_of("principal_type", principal_type, _VALID_PRINCIPAL_TYPES)
        if resource_type is not None:
            require_one_of("resource_type", resource_type, _VALID_RESOURCE_TYPES)
        if role is not None:
            require_one_of("role", role, _VALID_ROLE_NAMES)

        filters: dict[str, str] = {}
        if principal_type is not None:
            filters["principal_type"] = principal_type
        if principal_id is not None:
            filters["principal_id"] = principal_id
        if resource_type is not None:
            filters["resource_type"] = resource_type
        if resource_id is not None:
            filters["resource_id"] = resource_id
        if role is not None:
            filters["role"] = role

        logger.info("Listing role bindings (filters=%r, limit=%r)", sorted(filters), limit)

        def fetch_page(token: str | None) -> Page[RoleBindingModel]:
            params: dict[str, str | int] = dict(filters)
            if limit is not None:
                params["limit"] = limit
            if token is not None:
                params["paginationToken"] = token
            response = self._http.get("/admin/role-bindings", params=params)
            result = self._adapter.to_role_binding_list(response.content)
            logger.debug("Listed %d role bindings (has_more=%s)", len(result), result.has_more)
            return Page(items=result.data, pagination_token=result.pagination_token)

        return Paginator(fetch_page=fetch_page, initial_token=pagination_token)

    def create(
        self,
        *,
        principal_type: str | PrincipalType,
        principal_id: str,
        resource_type: str | ResourceType,
        role: str | RoleName,
        resource_id: str | None = None,
    ) -> RoleBindingModel:
        """Grant a role to a principal at an organization or project scope.

        The binding takes effect immediately and comes back carrying the ``id``
        :meth:`delete` needs to revoke it; bindings cannot be edited in place.
        A user must already belong to the organization: to add someone new,
        invite them with :meth:`Invites.create()
        <pinecone.admin.invites.Invites.create>`. The same scope-and-role pair
        is accepted as an
        initial binding by :meth:`Invites.create()
        <pinecone.admin.invites.Invites.create>` and
        :meth:`ServiceAccounts.create() <pinecone.admin.service_accounts.ServiceAccounts.create>`.

        Args:
            principal_type (str | PrincipalType): The kind of principal receiving
                the role: ``"user"``, ``"service_account"``, ``"api_key"``, or
                ``"invite"``. Binding to an ``invite`` grants the role to whoever
                accepts it; once it is accepted the server refuses further
                bindings on the invite, so manage the roles on the resulting user
                instead. An ``api_key`` binding must be project-scoped, to the
                key's own project.
            principal_id (str): The principal's UUID.
            resource_type (str | ResourceType): The scope: ``"organization"`` or
                ``"project"``.
            role (str | RoleName): The role to grant, spelled as the wire name
                (``"DataPlaneEditor"``).
                :class:`~pinecone.models.admin.role_binding.RoleName` members are
                accepted interchangeably. Which roles are allowed depends on the
                scope, the principal type, and the plan; see the note below.
            resource_id (str | None): The project UUID. Required when
                *resource_type* is ``"project"``. For ``"organization"`` scope
                leave it unset: the server uses the organization the credentials
                belong to.

        Returns:
            The created
            :class:`~pinecone.models.admin.role_binding.RoleBindingModel`, whose
            ``resource_id`` is always populated: an organization-scoped binding
            comes back carrying the organization the credentials resolved to,
            even though the request omitted it.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeTypeError`: If *principal_id* is not a
                ``str``.
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *principal_id* is empty; if *principal_type*, *resource_type*,
                or *role* names a value this SDK release does not know; or if
                *resource_type* is ``"project"`` and *resource_id* is missing.
                Raised before any network call.
            :exc:`~pinecone.errors.exceptions.NotFoundError`:
                If the principal or the resource does not exist in the caller's
                organization, including a user who is not yet a member.
            :exc:`~pinecone.errors.exceptions.ConflictError`:
                If an identical binding already exists (the grant is already in
                force, so this is usually safe to treat as success), or the
                principal is an invite that has already been accepted, in which
                case bind the role to the resulting user.
            :exc:`~pinecone.errors.exceptions.ForbiddenError`:
                If the caller may not create bindings at that scope, would be
                granting a permission it does not itself hold, or the
                organization's plan does not include the role.
            :exc:`~pinecone.errors.exceptions.ApiError`:
                With status 400 if the role cannot be bound at that scope or to
                that principal type.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> binding = admin.role_bindings.create(
            ...     principal_type="user",
            ...     principal_id="e2e92523-85dc-4142-b8c2-e681be8b78df",
            ...     resource_type="organization",
            ...     role="OrgMember",
            ... )
            >>> binding.principal_type, binding.role, binding.resource_type
            ('user', 'OrgMember', 'organization')

            The grant comes back with its own ``id``, and with ``resource_id``
            filled in even though an organization-scoped request omits it:

            >>> bool(binding.id)
            True
            >>> bool(binding.resource_id)
            True

            A project-scoped grant, with enums:

            >>> from pinecone.models.admin import PrincipalType, ResourceType, RoleName
            >>> binding = admin.role_bindings.create(
            ...     principal_type=PrincipalType.SERVICE_ACCOUNT,
            ...     principal_id="f8a3b2c1-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
            ...     resource_type=ResourceType.PROJECT,
            ...     resource_id="a2f7dddb-1597-4eff-9f71-535fde243f58",
            ...     role=RoleName.DATA_PLANE_EDITOR,
            ... )

        .. note::
            The server decides whether a grant is allowed. A project-scoped
            binding must name a project role, an ``api_key`` principal accepts
            only the roles a key can hold (see
            :class:`~pinecone.models.admin.api_key.APIKeyRole`), some roles
            depend on the organization's plan, and the caller cannot grant a
            permission it does not itself hold. Each rejection names the role
            and the scope, and a plan rejection names the plan, so read the
            message rather than checking the rules in advance.

        .. seealso::
           - :meth:`delete`: the second half of a role change, which must run
             after this call rather than before it.
        """
        require_one_of("principal_type", principal_type, _VALID_PRINCIPAL_TYPES)
        require_non_empty("principal_id", principal_id)

        scope = RoleBindingInput(resource_type=resource_type, role=role, resource_id=resource_id)
        body: dict[str, str] = {
            "principal_type": principal_type,
            "principal_id": principal_id,
            **binding_to_payload(scope),
        }

        logger.info(
            "Creating role binding (principal_type=%r, resource_type=%r, role=%r)",
            principal_type,
            scope.resource_type,
            scope.role,
        )
        response = self._http.post("/admin/role-bindings", json=body)
        result = self._adapter.to_role_binding(response.content)
        logger.debug("Created role binding %r", result.id)
        return result

    def describe(self, *, role_binding_id: str) -> RoleBindingModel:
        """Get one role binding's details.

        Args:
            role_binding_id (str): The binding's own UUID, from :meth:`list` or a
                :meth:`create` result (not the principal's ID and not the
                project's).

        Returns:
            A :class:`~pinecone.models.admin.role_binding.RoleBindingModel` with
            the principal, the scope, the role, and when it was granted.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeTypeError`: If *role_binding_id* is not a
                ``str``.
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *role_binding_id* is empty.
            :exc:`~pinecone.errors.exceptions.NotFoundError`:
                If no such role binding is visible to the caller. A binding in
                another organization, or a project binding the caller cannot
                see, returns the same error as one that does not exist, so do
                not read this as proof the binding is gone.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> binding = admin.role_bindings.describe(
            ...     role_binding_id="9a8e3528-b9c0-4358-84ce-84c28e91b566"
            ... )
            >>> binding.principal_type, binding.role, binding.resource_type
            ('user', 'OrgMember', 'organization')
        """
        require_non_empty("role_binding_id", role_binding_id)
        logger.info("Describing role binding %r", role_binding_id)
        response = self._http.get(f"/admin/role-bindings/{quote(role_binding_id, safe='')}")
        result = self._adapter.to_role_binding(response.content)
        logger.debug("Described role binding %r", role_binding_id)
        return result

    def delete(self, *, role_binding_id: str) -> None:
        """Revoke a role binding, by the binding's own ID.

        Deletion is addressed by ``role_binding_id`` rather than by the
        principal/scope/role triple, so revoking a role means finding the binding
        first, usually with :meth:`list` filtered by ``principal_type`` and
        ``principal_id``, or from the :meth:`create` result. The permissions are
        revoked immediately. A repeat of this call raises
        :exc:`~pinecone.errors.exceptions.NotFoundError`.

        Args:
            role_binding_id (str): The binding's own UUID.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeTypeError`: If *role_binding_id* is not a
                ``str``.
            :exc:`~pinecone.errors.exceptions.PineconeValueError`:
                If *role_binding_id* is empty.
            :exc:`~pinecone.errors.exceptions.NotFoundError`:
                If no such role binding is visible to the caller, including a
                repeat of a successful delete.
            :exc:`~pinecone.errors.exceptions.ConflictError`:
                If deleting the binding would leave the organization without an
                owner, remove a user's last organization role while they hold
                other bindings, or remove a pending invite's last organization
                role; or if the organization's users are managed by an identity
                provider. Grant the replacement binding first, or make the change
                in the identity provider.

        Examples:
            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> admin.role_bindings.delete(
            ...     role_binding_id="9a8e3528-b9c0-4358-84ce-84c28e91b566"
            ... )

        .. note::
            Deleting a user's only binding removes the user from the
            organization. To withdraw a pending invite, use
            :meth:`Invites.delete() <pinecone.admin.invites.Invites.delete>`
            rather than deleting its bindings.

        .. seealso::
           - :meth:`create`: run it *before* this call when changing a role.
        """
        require_non_empty("role_binding_id", role_binding_id)
        logger.info("Deleting role binding %r", role_binding_id)
        self._http.delete(f"/admin/role-bindings/{quote(role_binding_id, safe='')}")
        logger.debug("Deleted role binding %r", role_binding_id)
