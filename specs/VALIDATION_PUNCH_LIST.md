# Spec Validation Punch List

Generated from Phase 1 validation (eval-completeness + eval-accuracy) on 2026-03-17.

## Cross-Cutting Issues

These issues appear across multiple files and should be fixed systematically.

### CX-1: False `*,` keyword-only signatures
Methods that use `@require_kwargs` decorator are documented with `*,` in `### Signature` code blocks, but the actual Python signatures don't have `*`. The decorator enforces keyword-only at runtime, not via syntax.

**Affected files:**
- admin_api_key_operations.md (all 7 methods)
- admin_organization_operations.md (all 7 methods)
- admin_project_operations.md (all 8 methods)
- collection_operations.md (create_collection, delete_collection, describe_collection — sync only)
- index_operations.md (create_index, create_index_for_model — sync and async; configure_index — sync and async)

**Fix:** Remove `*,` from signatures where the actual source code doesn't have it. Keep `*,` only for methods that genuinely use keyword-only syntax in the source (e.g., backup/restore methods at pinecone.py do use `*`).

### CX-2: False ValueError claims for input validation
Several specs claim methods raise `ValueError` for empty names or length limits, but validation is server-side (results in `PineconeApiException`), not client-side.

**Affected files:**
- admin_api_key_operations.md — create() ValueError for name length
- admin_project_operations.md — create() ValueError for name length
- collection_operations.md — create/delete/describe ValueError for empty name
- index_operations.md — configure_index ValueError for deletion_protection (check if this one is real)

**Fix:** Remove false ValueError entries from Raises tables. Keep only exceptions that the client code actually raises.

### CX-3: Backup status value "Initialized" vs "Initializing"
The OpenAPI model docstring uses "Initializing" but specs say "Initialized".

**Affected files:**
- backup_operations.md
- backup_management_operations.md

**Fix:** Change "Initialized" to "Initializing" in all status references.

### CX-4: Missing async variants
Several files document sync methods but omit async equivalents.

**Affected files:**
- index_data_operations.md — missing IndexAsyncio.upsert(), IndexAsyncio.delete()
- index_namespace_operations.md — missing all async namespace methods

**Fix:** Add async variant documentation for each missing method.

---

## Per-File Issues

### admin_api_key_operations.md
- [ ] **CX-1**: Remove `*,` from all method signatures
- [ ] **CX-2**: Remove false ValueError from create() Raises
- [ ] Add Raises tables to `Admin.api_key.get()` and `Admin.api_key.describe()` aliases
- [ ] Fix APIKey.id field description — remove "pckey_" format reference (that's the value format, not the id format)

### admin_organization_operations.md
- [ ] **CX-1**: Remove `*,` from all method signatures
- [ ] Fix Organization.created_at type: should be `datetime` not `string (date-time)` (source uses Python datetime)
- [ ] Fix update() idempotency: change from "Not idempotent" to "Idempotent" (same input -> same output)

### admin_project_operations.md
- [ ] **CX-1**: Remove `*,` from all method signatures
- [ ] **CX-2**: Remove false ValueError from create() Raises
- [ ] Fix Project.created_at type: should be `datetime` not `string (date-time)`
- [ ] Add PineconeApiException to list() Raises table for consistency
- [ ] Fix fetch/get/describe/exists parameter defaults: show `None` instead of `—` for project_id and name

### backup_operations.md
- [ ] **CX-3**: Change "Initialized" to "Initializing" in status references
- [ ] Fix BackupModel optional fields: name, description, created_at, dimension, metric should be marked nullable (they're optional in OpenAPI model)
- [ ] Add ForbiddenException to create_backup Raises (both sync and async)
- [ ] Add ForbiddenException + NotFoundException to async list_backups Raises

### backup_management_operations.md
- [ ] **CX-3**: Change "Initialized" to "Initializing" in status references
- [ ] Add ForbiddenException to async describe_backup and async delete_backup Raises
- [ ] Fix delete_backup idempotency contradiction (claims idempotent but also lists NotFoundException)

### collection_operations.md
- [ ] **CX-1**: Remove `*,` from create/delete/describe signatures (sync methods)
- [ ] **CX-2**: Remove false ValueError from create/delete/describe Raises
- [ ] Add CollectionList.__str__() and __repr__() methods
- [ ] Add response field table to async describe_collection
- [ ] Make async Raises tables consistent with sync versions (add ForbiddenException, NotFoundException)

### index_client_access_operations.md
- [ ] Add pool_threads and connection_pool_maxsize parameters to Pinecone.Index() parameter table

### index_data_operations.md (CRITICAL)
- [ ] **CX-4**: Add IndexAsyncio.upsert() async variant
- [ ] **CX-4**: Add IndexAsyncio.delete() async variant
- [ ] Add Index.update() method (pinecone/db_data/index.py:1222-1371)
- [ ] Add Index.upsert_from_dataframe() method (pinecone/db_data/index.py:483-567)
- [ ] Add Index.upsert_records() method (pinecone/db_data/index.py:569-658)
- [ ] Add async variants for update, upsert_from_dataframe, upsert_records

### index_management_operations.md
- [ ] Fix IndexModel nullable fields: embed, private_host, tags should be Nullable: Yes
- [ ] Add IndexModel.__getitem__(), __str__(), __repr__() methods
- [ ] Add IndexList.__str__(), __repr__() methods (already missing from source)

### index_namespace_operations.md (CRITICAL)
- [ ] **CX-4**: Add all async namespace resource methods (create, describe, delete, list, list_paginated)
- [ ] Add Index.namespace.list_paginated() method
- [ ] Add convenience methods: Index.create_namespace(), describe_namespace(), delete_namespace(), list_namespaces(), list_namespaces_paginated()
- [ ] Add all IndexAsyncio convenience methods
- [ ] Fix namespace.list() return type description (yields individual items, not full response pages)

### index_operations.md
- [ ] **CX-1**: Remove `*,` from create_index and create_index_for_model signatures (sync and async)
- [ ] Fix configure_index read_capacity type: expand from `dict | None` to full union type
- [ ] Fix type unions: some params are `(X | str) | None` in source but shown without outer `| None`

### index_read_operations.md
- [ ] Add _response_info field to FetchResponse and FetchByMetadataResponse data models (minor — semi-private field)

### index_search_and_query_operations.md
- [ ] Add formal SearchRecordsResponse data model table
- [ ] Fix SearchRerank.model type: should be `str` not `str | RerankModel` (per actual annotation)
- [ ] Add full method documentation for data model methods (as_dict, to_dict, from_dict)

### inference_operations.md (CRITICAL)
- [ ] Add Inference.list_models() method (pinecone/inference/inference.py:348-384)
- [ ] Add Inference.get_model() method (pinecone/inference/inference.py:386-426)
- [ ] Add AsyncioInference.list_models() async variant
- [ ] Add AsyncioInference.get_model() async variant
- [ ] Add ModelInfo data model
- [ ] Add ModelInfoList data model
- [ ] Formalize EmbeddingsList data model with proper field table (Nullable/Since/Deprecated columns)

### restore_operations.md
- [ ] Fix async create_index_from_backup deletion_protection type: add `| None` to match source
- [ ] Add full method documentation for RestoreJobModel.to_dict()

---

## Priority Order

1. **Critical completeness gaps** — index_data (5 missing methods), index_namespace (12+ missing), inference (4 missing methods + 3 models)
2. **Cross-cutting CX-1** — False `*,` signatures (affects 5 files, ~30 methods)
3. **Cross-cutting CX-2** — False ValueError claims (affects 3 files)
4. **Cross-cutting CX-3** — Backup status naming (affects 2 files)
5. **Cross-cutting CX-4** — Missing async variants (affects 2 files, partially overlaps with critical gaps)
6. **Per-file detail issues** — nullable fields, missing Raises entries, type corrections
