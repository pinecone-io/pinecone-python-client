use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tonic::service::interceptor::InterceptedService;
use tonic::transport::{Channel, ClientTlsConfig};

use crate::proto::vector_service_client::VectorServiceClient;
use crate::proto;

/// Interceptor that attaches API key, request ID (UUID v4), and API version
/// metadata to every outgoing gRPC request.
#[derive(Clone)]
struct MetadataInterceptor {
    api_key: tonic::metadata::MetadataValue<tonic::metadata::Ascii>,
    api_version: tonic::metadata::MetadataValue<tonic::metadata::Ascii>,
}

impl MetadataInterceptor {
    fn new(api_key: &str, api_version: &str) -> Result<Self, tonic::metadata::errors::InvalidMetadataValue> {
        Ok(Self {
            api_key: api_key.parse()?,
            api_version: api_version.parse()?,
        })
    }
}

impl tonic::service::Interceptor for MetadataInterceptor {
    fn call(&mut self, mut request: tonic::Request<()>) -> Result<tonic::Request<()>, tonic::Status> {
        let metadata = request.metadata_mut();
        metadata.insert("api-key", self.api_key.clone());
        metadata.insert(
            "x-request-id",
            uuid::Uuid::new_v4()
                .to_string()
                .parse()
                .map_err(|_| tonic::Status::internal("failed to create request ID"))?,
        );
        metadata.insert("x-pinecone-api-version", self.api_version.clone());
        Ok(request)
    }
}

/// Convert a `tonic::Status` into a Python `RuntimeError`.
fn status_to_py_err(status: tonic::Status) -> PyErr {
    PyRuntimeError::new_err(format!(
        "gRPC error ({}): {}",
        status.code(),
        status.message()
    ))
}

/// Convert a `prost_types::Struct` to a Python dict.
fn struct_to_py_dict(py: Python<'_>, s: &prost_types::Struct) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    for (key, value) in &s.fields {
        dict.set_item(key, prost_value_to_py(py, value)?)?;
    }
    Ok(dict.unbind())
}

/// Convert a `prost_types::Value` to a Python object.
fn prost_value_to_py(py: Python<'_>, value: &prost_types::Value) -> PyResult<PyObject> {
    use prost_types::value::Kind;
    match &value.kind {
        Some(Kind::NullValue(_)) => Ok(py.None()),
        Some(Kind::NumberValue(n)) => Ok(n.into_pyobject(py)?.into_any().unbind()),
        Some(Kind::StringValue(s)) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        Some(Kind::BoolValue(b)) => Ok(b.into_pyobject(py)?.to_owned().into_any().unbind()),
        Some(Kind::StructValue(s)) => Ok(struct_to_py_dict(py, s)?.into_any()),
        Some(Kind::ListValue(list)) => {
            let items: Vec<PyObject> = list
                .values
                .iter()
                .map(|v| prost_value_to_py(py, v))
                .collect::<PyResult<_>>()?;
            Ok(pyo3::types::PyList::new(py, items)?.into_any().unbind())
        }
        None => Ok(py.None()),
    }
}

/// Convert a Python object to a `prost_types::Value`.
fn py_to_prost_value(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<prost_types::Value> {
    use prost_types::value::Kind;

    if obj.is_none() {
        return Ok(prost_types::Value {
            kind: Some(Kind::NullValue(0)),
        });
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(prost_types::Value {
            kind: Some(Kind::BoolValue(b)),
        });
    }
    if let Ok(n) = obj.extract::<f64>() {
        return Ok(prost_types::Value {
            kind: Some(Kind::NumberValue(n)),
        });
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(prost_types::Value {
            kind: Some(Kind::StringValue(s)),
        });
    }
    if let Ok(dict) = obj.downcast::<PyDict>() {
        return Ok(prost_types::Value {
            kind: Some(Kind::StructValue(py_dict_to_struct(dict)?)),
        });
    }
    if let Ok(list) = obj.downcast::<pyo3::types::PyList>() {
        let values: Vec<prost_types::Value> = list
            .iter()
            .map(|item| py_to_prost_value(&item))
            .collect::<PyResult<_>>()?;
        return Ok(prost_types::Value {
            kind: Some(Kind::ListValue(prost_types::ListValue { values })),
        });
    }

    Err(PyRuntimeError::new_err(format!(
        "Unsupported metadata value type: {}",
        obj.get_type().name()?
    )))
}

/// Convert a Python dict to a `prost_types::Struct`.
fn py_dict_to_struct(dict: &Bound<'_, PyDict>) -> PyResult<prost_types::Struct> {
    let mut fields = std::collections::BTreeMap::new();
    for (key, value) in dict.iter() {
        let key_str: String = key.extract()?;
        fields.insert(key_str, py_to_prost_value(&value)?);
    }
    Ok(prost_types::Struct { fields })
}

/// Convert a proto SparseValues into a Python dict with "indices" and "values" keys.
fn sparse_values_to_py_dict(
    py: Python<'_>,
    sv: &proto::SparseValues,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("indices", &sv.indices)?;
    dict.set_item("values", &sv.values)?;
    Ok(dict.unbind())
}

/// Extract sparse values from a Python dict.
fn py_dict_to_sparse_values(dict: &Bound<'_, PyDict>) -> PyResult<proto::SparseValues> {
    let indices: Vec<u32> = dict
        .get_item("indices")?
        .ok_or_else(|| PyRuntimeError::new_err("sparse_values missing 'indices'"))?
        .extract()?;
    let values: Vec<f32> = dict
        .get_item("values")?
        .ok_or_else(|| PyRuntimeError::new_err("sparse_values missing 'values'"))?
        .extract()?;
    Ok(proto::SparseValues { indices, values })
}

/// Convert a proto Vector to a Python dict.
fn vector_to_py_dict(py: Python<'_>, v: &proto::Vector) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", &v.id)?;
    dict.set_item("values", &v.values)?;
    if let Some(ref sv) = v.sparse_values {
        dict.set_item("sparse_values", sparse_values_to_py_dict(py, sv)?)?;
    }
    if let Some(ref md) = v.metadata {
        dict.set_item("metadata", struct_to_py_dict(py, md)?)?;
    }
    Ok(dict.unbind())
}

/// Convert a proto ScoredVector to a Python dict.
fn scored_vector_to_py_dict(py: Python<'_>, sv: &proto::ScoredVector) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", &sv.id)?;
    dict.set_item("score", sv.score)?;
    dict.set_item("values", &sv.values)?;
    if let Some(ref sparse) = sv.sparse_values {
        dict.set_item("sparse_values", sparse_values_to_py_dict(py, sparse)?)?;
    }
    if let Some(ref md) = sv.metadata {
        dict.set_item("metadata", struct_to_py_dict(py, md)?)?;
    }
    Ok(dict.unbind())
}

/// Convert a proto `NamespaceDescription` to a Python dict.
fn namespace_description_to_py_dict(
    py: Python<'_>,
    ns: &proto::NamespaceDescription,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("name", &ns.name)?;
    dict.set_item("record_count", ns.record_count)?;
    if let Some(ref schema) = ns.schema {
        let schema_dict = metadata_schema_to_py_dict(py, schema)?;
        dict.set_item("schema", schema_dict)?;
    }
    if let Some(ref indexed) = ns.indexed_fields {
        dict.set_item("indexed_fields", &indexed.fields)?;
    }
    Ok(dict.unbind())
}

/// Convert a proto `MetadataSchema` to a Python dict.
fn metadata_schema_to_py_dict(
    py: Python<'_>,
    schema: &proto::MetadataSchema,
) -> PyResult<Py<PyDict>> {
    let fields_dict = PyDict::new(py);
    for (name, props) in &schema.fields {
        let props_dict = PyDict::new(py);
        props_dict.set_item("filterable", props.filterable)?;
        fields_dict.set_item(name, props_dict)?;
    }
    let dict = PyDict::new(py);
    dict.set_item("fields", fields_dict)?;
    Ok(dict.unbind())
}

/// Convert a Python dict to a proto `MetadataSchema`.
fn py_dict_to_metadata_schema(dict: &Bound<'_, PyDict>) -> PyResult<proto::MetadataSchema> {
    let fields_obj = dict
        .get_item("fields")?
        .ok_or_else(|| PyRuntimeError::new_err("schema missing 'fields'"))?;
    let fields_dict = fields_obj.downcast::<PyDict>()?;
    let mut fields = std::collections::HashMap::new();
    for (key, value) in fields_dict.iter() {
        let key_str: String = key.extract()?;
        let props_dict = value.downcast::<PyDict>()?;
        let filterable: bool = props_dict
            .get_item("filterable")?
            .ok_or_else(|| PyRuntimeError::new_err("field properties missing 'filterable'"))?
            .extract()?;
        fields.insert(
            key_str,
            proto::MetadataFieldProperties { filterable },
        );
    }
    Ok(proto::MetadataSchema { fields })
}

/// A gRPC channel wrapper exposed to Python.
#[pyclass]
pub struct GrpcChannel {
    client: VectorServiceClient<InterceptedService<Channel, MetadataInterceptor>>,
    runtime: tokio::runtime::Runtime,
}

#[pymethods]
impl GrpcChannel {
    /// Create a new gRPC channel connected to the given endpoint.
    ///
    /// Args:
    ///     endpoint: The gRPC endpoint URL (e.g. "https://my-index-abc123.svc.pinecone.io:443")
    ///     api_key: The Pinecone API key for authentication.
    ///     api_version: The Pinecone API version string (e.g. "2025-10").
    ///     secure: Whether to use TLS encryption (default true).
    #[new]
    #[pyo3(signature = (endpoint, api_key, api_version, secure=true))]
    fn new(endpoint: &str, api_key: &str, api_version: &str, secure: bool) -> PyResult<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create tokio runtime: {e}")))?;

        let mut endpoint_builder = Channel::from_shared(endpoint.to_string())
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid endpoint: {e}")))?;

        if secure {
            endpoint_builder = endpoint_builder
                .tls_config(ClientTlsConfig::new())
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to configure TLS: {e}")))?;
        }

        let channel = runtime
            .block_on(endpoint_builder.connect())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to connect: {e}")))?;

        let interceptor = MetadataInterceptor::new(api_key, api_version)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid metadata value: {e}")))?;

        Ok(Self {
            client: VectorServiceClient::with_interceptor(channel, interceptor),
            runtime,
        })
    }

    /// Upsert vectors.
    ///
    /// Args:
    ///     vectors: List of dicts with keys: id, values, sparse_values (optional), metadata (optional)
    ///     namespace: Target namespace (default "")
    ///
    /// Returns:
    ///     Dict with "upserted_count".
    #[pyo3(signature = (vectors, namespace=None))]
    fn upsert(
        &self,
        py: Python<'_>,
        vectors: Vec<Bound<'_, PyDict>>,
        namespace: Option<&str>,
    ) -> PyResult<Py<PyDict>> {
        let mut proto_vectors = Vec::with_capacity(vectors.len());
        for v in &vectors {
            let id: String = v
                .get_item("id")?
                .ok_or_else(|| PyRuntimeError::new_err("vector missing 'id'"))?
                .extract()?;
            let values: Vec<f32> = v
                .get_item("values")?
                .ok_or_else(|| PyRuntimeError::new_err("vector missing 'values'"))?
                .extract()?;
            let sparse_values = match v.get_item("sparse_values")? {
                Some(sv) => Some(py_dict_to_sparse_values(&sv.downcast_into::<PyDict>()?)?),
                None => None,
            };
            let metadata = match v.get_item("metadata")? {
                Some(md) => Some(py_dict_to_struct(&md.downcast_into::<PyDict>()?)?),
                None => None,
            };
            proto_vectors.push(proto::Vector {
                id,
                values,
                sparse_values,
                metadata,
            });
        }

        let request = proto::UpsertRequest {
            vectors: proto_vectors,
            namespace: namespace.unwrap_or("").to_string(),
        };

        let mut client = self.client.clone();
        #[allow(clippy::result_large_err)]
        let response = py
            .allow_threads(|| self.runtime.block_on(client.upsert(request)))
            .map_err(status_to_py_err)?;

        let inner = response.into_inner();
        let dict = PyDict::new(py);
        dict.set_item("upserted_count", inner.upserted_count)?;
        Ok(dict.unbind())
    }

    /// Query vectors.
    ///
    /// Args:
    ///     top_k: Number of results to return.
    ///     vector: Query vector (optional).
    ///     id: Query by vector ID (optional).
    ///     namespace: Namespace to query (default "").
    ///     filter: Metadata filter dict (optional).
    ///     include_values: Include vector values in response (default false).
    ///     include_metadata: Include metadata in response (default false).
    ///
    /// Returns:
    ///     Dict with "matches" (list of scored vector dicts) and "namespace".
    #[pyo3(signature = (top_k, vector=None, id=None, namespace=None, filter=None, include_values=false, include_metadata=false))]
    #[allow(clippy::too_many_arguments)]
    fn query(
        &self,
        py: Python<'_>,
        top_k: u32,
        vector: Option<Vec<f32>>,
        id: Option<&str>,
        namespace: Option<&str>,
        filter: Option<Bound<'_, PyDict>>,
        include_values: bool,
        include_metadata: bool,
    ) -> PyResult<Py<PyDict>> {
        #[allow(deprecated)]
        let request = proto::QueryRequest {
            namespace: namespace.unwrap_or("").to_string(),
            top_k,
            filter: filter.map(|f| py_dict_to_struct(&f)).transpose()?,
            include_values,
            include_metadata,
            queries: vec![],
            vector: vector.unwrap_or_default(),
            sparse_vector: None,
            id: id.unwrap_or("").to_string(),
            scan_factor: None,
            max_candidates: None,
        };

        let mut client = self.client.clone();
        #[allow(clippy::result_large_err)]
        let response = py
            .allow_threads(|| self.runtime.block_on(client.query(request)))
            .map_err(status_to_py_err)?;

        let inner = response.into_inner();
        let matches: Vec<Py<PyDict>> = inner
            .matches
            .iter()
            .map(|m| scored_vector_to_py_dict(py, m))
            .collect::<PyResult<_>>()?;

        let dict = PyDict::new(py);
        dict.set_item("matches", matches)?;
        dict.set_item("namespace", &inner.namespace)?;
        if let Some(usage) = &inner.usage {
            let usage_dict = PyDict::new(py);
            usage_dict.set_item("read_units", usage.read_units)?;
            dict.set_item("usage", usage_dict)?;
        }
        Ok(dict.unbind())
    }

    /// Fetch vectors by ID.
    ///
    /// Args:
    ///     ids: List of vector IDs to fetch.
    ///     namespace: Namespace (default "").
    ///
    /// Returns:
    ///     Dict with "vectors" (map of id → vector dict) and "namespace".
    #[pyo3(signature = (ids, namespace=None))]
    fn fetch(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
        namespace: Option<&str>,
    ) -> PyResult<Py<PyDict>> {
        let request = proto::FetchRequest {
            ids,
            namespace: namespace.unwrap_or("").to_string(),
        };

        let mut client = self.client.clone();
        #[allow(clippy::result_large_err)]
        let response = py
            .allow_threads(|| self.runtime.block_on(client.fetch(request)))
            .map_err(status_to_py_err)?;

        let inner = response.into_inner();
        let vectors_dict = PyDict::new(py);
        for (id, vector) in &inner.vectors {
            vectors_dict.set_item(id, vector_to_py_dict(py, vector)?)?;
        }

        let dict = PyDict::new(py);
        dict.set_item("vectors", vectors_dict)?;
        dict.set_item("namespace", &inner.namespace)?;
        if let Some(usage) = &inner.usage {
            let usage_dict = PyDict::new(py);
            usage_dict.set_item("read_units", usage.read_units)?;
            dict.set_item("usage", usage_dict)?;
        }
        Ok(dict.unbind())
    }

    /// Delete vectors.
    ///
    /// Args:
    ///     ids: List of vector IDs to delete (optional).
    ///     delete_all: Delete all vectors in namespace (default false).
    ///     namespace: Namespace (default "").
    ///     filter: Metadata filter dict (optional).
    ///
    /// Returns:
    ///     Empty dict (delete has no response fields).
    #[pyo3(signature = (ids=None, delete_all=false, namespace=None, filter=None))]
    fn delete(
        &self,
        py: Python<'_>,
        ids: Option<Vec<String>>,
        delete_all: bool,
        namespace: Option<&str>,
        filter: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDict>> {
        let request = proto::DeleteRequest {
            ids: ids.unwrap_or_default(),
            delete_all,
            namespace: namespace.unwrap_or("").to_string(),
            filter: filter.map(|f| py_dict_to_struct(&f)).transpose()?,
        };

        let mut client = self.client.clone();
        #[allow(clippy::result_large_err)]
        py.allow_threads(|| self.runtime.block_on(client.delete(request)))
            .map_err(status_to_py_err)?;

        let dict = PyDict::new(py);
        Ok(dict.unbind())
    }

    /// Update a vector.
    ///
    /// Args:
    ///     id: The unique ID of the vector to update.
    ///     values: New dense vector values (optional).
    ///     sparse_values: New sparse vector values dict with "indices" and "values" keys (optional).
    ///     set_metadata: Metadata dict to set/overwrite (optional).
    ///     namespace: Namespace (default "").
    ///     filter: Metadata filter for bulk update (optional).
    ///     dry_run: If true, return matched count without executing (optional).
    ///
    /// Returns:
    ///     Dict with optional "matched_records" count.
    #[pyo3(signature = (id, values=None, sparse_values=None, set_metadata=None, namespace=None, filter=None, dry_run=None))]
    #[allow(clippy::too_many_arguments)]
    fn update(
        &self,
        py: Python<'_>,
        id: &str,
        values: Option<Vec<f32>>,
        sparse_values: Option<Bound<'_, PyDict>>,
        set_metadata: Option<Bound<'_, PyDict>>,
        namespace: Option<&str>,
        filter: Option<Bound<'_, PyDict>>,
        dry_run: Option<bool>,
    ) -> PyResult<Py<PyDict>> {
        let request = proto::UpdateRequest {
            id: id.to_string(),
            values: values.unwrap_or_default(),
            sparse_values: sparse_values
                .map(|sv| py_dict_to_sparse_values(&sv))
                .transpose()?,
            set_metadata: set_metadata
                .map(|md| py_dict_to_struct(&md))
                .transpose()?,
            namespace: namespace.unwrap_or("").to_string(),
            filter: filter.map(|f| py_dict_to_struct(&f)).transpose()?,
            dry_run,
        };

        let mut client = self.client.clone();
        #[allow(clippy::result_large_err)]
        let response = py
            .allow_threads(|| self.runtime.block_on(client.update(request)))
            .map_err(status_to_py_err)?;

        let inner = response.into_inner();
        let dict = PyDict::new(py);
        if let Some(matched) = inner.matched_records {
            dict.set_item("matched_records", matched)?;
        }
        Ok(dict.unbind())
    }

    /// List vector IDs.
    ///
    /// Args:
    ///     prefix: ID prefix filter (optional).
    ///     limit: Max number of IDs to return (optional).
    ///     pagination_token: Token to continue a previous listing (optional).
    ///     namespace: Namespace (default "").
    ///
    /// Returns:
    ///     Dict with "vectors" (list of dicts with "id"), optional "pagination" dict,
    ///     "namespace", and optional "usage" dict.
    #[pyo3(signature = (prefix=None, limit=None, pagination_token=None, namespace=None))]
    fn list(
        &self,
        py: Python<'_>,
        prefix: Option<&str>,
        limit: Option<u32>,
        pagination_token: Option<&str>,
        namespace: Option<&str>,
    ) -> PyResult<Py<PyDict>> {
        let request = proto::ListRequest {
            prefix: prefix.map(|s| s.to_string()),
            limit,
            pagination_token: pagination_token.map(|s| s.to_string()),
            namespace: namespace.unwrap_or("").to_string(),
        };

        let mut client = self.client.clone();
        #[allow(clippy::result_large_err)]
        let response = py
            .allow_threads(|| self.runtime.block_on(client.list(request)))
            .map_err(status_to_py_err)?;

        let inner = response.into_inner();
        let vectors: Vec<Py<PyDict>> = inner
            .vectors
            .iter()
            .map(|item| {
                let d = PyDict::new(py);
                d.set_item("id", &item.id)?;
                Ok(d.unbind())
            })
            .collect::<PyResult<_>>()?;

        let dict = PyDict::new(py);
        dict.set_item("vectors", vectors)?;
        if let Some(ref pag) = inner.pagination {
            let pag_dict = PyDict::new(py);
            pag_dict.set_item("next", &pag.next)?;
            dict.set_item("pagination", pag_dict)?;
        }
        dict.set_item("namespace", &inner.namespace)?;
        if let Some(ref usage) = inner.usage {
            let usage_dict = PyDict::new(py);
            usage_dict.set_item("read_units", usage.read_units)?;
            dict.set_item("usage", usage_dict)?;
        }
        Ok(dict.unbind())
    }

    /// Get index statistics.
    ///
    /// Args:
    ///     filter: Metadata filter dict (optional). If present, stats reflect only
    ///             vectors matching the filter.
    ///
    /// Returns:
    ///     Dict with "namespaces" (map of namespace → {"vector_count"}), "dimension",
    ///     "index_fullness", "total_vector_count", and optional "metric", "vector_type",
    ///     "memory_fullness", "storage_fullness".
    #[pyo3(signature = (filter=None))]
    fn describe_index_stats(
        &self,
        py: Python<'_>,
        filter: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDict>> {
        let request = proto::DescribeIndexStatsRequest {
            filter: filter.map(|f| py_dict_to_struct(&f)).transpose()?,
        };

        let mut client = self.client.clone();
        #[allow(clippy::result_large_err)]
        let response = py
            .allow_threads(|| self.runtime.block_on(client.describe_index_stats(request)))
            .map_err(status_to_py_err)?;

        let inner = response.into_inner();
        let namespaces_dict = PyDict::new(py);
        for (name, summary) in &inner.namespaces {
            let ns_dict = PyDict::new(py);
            ns_dict.set_item("vector_count", summary.vector_count)?;
            namespaces_dict.set_item(name, ns_dict)?;
        }

        let dict = PyDict::new(py);
        dict.set_item("namespaces", namespaces_dict)?;
        if let Some(dim) = inner.dimension {
            dict.set_item("dimension", dim)?;
        }
        dict.set_item("index_fullness", inner.index_fullness)?;
        dict.set_item("total_vector_count", inner.total_vector_count)?;
        if let Some(ref metric) = inner.metric {
            dict.set_item("metric", metric)?;
        }
        if let Some(ref vt) = inner.vector_type {
            dict.set_item("vector_type", vt)?;
        }
        if let Some(mf) = inner.memory_fullness {
            dict.set_item("memory_fullness", mf)?;
        }
        if let Some(sf) = inner.storage_fullness {
            dict.set_item("storage_fullness", sf)?;
        }
        Ok(dict.unbind())
    }

    /// List namespaces.
    ///
    /// Args:
    ///     pagination_token: Token to continue a previous listing (optional).
    ///     limit: Max number of namespaces to return (optional).
    ///     prefix: Namespace prefix filter (optional).
    ///
    /// Returns:
    ///     Dict with "namespaces" (list of namespace description dicts),
    ///     optional "pagination" dict, and "total_count".
    #[pyo3(signature = (pagination_token=None, limit=None, prefix=None))]
    fn list_namespaces(
        &self,
        py: Python<'_>,
        pagination_token: Option<&str>,
        limit: Option<u32>,
        prefix: Option<&str>,
    ) -> PyResult<Py<PyDict>> {
        let request = proto::ListNamespacesRequest {
            pagination_token: pagination_token.map(|s| s.to_string()),
            limit,
            prefix: prefix.map(|s| s.to_string()),
        };

        let mut client = self.client.clone();
        #[allow(clippy::result_large_err)]
        let response = py
            .allow_threads(|| self.runtime.block_on(client.list_namespaces(request)))
            .map_err(status_to_py_err)?;

        let inner = response.into_inner();
        let namespaces: Vec<Py<PyDict>> = inner
            .namespaces
            .iter()
            .map(|ns| namespace_description_to_py_dict(py, ns))
            .collect::<PyResult<_>>()?;

        let dict = PyDict::new(py);
        dict.set_item("namespaces", namespaces)?;
        if let Some(ref pag) = inner.pagination {
            let pag_dict = PyDict::new(py);
            pag_dict.set_item("next", &pag.next)?;
            dict.set_item("pagination", pag_dict)?;
        }
        dict.set_item("total_count", inner.total_count)?;
        Ok(dict.unbind())
    }

    /// Describe a namespace.
    ///
    /// Args:
    ///     namespace: The namespace to describe.
    ///
    /// Returns:
    ///     Dict with "name", "record_count", and optional "schema" and "indexed_fields".
    #[pyo3(signature = (namespace))]
    fn describe_namespace(
        &self,
        py: Python<'_>,
        namespace: &str,
    ) -> PyResult<Py<PyDict>> {
        let request = proto::DescribeNamespaceRequest {
            namespace: namespace.to_string(),
        };

        let mut client = self.client.clone();
        #[allow(clippy::result_large_err)]
        let response = py
            .allow_threads(|| self.runtime.block_on(client.describe_namespace(request)))
            .map_err(status_to_py_err)?;

        let inner = response.into_inner();
        namespace_description_to_py_dict(py, &inner)
    }

    /// Delete a namespace.
    ///
    /// Args:
    ///     namespace: The namespace to delete.
    ///
    /// Returns:
    ///     Empty dict.
    #[pyo3(signature = (namespace))]
    fn delete_namespace(
        &self,
        py: Python<'_>,
        namespace: &str,
    ) -> PyResult<Py<PyDict>> {
        let request = proto::DeleteNamespaceRequest {
            namespace: namespace.to_string(),
        };

        let mut client = self.client.clone();
        #[allow(clippy::result_large_err)]
        py.allow_threads(|| self.runtime.block_on(client.delete_namespace(request)))
            .map_err(status_to_py_err)?;

        let dict = PyDict::new(py);
        Ok(dict.unbind())
    }

    /// Create a namespace.
    ///
    /// Args:
    ///     name: The name of the namespace to create.
    ///     schema: Optional metadata schema dict with "fields" mapping field names
    ///             to {"filterable": bool}.
    ///
    /// Returns:
    ///     Dict with "name", "record_count", and optional "schema" and "indexed_fields".
    #[pyo3(signature = (name, schema=None))]
    fn create_namespace(
        &self,
        py: Python<'_>,
        name: &str,
        schema: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDict>> {
        let metadata_schema = schema
            .map(|s| py_dict_to_metadata_schema(&s))
            .transpose()?;

        let request = proto::CreateNamespaceRequest {
            name: name.to_string(),
            schema: metadata_schema,
        };

        let mut client = self.client.clone();
        #[allow(clippy::result_large_err)]
        let response = py
            .allow_threads(|| self.runtime.block_on(client.create_namespace(request)))
            .map_err(status_to_py_err)?;

        let inner = response.into_inner();
        namespace_description_to_py_dict(py, &inner)
    }

    /// Fetch vectors by metadata filter.
    ///
    /// Args:
    ///     namespace: Namespace to fetch from (default "").
    ///     filter: Metadata filter dict (optional).
    ///     limit: Max number of vectors to return (optional).
    ///     pagination_token: Token to continue a previous listing (optional).
    ///
    /// Returns:
    ///     Dict with "vectors" (map of id → vector dict), "namespace",
    ///     optional "usage" dict, and optional "pagination" dict.
    #[pyo3(signature = (namespace=None, filter=None, limit=None, pagination_token=None))]
    fn fetch_by_metadata(
        &self,
        py: Python<'_>,
        namespace: Option<&str>,
        filter: Option<Bound<'_, PyDict>>,
        limit: Option<u32>,
        pagination_token: Option<&str>,
    ) -> PyResult<Py<PyDict>> {
        let request = proto::FetchByMetadataRequest {
            namespace: namespace.unwrap_or("").to_string(),
            filter: filter.map(|f| py_dict_to_struct(&f)).transpose()?,
            limit,
            pagination_token: pagination_token.map(|s| s.to_string()),
        };

        let mut client = self.client.clone();
        #[allow(clippy::result_large_err)]
        let response = py
            .allow_threads(|| self.runtime.block_on(client.fetch_by_metadata(request)))
            .map_err(status_to_py_err)?;

        let inner = response.into_inner();
        let vectors_dict = PyDict::new(py);
        for (id, vector) in &inner.vectors {
            vectors_dict.set_item(id, vector_to_py_dict(py, vector)?)?;
        }

        let dict = PyDict::new(py);
        dict.set_item("vectors", vectors_dict)?;
        dict.set_item("namespace", &inner.namespace)?;
        if let Some(ref usage) = inner.usage {
            let usage_dict = PyDict::new(py);
            usage_dict.set_item("read_units", usage.read_units)?;
            dict.set_item("usage", usage_dict)?;
        }
        if let Some(ref pag) = inner.pagination {
            let pag_dict = PyDict::new(py);
            pag_dict.set_item("next", &pag.next)?;
            dict.set_item("pagination", pag_dict)?;
        }
        Ok(dict.unbind())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use tonic::service::Interceptor;

    #[test]
    fn interceptor_attaches_all_metadata_headers() {
        let mut interceptor = MetadataInterceptor::new("test-api-key-123", "2025-10").unwrap();
        let request = tonic::Request::new(());
        let result = interceptor.call(request).unwrap();
        let metadata = result.metadata();

        assert_eq!(
            metadata.get("api-key").unwrap().to_str().unwrap(),
            "test-api-key-123"
        );
        assert_eq!(
            metadata.get("x-pinecone-api-version").unwrap().to_str().unwrap(),
            "2025-10"
        );

        let request_id = metadata.get("x-request-id").unwrap().to_str().unwrap();
        // Validate UUID v4 format (8-4-4-4-12 hex chars)
        assert_eq!(request_id.len(), 36);
        assert!(uuid::Uuid::parse_str(request_id).is_ok());
    }

    #[test]
    fn tls_enabled_endpoint_builder_succeeds() {
        // Verify that configuring TLS on a valid endpoint does not error
        let endpoint = Channel::from_shared("https://example.pinecone.io:443".to_string())
            .expect("valid endpoint");
        let result = endpoint.tls_config(ClientTlsConfig::new());
        assert!(result.is_ok(), "TLS config should succeed on a valid endpoint");
    }

    #[test]
    fn insecure_endpoint_builder_succeeds() {
        // Verify that creating an endpoint without TLS config does not error
        let endpoint = Channel::from_shared("http://localhost:5080".to_string());
        assert!(endpoint.is_ok(), "Insecure endpoint should be constructable");
    }

    #[test]
    fn each_call_gets_distinct_request_id() {
        let mut interceptor = MetadataInterceptor::new("key", "2025-10").unwrap();
        let mut ids = HashSet::new();

        for _ in 0..100 {
            let request = tonic::Request::new(());
            let result = interceptor.call(request).unwrap();
            let request_id = result
                .metadata()
                .get("x-request-id")
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            ids.insert(request_id);
        }

        assert_eq!(ids.len(), 100, "All 100 request IDs should be unique");
    }
}
