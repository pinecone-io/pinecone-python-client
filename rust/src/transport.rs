use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tonic::transport::Channel;

use crate::proto::vector_service_client::VectorServiceClient;
use crate::proto;

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

/// A gRPC channel wrapper exposed to Python.
#[pyclass]
pub struct GrpcChannel {
    client: VectorServiceClient<Channel>,
    runtime: tokio::runtime::Runtime,
}

#[pymethods]
impl GrpcChannel {
    /// Create a new gRPC channel connected to the given endpoint.
    ///
    /// Args:
    ///     endpoint: The gRPC endpoint URL (e.g. "https://my-index-abc123.svc.pinecone.io:443")
    #[new]
    fn new(endpoint: &str) -> PyResult<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create tokio runtime: {e}")))?;

        let channel = runtime
            .block_on(Channel::from_shared(endpoint.to_string())
                .map_err(|e| PyRuntimeError::new_err(format!("Invalid endpoint: {e}")))?
                .connect())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to connect: {e}")))?;

        Ok(Self {
            client: VectorServiceClient::new(channel),
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
}
